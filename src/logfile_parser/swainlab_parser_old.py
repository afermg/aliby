#!/usr/bin/env jupyter
"""
Description of new logfile:

All three conditions are concatenated in a single file, in this order:
 - Experiment basic information  (URL in acquisition PC, project, user input)
 - Acquisition settings
 - Experiment start

The section separators are:
-----Acquisition settings-----
-----Experiment started-----

And for a successfully finished experiment we get:

YYYY-MM-DD HH:mm:ss,ms*3 Image acquisition complete WeekDay Mon Day  HH:mm:ss,ms*3 YYYY

For example:
2022-09-30 05:40:59,765 Image acquisition complete Fri Sep 30 05:40:59 2022

Data to extract:
* Basic information
 - Experiment details, which may indicate technical issues
 -  GIT commit
 - (Not working as of 2022/10/03, but projects and tags)
* Basic information
"""

import logging
import typing as t
from itertools import product
from pathlib import PosixPath

import pandas as pd
from pyparsing import (
    CharsNotIn,
    Combine,
    Group,
    Keyword,
    LineEnd,
    LineStart,
    Literal,
    OneOrMore,
    ParserElement,
    Word,
    printables,
)

atomic = t.Union[str, int, float, bool]

# grammar specification
sl_grammar = {
    "general": {
        "type": "fields",
        "start_trigger": Literal("Swain Lab microscope experiment log file"),
        "end_trigger": "-----Acquisition settings-----",
    },
    "image_config": {
        "type": "table",
        "start_trigger": "Image Configs:",
    },
    "device_properties": {
        "type": "table",
        "start_trigger": "Device properties:",
    },
    "group": {
        "position": {
            "type": "table",
            "start_trigger": Group(
                Group(Literal("group:") + Word(printables))
                + Group(Literal("field:") + "position")
            ),
        },
        **{
            key: {
                "type": "fields",
                "start_trigger": Group(
                    Group(Literal("group:") + Word(printables))
                    + Group(Literal("field:") + key)
                ),
            }
            for key in ("time", "config")
        },
    },
}


HEADER_END = "-----Experiment started-----"
MAX_NLINES = 2000  # in case of malformed logfile

ParserElement.setDefaultWhitespaceChars(" \t")


def extract_header(filepath: PosixPath, **kwargs):
    """Extract content of log file upto HEADER_END."""
    with open(filepath, "r", **kwargs) as f:
        try:
            header = ""
            for _ in range(MAX_NLINES):
                line = f.readline()
                header += line
                if HEADER_END in line:
                    break
        except Exception as e:
            raise e(f"{MAX_NLINES} checked and no header found")
        return header


def parse_from_swainlab_grammar(filepath: t.Union[str, PosixPath]):
    """Parse using a grammar for the Swain lab."""
    try:
        header = extract_header(filepath, encoding="latin1")
        res = parse_from_grammar(header, sl_grammar)
    except Exception:
        # removes unwanted windows characters
        header = extract_header(filepath, errors="ignore", encoding="unicode_escape")
        res = parse_from_grammar(header, sl_grammar)
    return res


def parse_from_grammar(header: str, grammar: t.Dict):
    """Parse a string using the specified grammar."""
    d = {}
    for key, values in grammar.items():
        try:
            if "type" in values:
                # use values to find parsing function
                d[key] = parse_x(header, **values)
            else:
                # for group, use subkeys to parse
                for subkey, subvalues in values.items():
                    subkey = "_".join((key, subkey))
                    # use subvalues to find parsing function
                    d[subkey] = parse_x(header, **subvalues)
        except Exception as e:
            logging.getLogger("aliby").critical(
                f"Parsing failed for key {key} and values {values}"
            )
            raise (e)
    return d


def parse_x(string_input: str, type: str, **kwargs):
    """Parse a string using a function specifed by data_type."""
    return eval(f"parse_{type}(string_input, **kwargs)")


def parse_table(
    string_input: str,
    start_trigger: t.Union[str, Keyword],
) -> pd.DataFrame:
    """
    Parse csv-like table.

    Parameters
    ----------
    string : str
        contents to parse
    start_trigger : t.Union[str, t.Collection]
        string or triggers that indicate section start.

    Returns
    -------
    pd.Dataframe or dict of atomic values (int,str,bool,float)
        DataFrame representing table.

    Examples
    --------
    >>> table = parse_table()
    """
    if isinstance(start_trigger, str):
        start_trigger: Keyword = Keyword(start_trigger)
    EOL = LineEnd().suppress()
    field = OneOrMore(CharsNotIn(":,\n"))
    line = LineStart() + Group(OneOrMore(field + Literal(",").suppress()) + field + EOL)
    parser = start_trigger + EOL + Group(OneOrMore(line)) + EOL
    parser_result = parser.search_string(string_input)
    assert all([len(row) == len(parser_result[0]) for row in parser_result]), (
        f"Table {start_trigger} has unequal number of columns"
    )
    assert len(parser_result), f"Parsing is empty. {parser}"
    return table_to_df(parser_result.as_list())


def parse_fields(
    string_input: str, start_trigger, end_trigger=None
) -> t.Union[pd.DataFrame, t.Dict[str, atomic]]:
    """
    Parse fields are parsed as key-value pairs.

    By default the end is an empty newline.

    For example

    group: YST_1510 field: time
    start: 0
    interval: 300
    frames: 180
    """
    EOL = LineEnd().suppress()
    if end_trigger is None:
        end_trigger = EOL
    elif isinstance(end_trigger, str):
        end_trigger = Literal(end_trigger)
    field = OneOrMore(CharsNotIn(":\n"))
    line = (
        LineStart()
        + Group(field + Combine(OneOrMore(Literal(":").suppress() + field)))
        + EOL
    )
    parser = start_trigger + EOL + Group(OneOrMore(line)) + end_trigger.suppress()
    parser_result = parser.search_string(string_input)
    results = parser_result.as_list()
    assert len(results), "Parsing returned nothing"
    return fields_to_dict_or_df(results)


def table_to_df(result: t.List[t.List]):
    """Convert table to data frame."""
    if len(result) > 1:
        # multiple tables with ids to append
        group_name = [
            product((table[0][0][1],), (row[0] for row in table[1][1:]))
            for table in result
        ]
        tmp = [pair for pairset in group_name for pair in pairset]
        multiindices = pd.MultiIndex.from_tuples(tmp)
        df = pd.DataFrame(
            [row for pr in result for row in pr[1][1:]],
            columns=result[0][1][0],
            index=multiindices,
        )
        df.name = result[0][0][1][1]
    else:
        # a single table
        df = pd.DataFrame(result[0][1][1:], columns=result[0][1][0])
    return df


def fields_to_dict_or_df(result: t.List[t.List]):
    """Convert field to dict or dataframe."""
    if len(result) > 1:
        formatted = pd.DataFrame(
            [[row[1] for row in pr[1]] for pr in result],
            columns=[x[0] for x in result[0][1]],
            index=[x[0][0][1] for x in result],
        )
        formatted.name = result[0][0][1][1]
    else:
        # a single table
        formatted = {k: cast_type(v) for k, v in dict(result[0][1]).items()}
    return formatted


def cast_type(x: str) -> t.Union[str, int, float, bool]:
    """Convert to either an integer or float or boolean."""
    x = x.strip()
    if x.isdigit():
        x = int(x)
    else:
        try:
            x = float(x)
        except Exception:
            try:
                x = ("false", "true").index(x.lower())
            except Exception:
                pass
    return x
