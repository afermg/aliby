#!/usr/bin/env jupyter
# TODO should this be merged to the regular logfile_parser structure?
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
 -

New grammar

- Tables are assumed to end with an empty line.
"""

import logging
import typing as t
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


class HeaderEndNotFound(Exception):
    def __init__(self, message, errors):
        super().__init__(message)

        self.errors = errors


def extract_header(filepath: PosixPath):
    # header_contents = ""
    with open(filepath, "r") as f:
        try:
            header = ""
            for _ in range(MAX_NLINES):
                line = f.readline()
                header += line
                if HEADER_END in line:
                    break
        except HeaderEndNotFound as e:
            print(f"{MAX_NLINES} checked and no header found")
            raise (e)
        return header


def parse_table(
    string: str,
    start_trigger: t.Union[str, Keyword],
) -> pd.DataFrame:
    """Parse csv-like table

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
    line = LineStart() + Group(
        OneOrMore(field + Literal(",").suppress()) + field + EOL
    )
    parser = (
        start_trigger
        + EOL
        + Group(OneOrMore(line))
        + EOL  # end_trigger.suppress()
    )
    parser_result = parser.search_string(string)

    assert all(
        [len(row) == len(parser_result[0]) for row in parser_result]
    ), f"Table {start_trigger} has unequal number of columns"

    assert len(parser_result), f"Parsing is empty. {parser}"
    return table_to_df(parser_result.as_list())


def parse_fields(
    string: str, start_trigger, end_trigger=None
) -> t.Union[pd.DataFrame, t.Dict[str, atomic]]:
    """
    Fields are parsed as key: value

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
    parser = (
        start_trigger + EOL + Group(OneOrMore(line)) + end_trigger.suppress()
    )
    parser_result = parser.search_string(string)
    results = parser_result.as_list()
    assert len(results), "Parsing returned nothing"
    return fields_to_dict_or_table(results)


# Grammar specification
grammar = {
    "general": {
        "start_trigger": Literal("Swain Lab microscope experiment log file"),
        "type": "fields",
        "end_trigger": "-----Acquisition settings-----",
    },
    "image_config": {
        "start_trigger": "Image Configs:",
        "type": "table",
    },
    "device_properties": {
        "start_trigger": "Device properties:",
        "type": "table",
    },
    "group": {
        "position": {
            "start_trigger": Group(
                Group(Literal("group:") + Word(printables))
                + Group(Literal("field:") + "position")
            ),
            "type": "table",
        },
        **{
            key: {
                "start_trigger": Group(
                    Group(Literal("group:") + Word(printables))
                    + Group(Literal("field:") + key)
                ),
                "type": "fields",
            }
            for key in ("time", "config")
        },
    },
}


ACQ_START = "-----Acquisition settings-----"
HEADER_END = "-----Experiment started-----"
MAX_NLINES = 2000  # In case of malformed logfile
# test_file = "/home/alan/Downloads/pH_med_to_low.log"
# test_file = "/home/alan/Documents/dev/skeletons/scripts/dev/C1_60x.log"

ParserElement.setDefaultWhitespaceChars(" \t")


# time_fields = parse_field(acq, start_trigger=grammar["group"]["time"]["start_trigger"])
# config_fields = parse_fields(
#     acq, start_trigger=grammar["group"]["config"]["start_trigger"]
# )

# general_fields = parse_fields(basic, start_trigger=grammar["general"]["start_trigger"])


def parse_from_grammar(filepath: str, grammar: t.Dict):
    header = extract_header(filepath)
    d = {}
    for key, values in grammar.items():
        try:
            if "type" in values:
                d[key] = parse_x(header, **values)
            else:  # Use subkeys to parse groups
                for subkey, subvalues in values.items():
                    subkey = "_".join((key, subkey))
                    d[subkey] = parse_x(header, **subvalues)
        except Exception as e:
            logging.getLogger("aliby").critical(
                f"Parsing failed for key {key}"
            )
            raise (e)
    return d


def table_to_df(result: t.List[t.List]):
    if len(result) > 1:  # Multiple tables with ids to append
        # Generate multiindex from "Name column"
        # index = [row[1][0][1] for table in result for row in table]
        # table[1][0].index("Name") # for automatic indexing
        from itertools import product

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
    else:  # If it is a single table
        df = pd.DataFrame(result[0][1][1:], columns=result[0][1][0])

    return df


def fields_to_dict_or_table(result: t.List[t.List]):
    if len(result) > 1:
        formatted = pd.DataFrame(
            [[row[1] for row in pr[1]] for pr in result],
            columns=[x[0] for x in result[0][1]],
            index=[x[0][0][1] for x in result],
        )

        formatted.name = result[0][0][1][1]

    else:  # If it is a single table
        formatted = {k: _cast_type(v) for k, v in dict(result[0][1]).items()}

    return formatted


def _cast_type(x: str) -> t.Union[str, int, float, bool]:
    # Convert to any possible when possible
    x = x.strip()
    if x.isdigit():
        x = int(x)
    else:
        try:
            x = float(x)
        except:
            try:
                x = ("false", "true").index(x.lower())
            except:
                pass
    return x


def parse_x(string: str, type: str, **kwargs):
    # return eval(f"parse_{type}({string}, **{kwargs})")
    return eval(f"parse_{type}(string, **kwargs)")


def parse_from_swainlab_grammar(filepath: t.Union[str, PosixPath]):
    return parse_from_grammar(filepath, grammar)
