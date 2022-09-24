# -*- coding: utf-8 -*-

import json
import pkgutil
import re
import typing as t
from datetime import datetime
from os.path import dirname, exists, join

CONFIG_KEY = "@@CONFIG@@"
DEFAULT_NOSKIP = {"regex", "regexs", "list", "lists"}
DEFAULT_NOT_USE_UNMATCHED = {"regex", "regexs"}


class GrammarNotFound(OSError):
    pass


class ParseError(Exception):
    pass


class Parser(object):
    def __init__(self, grammar_filename):
        """Create a Parser object based on the grammar defined in a file

        :param grammar_filename: path to json file specifying grammar for this
        parser, or one of the default grammars included with the package
        """

        if exists(grammar_filename):
            with open(grammar_filename, "r") as f:
                self.grammar = json.load(f)
        else:
            if not grammar_filename.endswith(".json"):
                grammar_filename = grammar_filename + ".json"
            try:
                grammar_fd = pkgutil.get_data(
                    __package__, "grammars/" + grammar_filename
                )
            except FileNotFoundError as e:
                raise GrammarNotFound(
                    "{}:specified grammar could not be found:".format(e)
                )
            self.grammar = json.loads(grammar_fd)

        self._config = self.grammar.get(CONFIG_KEY, {})
        if CONFIG_KEY in self.grammar:
            del self.grammar[CONFIG_KEY]

        # Preprocessing to be applied to each line before checking triggers
        self._preprocessing = self._config.get("regex_preprocessing", [])
        self._preprocessing = [re.compile(r) for r in self._preprocessing]

        self._triggers = {
            trigger_type: [
                (k, v[f"trigger_{trigger_type}"])
                if trigger_type != "re"
                else (k, re.compile(v[f"trigger_{trigger_type}"]))
                for k, v in self.grammar.items()
                if f"trigger_{trigger_type}" in v
            ]
            for trigger_type in ("startswith", "endswith", "contains", "re")
        }

    def _set_section(self, k=None):
        if k in self.grammar:
            self._active_section = self.grammar[k]
            self._section_name = k
            self._section_type = self._active_section.get("type")
        else:
            self._active_section = None
            self._section_name = ""
            self._section_type = None

    def parse(self, filehandle):
        """Parse contents of file according to the loaded grammar

        :param filehandle: a line generator, e.g., a valid file handle
        """

        self._set_section()
        table_header = []
        column_types = []

        output = {}

        for line in filehandle:
            line = line.strip()
            if len(line) == 0:
                # skip blank lines
                continue

            line_pp = [r.findall(line) for r in self._preprocessing]
            line_pp = [m[0].strip() for m in line_pp if len(m) == 1]
            line_unmatched = line_pp[0] if len(line_pp) == 1 else line
            line_pp += [line]

            trigger_check_methods = {
                k: lam
                for k, lam in zip(
                    self._triggers.keys(),
                    (
                        lambda x, t: x.startswith(t),
                        lambda x, t: x.endswith(t),
                        lambda x, t: x.find(t),
                        lambda x, re: re.findall(x),
                    ),
                )
            }
            matches = {
                trigger: [
                    (k, trig_str)
                    for k, trig_str in self._triggers[trigger]
                    if any(
                        [
                            trigger_check_methods[trigger](line, trig_str)
                            for line in line_pp
                        ]
                    )
                ]
                for trigger, method in trigger_check_methods.items()
            }
            section_match = {
                k
                for trigger_matches in matches.values()
                for k, _ in trigger_matches
            }

            # if len(section_match) > 1:
            assert len(section_match) <= 1, ParseError(
                "conflicting sections triggered"
            )

            if len(section_match) == 1:
                # Update the active section
                self._set_section(list(section_match)[0])

                # Determine the unmatched part of the line
                line_unmatched = self.determine_unmatched_part(
                    matches, line_pp
                )

                # Skip the matched line if requested
                if self._active_section.get(
                    "skip", self._section_type not in DEFAULT_NOSKIP
                ):
                    continue

            if self._active_section is None:
                continue

            active_section = self._active_section
            section_type = self._section_type
            section_name = self._section_name

            if active_section.get(
                "use_unmatched",
                self._section_type not in DEFAULT_NOT_USE_UNMATCHED,
            ):
                line = line_unmatched.strip()
                if len(line) == 0:
                    continue

            if section_type == "table":
                sep = active_section.get("separator", ",")
                row = line.split(sep)

                if section_name not in output:
                    # Table needs initialisation
                    (
                        has_header,
                        row,
                        table_header,
                        column_types,
                    ) = self._parse_table(active_section, row)

                    output[section_name] = {k: [] for k in table_header}

                    if active_section.get("has_header", True):
                        continue

                if len(row) < len(table_header):
                    # skip lines that have fewer columns than expected
                    continue

                # Merge extra columns into final column

                row = self._table_merge_extra_columns(
                    table_header, sep, row, column_types
                )

                # Fill out current row
                for val, colname, coltype in zip(
                    row, table_header, column_types
                ):
                    output[section_name][colname].append(
                        _map_to_type(val.strip(), coltype)
                    )

            elif section_type in {"list", "lists"}:
                sep = active_section.get("separator", ",")
                output[section_name] = output.get(section_name, [])

                map_type = active_section.get("map")
                next_list = [
                    _map_to_type(el.strip(), map_type)
                    for el in line.split(sep)
                ]

                list_to_append = (
                    [next_list] if section_type == "lists" else next_list
                )
                output[section_name] += list_to_append

            elif section_type in {"regex", "regexs"}:
                regex = active_section.get("regex", "^(.*)$")
                map_type = active_section.get("map")

                matches = re.findall(regex, line)
                if len(matches) == 0:
                    continue
                elif len(matches) == 1 and section_type == "regex":
                    output[section_name] = _map_to_type(matches[0], map_type)
                else:
                    output[section_name] = output.get(section_name, [])
                    output[section_name] += [
                        _map_to_type(m, map_type) for m in matches
                    ]

                # Terminate after finding the first match
                self._terminate_after_first_match(active_section, section_type)

            elif section_type == "stop":
                break

            else:
                # By default, just append additional lines as text
                new_str = (
                    f"{output[section_name]}\n{line}"
                    if section_name in output
                    else line
                )
                output[section_name] = new_str

        return output

    @staticmethod
    def determine_unmatched_part(
        matches: t.Dict[str, t.List], line_pp: t.List[str]
    ):

        if matches["startswith"]:
            _, t = matches["startswith"][0]
            line_unmatched = [
                line[len(t) :] for line in line_pp if line.startswith(t)
            ][0]
        elif matches["endswith"]:
            _, t = matches["endwith"][0]
            line_unmatched = [
                line[: -(len(t) + 1)] for line in line_pp if line.endswith(t)
            ][0]
        elif matches["contains"]:
            _, t = matches["contains"][0]
            lpp = [line for line in line_pp if line.find(t) >= 0][0]
            i = lpp.find(t)
            line_unmatched = lpp[:i] + lpp[(i + len(t)) :]
        elif matches["re"]:
            _, r = matches["re"][0]
            line_unmatched = [
                r.sub("", line) for line in line_pp if len(r.findall(line)) > 0
            ][0]
        return line_unmatched

    def _terminate_after_first_match(self, active_section, section_type):
        # Terminate after finding the first match
        if section_type == "regex":
            next_section = active_section.get("next_section")
            self._set_section(next_section)
        return next_section

    @staticmethod
    def _parse_table(active_section, row):

        has_header = active_section.get("has_header", True)
        if has_header:
            row = [col.strip() for col in row]
        default_type = active_section.get("default_map", "str")
        colmap = active_section.get("column_map", len(row) * [(None, None)])
        if type(colmap) == list:
            # Columns are defined in order
            if has_header:
                table_header = [mn or rn for rn, (mn, _) in zip(row, colmap)]
                table_header += row[len(colmap) :]
                column_types = [mt for _, mt in colmap]
                column_types += (len(row) - len(colmap)) * [default_type]
            else:
                table_header = [
                    mn or "column{:02d}".format(i + 1)
                    for i, (mn, _) in enumerate(colmap)
                ]
                column_types = [mt or default_type for _, mt in colmap]
        elif type(colmap) == dict:
            if not has_header:
                raise ParseError("dict column maps must have a header")
            # First row is a header
            table_header = [colmap.get(rn, (rn, None))[0] for rn in row]
            column_types = [
                colmap.get(rn, (None, default_type))[1] for rn in row
            ]
        else:
            raise ParseError("badly formatted column map")
        return has_header, row, table_header, column_types

    @staticmethod
    def _table_merge_extra_columns(table_header, sep, row, column_types):
        # Merge extra columns into final column
        ncol = len(table_header)
        if len(row) > ncol:
            row[ncol - 1] = sep.join(row[ncol - 1 :])
            del row[ncol:]
        assert len(row) == len(table_header) and len(row) == len(column_types)
        return row


def _map_to_type(val, map_type):
    if map_type and map_type.startswith("datetime"):
        date_format = "%Y-%m-%dT%H:%M:%S.%fZ"  # ISO 8601 format
        if map_type.startswith("datetime:"):
            date_format = map_type[9:]
        try:
            return datetime.strptime(val, date_format)
        except ValueError:
            return None
    else:
        try:
            return {"str": str, "int": int, "float": float, "bool": bool}.get(
                map_type, str
            )(val)
        except ValueError or TypeError:
            return {"float": float("nan")}.get(map_type)
