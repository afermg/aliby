# -*- coding: utf-8 -*-
"""
logfile_parser
~~~~~~~~~~~~

Simple log file parsing according to grammars specified in JSON

:copyright: (c) 2020 by Julian Pietsch.
:license: LGPL
"""

from os.path import exists, join, dirname
import json
import re
from datetime import datetime
import pkgutil

CONFIG_KEY = '@@CONFIG@@'
DEFAULT_NOSKIP = {'regex', 'regexs', 'list', 'lists'}
DEFAULT_NOT_USE_UNMATCHED = {'regex', 'regexs'}


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
            with open(grammar_filename, 'r') as f:
                self.grammar = json.load(f)
        else:
            if not grammar_filename.endswith('.json'):
                grammar_filename = grammar_filename + '.json'
            try:
                grammar_fd = pkgutil.get_data(__package__,
                                          'grammars/' + grammar_filename)
            except FileNotFoundError as e:
                raise GrammarNotFound('specified grammar could not be '
                                      'found')
            self.grammar = json.loads(grammar_fd)

        self._config = self.grammar.get(CONFIG_KEY, {})
        if CONFIG_KEY in self.grammar:
            del self.grammar[CONFIG_KEY]

        # Preprocessing to be applied to each line before checking triggers
        self._preprocessing = self._config.get('regex_preprocessing', [])
        self._preprocessing = [re.compile(r) for r in self._preprocessing]

        self._trigger_startswith = [(k, v['trigger_startswith'])
                                    for k, v in self.grammar.items()
                                    if 'trigger_startswith' in v]
        self._trigger_endswith = [(k, v['trigger_endswith'])
                                  for k, v in self.grammar.items()
                                  if 'trigger_endswith' in v]
        self._trigger_contains = [(k, v['trigger_contains'])
                                  for k, v in self.grammar.items()
                                  if 'trigger_contains' in v]
        self._trigger_re = [(k, re.compile(v['trigger_re']))
                            for k, v in self.grammar.items()
                            if 'trigger_re' in v]

    def _set_section(self, k=None):
        if k in self.grammar:
            self._active_section = self.grammar[k]
            self._section_name = k
            self._section_type = self._active_section.get('type')
        else:
            self._active_section = None
            self._section_name = ''
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

            line_unmatched = line

            line_pp = [r.findall(line) for r in self._preprocessing]
            line_pp = [m[0].strip() for m in line_pp if len(m) == 1]
            if len(line_pp) == 1:
                line_unmatched = line_pp[0]
            line_pp += [line]

            sw_matches = [(k, t) for k, t in self._trigger_startswith
                          if any([l.startswith(t) for l in line_pp])]
            ew_matches = [(k, t) for k, t in self._trigger_endswith
                          if any([l.endswith(t) for l in line_pp])]
            co_matches = [(k, t) for k, t in self._trigger_contains
                          if any([l.find(t) >= 0 for l in line_pp])]
            re_matches = [(k, r) for k, r in self._trigger_re
                          if any([len(r.findall(l)) > 0 for l in line_pp])]

            section_match = {k for k, _ in (sw_matches + ew_matches +
                                            co_matches + re_matches)}
            if len(section_match) > 1:
                raise ParseError('conflicting sections triggered')

            if len(section_match) == 1:
                # Update the active section
                self._set_section(list(section_match)[0])

                # Determine the unmatched part of the line
                if len(sw_matches) > 0:
                    _, t = sw_matches[0]
                    line_unmatched = [l[len(t):] for l in line_pp
                                      if l.startswith(t)][0]
                elif len(ew_matches) > 0:
                    _, t = ew_matches[0]
                    line_unmatched = [l[:-(len(t)+1)] for l in line_pp
                                      if l.endswith(t)][0]
                elif len(co_matches) > 0:
                    _, t = co_matches[0]
                    lpp = [l for l in line_pp if l.find(t) >= 0][0]
                    i = lpp.find(t)
                    line_unmatched = lpp[:i] + lpp[(i + len(t)):]
                elif len(re_matches) > 0:
                    _, r = re_matches[0]
                    line_unmatched = [r.sub('', l) for l in line_pp
                                      if len(r.findall(l)) > 0][0]

                # Skip the matched line if requested
                if self._active_section.get('skip', self._section_type not in
                                            DEFAULT_NOSKIP):
                    continue

            if self._active_section is None:
                continue

            active_section = self._active_section
            section_type = self._section_type
            section_name = self._section_name

            if active_section.get('use_unmatched', self._section_type not in
                                  DEFAULT_NOT_USE_UNMATCHED):
                line = line_unmatched.strip()
                if len(line) == 0:
                    continue

            if section_type == 'table':
                sep = active_section.get('separator', ',')
                row = line.split(sep)

                if section_name not in output:
                    # Table needs initialisation
                    has_header = active_section.get('has_header', True)
                    if has_header:
                        row = [col.strip() for col in row]
                    default_type = active_section.get('default_map', 'str')
                    colmap = active_section.get(
                        'column_map', len(row)*[(None, None)]
                    )
                    if type(colmap) == list:
                        # Columns are defined in order
                        if has_header:
                            table_header = [mn or rn for rn, (mn, _) in
                                            zip(row, colmap)]
                            table_header += row[len(colmap):]
                            column_types = [mt for _, mt in colmap]
                            column_types += (len(row) - len(colmap)) * [default_type]
                        else:
                            table_header = [mn or 'column{:02d}'.format(i+1)
                                            for i, (mn, _) in enumerate(colmap)]
                            column_types = [mt or default_type for _, mt in colmap]
                    elif type(colmap) == dict:
                        if not has_header:
                            raise ParseError('dict column maps must have a header')
                        # First row is a header
                        table_header = [colmap.get(rn, (rn, None))[0] for rn in row]
                        column_types = [colmap.get(rn, (None, default_type))[1]
                                        for rn in row]
                    else:
                        raise ParseError('badly formatted column map')

                    output[section_name] = {k: [] for k in table_header}
                    if has_header:
                        continue

                if len(row) < len(table_header):
                    # skip lines that have fewer columns than expected
                    continue

                # Merge extra columns into final column
                ncol = len(table_header)
                if len(row) > ncol:
                    row[ncol - 1] = sep.join(row[ncol - 1:])
                    del row[ncol:]
                assert len(row) == len(table_header) and len(row) == len(column_types)

                # Fill out current row
                for val, colname, coltype in zip(row, table_header, column_types):
                    output[section_name][colname].append(
                        _map_to_type(val.strip(), coltype)
                    )

            elif section_type in {'list', 'lists'}:
                sep = active_section.get('separator', ',')
                if section_name not in output:
                    output[section_name] = []

                map_type = active_section.get('map')
                next_list = [_map_to_type(el.strip(), map_type)
                             for el in line.split(sep)]
                if section_type == 'lists':
                    output[section_name].append(next_list)
                else:
                    output[section_name] += next_list

            elif section_type in {'regex', 'regexs'}:
                regex = active_section.get('regex', '^(.*)$')
                map_type = active_section.get('map')

                matches = re.findall(regex, line)
                if len(matches) == 0:
                    continue
                elif len(matches) == 1 and section_type == 'regex':
                    output[section_name] = _map_to_type(matches[0], map_type)
                else:
                    if section_name not in output:
                        output[section_name] = []
                    output[section_name] += [_map_to_type(m, map_type)
                                            for m in matches]

                # Terminate after finding the first match
                if section_type == 'regex':
                    next_section = active_section.get('next_section')
                    self._set_section(next_section)

            elif section_type == 'stop':
                break

            else:
                # By default, just append additional lines as text
                if section_name in output:
                    output[section_name] += '\n' + line
                else:
                    output[section_name] = line

        return output


def _map_to_type(val, map_type):
    if map_type and map_type.startswith('datetime'):
        date_format = '%Y-%m-%dT%H:%M:%S.%fZ'  # ISO 8601 format
        if map_type.startswith('datetime:'):
            date_format = map_type[9:]
        try:
            return datetime.strptime(val, date_format)
        except ValueError:
            return None
    else:
        try:
            return {
                'str': str, 'int': int, 'float': float, 'bool': bool
            }.get(map_type, str)(val)
        except ValueError or TypeError:
            return {'float': float('nan')}.get(map_type)

