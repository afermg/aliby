# logfile\_parser

Simple log file parsing according to grammars specified in JSON

## Basic usage

This package comes with three built-in grammars: 'multiDGUI\_acq\_format',
'multiDGUI\_log\_format' and 'cExperiment\_log\_format'. As an example, the
'multiDGUI\_acq\_format' grammar can be used to parse the included example
using:

```python
>>> from logfile_parser import Parser
>>> acq_parser = Parser('multiDGUI_acq_format')
>>> with open('examples/example_multiDGUI_acq.txt', 'r') as f:
...     parsed = acq_parser.parse(f)
>>> print(parsed)
```

The parsed output is a `dict` containing any fields satisfying the grammar.

## Defining new grammars

Custom grammars should be written in json as a dictionary with keys specifying
the information to extract from the log file.

The built-in grammars are useful examples or starting points for defining custom
grammars. They can be found in the `logfile_parser/grammars` directory.

Let's start with a basic example of a log file that we might want to parse:

```text
Date: 16 Apr 2020
Microscope: Batgirl
Experiment details:
My lengthy description of what will certainly be a great experiment.
This description takes multiple lines.
Tags:
User name, Project name, Experiment name
```

A basic grammar that just extracts the description of the experiment could be
defined using:

```json
{
    "description": {
        "trigger_startswith": "Experiment details:"
    },
    "stop": {
        "trigger_startswith": "Tags:",
        "type": "stop"
    }
}
```

This tells the parser to fill the "description" field of the parsed result with
text on lines *after* that starting with the text "Experiment details:", and
then tells the parser to terminate parsing whenever it encounters a line that
starts with the text "Tags:". If you wanted it to include the trigger line, you
would specify `"skip": "false"` as an additional property for `"description"`.

If we also wanted to fill a "tags" field with the comma separated tags, we would
just need to change the type to "list":

```json
{
    "description": {
        "trigger_startswith": "Experiment details:"
    },
    "tags": {
        "trigger_startswith": "Tags:",
        "type": "list"
    }
}
```

To extract the microscope name, we can make use of the "regex" type:

```json
{
    "microscope": {
        "trigger_startswith": "Microscope:",
        "type": "regex",
        "regex": "^Microscope:\\s*(.*)$"
    }
}
```

The expression found in the bracketed group will be stored in the "microscope"
field of the parsed result.

Finally, to extract a date, we combine a "regex" with a "map" to map the text
to a Python `datetime` object:

```json
{
    "date": {
        "trigger_startswith": "Date:",
        "type": "regex",
        "regex": "^.*(\\d{2} [A-Z][a-z]{2} \\d{4})$",
        "map": "datetime:%d %b %Y"
    }
}
```

Putting this all together gives us the following grammar:

```json
{
    "date": {
        "trigger_startswith": "Date:",
        "type": "regex",
        "regex": "^.*(\\d{2} [A-Z][a-z]{2} \\d{4})$",
        "map": "datetime:%d %b %Y"
    },
    "microscope": {
        "trigger_startswith": "Microscope:",
        "type": "regex",
        "regex": "^Microscope:\\s*(.*)$"
    },
    "description": {
        "trigger_startswith": "Experiment details:"
    },
    "tags": {
        "trigger_startswith": "Tags:",
        "type": "list"
    }
}
```

If this is saved to a file `newgrammar.json` we could parse the log file as 
listed above (say it is in `logfile.txt`) using the following:

```python
>>> from logfile_parser import Parser
>>> parser = Parser('newgrammar.json')
>>> with open('logfile.txt', 'r') as f:
...     parsed = parser.parse(f)
>>> print(parsed)
{'date': datetime.datetime(2020, 4, 16, 0, 0), 'microscope': 'Batgirl',
'description': 'My lengthy description of what will certainly be a great 
experiment.\nThis description takes multiple lines.', 'tags': ['User name', 
'Project name', 'Experiment name']}
```

