# Running the analysis pipeline

You can run the analysis pipeline either via the command line interface (CLI) or using a script that incorporates the `aliby.pipeline.Pipeline` object.

## CLI

On a CLI, you can use the `aliby-run` command.  This command takes options as follows:
- `--host`: Address of image-hosting server.
- `--username`: Username to access image-hosting server.
- `--password`: Password to access image-hosting server.
- `--expt_id`: Number ID of experiment stored on host server.
- `--distributed`: Number of distributed cores to use for segmentation and signal processing.  If 0, there is no parallelisation.
- `--tps`: Optional.  Number of time points from the beginning of the experiment to use.  If not specified, the pipeline processes all time points.
- `--directory`: Optional.  Parent directory to save the data files (HDF5) generated, `./data` by default; the files will be stored in a child directory whose name is the name of the experiment.
- `--filter`: Optional.  List of positions to use for analysis.  Alternatively, a regex (regular expression) or list of regexes to search for positions.  **Note: for the CLI, currently it is not able to take a list of strings as input.**
- `--overwrite`: Optional.  Whether to overwrite an existing data directory.  True by default.
- `--override_meta`: Optional.  Whether to overwrite an existing data directory.  True by default.

Example usage:
 ```bash
aliby-run --expt_id EXPT_PATH --distributed 4 --tps None
 ```

And to run Omero servers, the basic arguments are shown:
 ```bash
 aliby-run --expt_id XXX --host SERVER.ADDRESS --user USER --password PASSWORD 
 ```


## Script

Use the `aliby.pipeline.Pipeline` object and supply a dictionary, following the example below.  The meaning of the parameters are the same as described in the CLI section above.

```python
#!/usr/bin/env python3

from aliby.pipeline import Pipeline, PipelineParameters

# Specify experiment IDs
ids = [101, 102]

for i in ids:
    print(i)
    try:
        params = PipelineParameters.default(
            # Create dictionary to define pipeline parameters.
            general={
                "expt_id": i,
                "distributed": 6,
                "host": "INSERT ADDRESS HERE",
                "username": "INSERT USERNAME HERE",
                "password": "INSERT PASSWORD HERE",
                # Ensure data will be overwriten
                "override_meta": True,
                "overwrite": True,
            }
        )

        # Fine-grained control beyond general parameters:
        # change specific leaf in the extraction tree.
        # This example tells the pipeline to additionally compute the
        # nuc_est_conv quantity, which is a measure of the degree of
        # localisation of a signal in a cell.
        params = params.to_dict()
        leaf_to_change = params["extraction"]["tree"]["GFP"]["np_max"]
        leaf_to_change.add("nuc_est_conv")

        # Regenerate PipelineParameters
        p = Pipeline(PipelineParameters.from_dict(params))

        # Run pipeline
        p.run()
        
    # Error handling
    except Exception as e:
        print(e)
```

This example code can be the contents of a `run.py` file, and you can run it via

```bash
python run.py
```

in the appropriate virtual environment.

Alternatively, the example code can be the contents of a cell in a jupyter notebook.
