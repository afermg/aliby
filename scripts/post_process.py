from postprocessor.core.processor import PostProcessorParameters, PostProcessor
import gc
import h5py
import logging 


def post_process(filepath, params):
    pp = PostProcessor(filepath, params)
    tmp = pp.run()
    return tmp


if __name__ == "__main__":
    # Close open objects
    for obj in gc.get_objects():   # Browse through ALL objects
        if isinstance(obj, h5py.File):   # Just HDF5 files
            try:
                obj.close()
                logging.warn(f"Object {obj} was open")
            except:
                pass # Was already closed

    params = PostProcessorParameters.default()
    filepath = '/home/jupyter-diane/python-pipeline/data/2tozero_Hxts_02/Hxt1_024.h5'

    tmp = post_process(filepath, params) 

