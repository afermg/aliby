"""Function to merge parameter dicts for Extraction."""


def recursive_merge_extractor(dict1, dict2):
    """
    Merge the second extractor tree into the first and return the first.

    Dicts are merged and sets are joined, but a list in the second
    replaces the list in the first: to give the functions for a channel
    and a reduction is to give all of them. To add a function to every
    channel, append it to global_settings.fluorescence_functions
    instead, as examples/run_custom_function.py does.
    """
    for key, value in dict2.items():
        if (
            key in dict1
            and isinstance(dict1[key], dict)
            and isinstance(value, dict)
        ):
            dict1[key] = recursive_merge_extractor(dict1[key], value)
        elif isinstance(dict1.get(key), set) and isinstance(value, set):
            # merge sets at leaf node
            dict1[key] = dict1[key].union(value)
        else:
            # overwrite scalars and new keys
            dict1[key] = value
    return dict1
