"""Function to merge parameter dicts for Extraction."""


def recursive_merge_extractor(dict1, dict2):
    """Merge two extractor trees."""
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
