def recursive_merge_extractor(dict1, dict2):
    """For merge two extractor trees."""
    for key, value in dict2.items():
        if (
            key in dict1
            and isinstance(dict1[key], dict)
            and isinstance(value, dict)
        ):
            # recursively merge nested dictionaries
            dict1[key] = recursive_merge_extractor(dict1[key], value)
        else:
            # merge sets at leaf node
            dict1[key] = dict1[key].union(value)
    return dict1
