from yaml import dump, load


def dict_to_yaml(d, f):
    with open(f, "w") as f:
        dump(d, f)


def add_attrs(hdfile, path, files):
    group = hdfile.create_group(path)
    for k, v in files:
        group.attrs[k] = v
