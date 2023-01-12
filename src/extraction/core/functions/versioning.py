import git


def get_sha():
    # FIXME Unused, but *should* be used...
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha
