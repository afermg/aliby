import git
import pkg_resources


def get_sha():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


def get_version(pkg="extraction"):
    return pkg_resources.require(pkg)[0].version
