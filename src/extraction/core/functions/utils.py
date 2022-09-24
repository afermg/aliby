from collections import deque


def depth(d):
    """
    Copied from https://stackoverflow.com/a/23499088

    Used to determine the depth of our config trees and fill them
    """
    queue = deque([(id(d), d, 1)])
    memo = set()
    while queue:
        id_, o, level = queue.popleft()
        if id_ in memo:
            continue
        memo.add(id_)
        if isinstance(o, dict):
            queue += ((id(v), v, level + 1) for v in o.values())
    return level
