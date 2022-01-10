from copy import copy


def reassign_mo_bud(mo_bud, trans):
    """
    FIXME:
    Update mother_bud dictionary using another dict with tracks joined

    input
    :param mo_bud: dict with mother's ids as keys and daughters' as values
    :param trans: dict of joint tracks where moved track -> static track

    output
    mo_bud with updated cell ids
    """

    val2lst = lambda x: [j for i in x.values() for j in i]

    bud_inter = set(val2lst(mo_bud)).intersection(trans.keys())

    # translate mothers and carry  untranslated daughters
    new_mb = copy(mo_bud)
    for origin, target in trans:
        if target in new_mb:
            new_mb[target] = new_mb.get(target, set()).union(new_mb[origin])
        del new_mb[origin]

    # translate daughters
    for origin, target in trans:
        if origin in bud_inter:  # Limit scope of searcg
            for mo, da in new_mb.items():
                if origin in da:
                    da.remove(origin)
                    new_mb[mo] = da.union(target)

    return new_mb
