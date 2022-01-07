from copy import copy

def reassign_mo_bud(mo_bud, trans):
    """
    Update mother_bud dictionary using another dict with tracks joined

    input
    :param mo_bud: dict with mother's ids as keys and daughters' as values
    :param trans: dict of joint tracks where moved track -> static track

    output
    mo_bud with updated cell ids
    """

    val2lst = lambda x: [j for i in x.values() for j in i]

    bud_inter=set(val2lst(mo_bud)).intersection(trans.keys())

    # translate daughters
    mo_bud = copy(mo_bud)
    for k,das in mo_bud.items():
        for da in bud_inter.intersection(das):
            mo_bud[k][mo_bud[k].index(da)] = trans[da]

    # translate mothers
    mo_inter = set(mo_bud.keys()).intersection(trans.keys())
    for k in mo_inter:
        mo_bud[trans[k]] = mo_bud.get(trans[k], []) + mo_bud[k]
        del mo_bud[k]

    return mo_bud
