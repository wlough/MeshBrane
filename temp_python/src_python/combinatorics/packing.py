###########################################################################################
# Pairing and packing functions
# -------------
#
###########################################################################################
def szudzik_pairing(a, b):
    return a * a + a + b if a >= b else a + b * b


def inverse_szudzik_pairing(z):
    sqrt_z = int(z**0.5)
    remainder = z - sqrt_z * sqrt_z
    if remainder < sqrt_z:
        return remainder, sqrt_z
    else:
        return sqrt_z, remainder - sqrt_z


def szudzik_packing(*args):
    if len(args) > 2:
        return szudzik_pairing(args[0], szudzik_packing(*args[1:]))
    elif len(args) == 1:
        return args[0]
    else:
        return szudzik_pairing(*args)


def szudzik_unpacking(*args, target_len=2):
    while len(args) < target_len:
        args = args[:-1] + inverse_szudzik_pairing(args[-1])
    return args


def cantor_pairing(a, b):
    """
    Pair two integers a and b into a single integer using the Cantor pairing function.

    Parameters
    ----------
    a : int
    b : int

    Returns
    -------
    int : Cantor pairing of a and b
    """
    return (a + b) * (a + b + 1) // 2 + b


def inverse_cantor_pairing(z):
    """
    Inverse of the Cantor pairing function.

    Parameters
    ----------
    z : int
        Cantor pairing of a and b

    Returns
    -------
    tuple of int : a, b
    """
    w = int((8 * z + 1) ** 0.5)
    t = (w - 1) // 2
    y = z - t * (t + 1) // 2
    x = t - y
    return x, y


def cantor_packing_right2left(*args):
    """
    Pack a list of integers into a single integer using the Cantor pairing function.

    Parameters
    ----------
    *args : list of int

    Returns
    -------
    int : Cantor pairing of *args
    """
    if len(args) > 2:
        return cantor_pairing(args[0], cantor_packing_right2left(*args[1:]))
    else:
        return cantor_pairing(*args)


def cantor_packing_left2right(*args):
    """
    Pack a list of integers into a single integer using the Cantor pairing function.

    cantor_packing(a, b, c)=cantor_pairing(cantor_pairing(a, b), c)
    Parameters
    ----------
    *args : list of int

    Returns
    -------
    int : Cantor pairing of *args
    """
    if len(args) > 2:
        return cantor_pairing(cantor_packing_left2right(*args[:-1]), args[-1])
    else:
        return cantor_pairing(*args)
