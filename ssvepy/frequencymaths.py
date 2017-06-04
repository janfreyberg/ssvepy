"""
Some helper functions to do frequency maths
"""
import numpy as np
import itertools


def compute_intermodulation(frequencies, fmin=0.1, fmax=50, orders=range(2, 5)):
    if not orders:
        return None, None

    if type(orders) is bool:
        orders = range(2, 5)
    elif type(orders) is int:
        orders = range(2, orders)

    if len(frequencies) == 2:
        output = []
        orderlist = []
        for order in orders:
            coeffs = [sign * k for sign in [-1, 1] for k in range(1, order)]
            output = output + \
                [k1 * frequencies[0] + k2 * frequencies[1]
                 for k1, k2 in itertools.product(coeffs, coeffs)
                 if abs(k1) + abs(k2) == order
                 and k1 * frequencies[0] + k2 * frequencies[1] > fmin
                 and k1 * frequencies[0] + k2 * frequencies[1] < fmax]
            orderlist = orderlist + [order] * (len(output) - len(orderlist))
    elif len(frequencies) < 2:
        raise ValueError('You need to provide more than one frequency to '
                         'calculate intermodulation frequencies.')

    return np.array(output), np.array(orderlist)


def compute_harmonics(frequencies, fmin=0.1, fmax=50, orders=range(2, 5)):
    if not orders:
        return None, None

    if type(orders) is bool:
        orders = [o for o in range(2, 6)]
    elif type(orders) is int:
        orders = [o for o in range(2, orders)]
    try:
        freqs = [[n * f for n in orders
                  if n * f <= fmax and n * f >= fmin]
                 for f in frequencies]
    except:
        freqs = [[n * f for n in orders
                  if n * f <= fmax and n * f >= fmin]
                 for f in [frequencies]]

    return np.array(freqs), np.array(orders)


def compute_subharmonics(frequencies, fmin=0.1, fmax=50, orders=range(2, 5)):
    if not orders:
        return None, None

    if type(orders) is bool:
        orders = [o for o in range(2, 6)]  # default: 4 harmonics
    elif type(orders) is int:
        orders = [o for o in range(2, orders)]

    freqs = [[f / o if (f / o <= fmax and f / o >= fmin) else np.nan
              for o in orders]
             for f in frequencies]
    # freqs = [f for sublist in freqs for f in sublist]

    return np.array(freqs), np.array(orders)
