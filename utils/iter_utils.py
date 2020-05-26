import numpy as np


def fold(lst):
    if isinstance(lst, np.ndarray):
        lst = lst.tolist()

    for i, current in enumerate(lst):
        rest = lst[:i] + lst[i+1:]
        yield current,  rest
