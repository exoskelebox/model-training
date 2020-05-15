def fold(list):
    for i, current in enumerate(list):
        rest = list[:i] + list[i+1:]
        yield current,  rest
