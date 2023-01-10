import numpy as np

def main():
    a = []

    b = np.array([1, 2, 3, 4])
    c = np.array([5, 6, 7])

    a.append(b)
    a.append(c)

    print(a)
    print(type(a))

    d = np.asarray(a, dtype=object)

    print(d)
    print(type(d))


def none():
    pass