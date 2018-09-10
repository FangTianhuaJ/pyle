import numpy as np

def test(n):
    def func(y):
        x = y+n
        return x
    return func