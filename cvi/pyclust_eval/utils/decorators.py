import numpy as np
import pandas as pd


def pd_to_np(function):
    def change_type(*args, **kwargs):
        kwargs.update((x, np.array(kwargs[x])) for x in kwargs if type(kwargs[x]) == pd.DataFrame)
        args = tuple(np.array(x) if type(x) == pd.DataFrame else x for x in list(args) )
        function(*args, **kwargs)
    return change_type


# Test
@pd_to_np
def asd(x):
    print(x)
    print(type(x))

# asd(x=pd.DataFrame([1,2,3]))