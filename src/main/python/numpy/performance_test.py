from timeit import timeit

import numpy as np

#Create an array with 10^7 elements
arr = np.arange(1e7)

# Converting ndarray to list
larr = arr.tolist()

def list_times(alist, scalar):
    return [val*scalar for val in alist]

# Using IPython's magic
# timeit command timeit arr * 1.1

# timeit list_times(larr, 1.1)



# box(x, y){line=snake}