import numpy as np
import pprint as pp

# Creating an array of zeros and defining column types
r = np.zeros((2,), dtype=('i4,f4,a10'))

# Now creating the columns we want to put in the recarray
col1 = np.arange(2) + 1
col2 = np.arange(2, dtype=np.float32)
col3 = ['Hello', 'World']

# Here we create a list of tuples that is
# identical to the previous toadd list.
toadd = list(zip(col1, col2, col3))

# Assigning values to rearray
r[:] = toadd
r.dtype.names = ('Integers', 'Floats', 'Strings')
pp.pprint(r)

# If we want to access one of the columns by its name, we # can do the following.
print(r['Integers'])


print(r.dtype)
print(r.shape)
print(r.strides)


# Access multiple fields
print(r['Integers', 'Strings'])