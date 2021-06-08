import numpy as np

# Regular python list: To return the (0,1) element we must index as shown below.
print("With regular array")
alist = [[1, 2], [3, 4]]
alist[0][1]
print(alist)

# With Numpy

# If we want to return the right-hand column, there is no trivial way
# to do so with Python lists. In NumPy, indexing follows a
# more convenient syntax.
# Converting the list defined above into an array
print("==================================")
print("With Numpy:")
arr = np.array(alist)
print(arr)
print(arr[0, 1])  # Access and element on line 0 and column 1
print(arr[:, 1])  # Now to access the last column
print(arr[1, :])  # Accessing the lines is achieved in the same way.

print("------------------------")

# Slicing
# Creating an array
arr = np.arange(5)
print(arr)
# Creating the index array
index = np.where(arr > 2)
print(index)

# Creating the desired array
print("------------------------")
new_arr = arr[index]
print(new_arr)

# We can also remove specific elements based on the conditional index
new_arr = np.delete(arr, index)
print(new_arr)
print("------------------------")
# or we can use a boolean array
index = arr > 2
print(index)  # [False False True True True]
new_arr = arr[index]
print(new_arr)
