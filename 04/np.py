"""
Numpy Crash Course
"""
from numpy import array, arange, zeros, ones, linspace
from numpy.random import randint, seed


def wrapper():
    """
    wrapper function
    """
    my_list = [0, 1, 2, 3, 4]
    arr = array(my_list)
    print(arr)
    print(arange(0, 10))
    print(arange(0, 10, 2))
    print(zeros((5, 5)))
    print(ones((2, 4)))
    print(randint(0, 10))
    print(randint(0, 10, (3, 3)))
    print(linspace(0, 10, 6))
    print(linspace(0, 10, 101))
    seed(101)
    arr = randint(0, 100, 10)
    print(arr)
    arr2 = randint(0, 100, 10)
    print(arr2)
    print(arr.max())
    print(arr.min())
    print(arr.mean())
    print(arr.argmin())
    print(arr.reshape(2, 5))
    mat = arange(0, 100).reshape(10, 10)
    print(mat)
    row = 0
    col = 1
    print(mat[row, col])
    print(mat[:, col])
    print(mat[row, :])
    print(mat[0:3, 0:3])
    print(mat > 50)
    print(mat[mat > 50])


if __name__ == "__main__":
    wrapper()
