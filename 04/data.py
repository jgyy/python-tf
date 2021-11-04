"""
Data Viz Crash Course
"""
from os.path import join, dirname
from numpy import arange
from numpy.random import randint
from pandas import DataFrame, read_csv
from matplotlib.pyplot import (
    plot,
    figure,
    show,
    xlim,
    ylim,
    title,
    xlabel,
    ylabel,
    imshow,
    colorbar,
)


def wrapper():
    """
    wrapper function
    """
    xda = arange(0, 10)
    yda = xda ** 2
    print(xda)
    print(yda)

    figure()
    plot(xda, yda)
    figure()
    plot(xda, yda, "*")
    figure()
    plot(xda, yda, "r--")
    figure()
    plot(xda, yda, "r--")
    xlim(2, 4)
    ylim(10, 20)
    title("Zoomed")
    xlabel("X Axis")
    ylabel("Y Axis")

    mat = arange(0, 100).reshape(10, 10)
    figure()
    imshow(mat)
    mat = randint(0, 100, (10, 10))
    figure()
    imshow(mat)
    figure()
    imshow(mat, cmap="coolwarm")
    colorbar()
    daf = DataFrame(read_csv(join(dirname(__file__), "salaries.csv")))
    print(daf)
    daf.plot(x="Salary", y="Age", kind="scatter")
    daf.plot(x='Salary',kind='hist')


if __name__ == "__main__":
    wrapper()
    show()
