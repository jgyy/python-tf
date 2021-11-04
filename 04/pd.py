"""
Pandas Crash Course
"""
from os.path import join, dirname
from pandas import DataFrame, read_csv


def wrapper():
    """
    wrapper function
    """
    daf = DataFrame(read_csv(join(dirname(__file__), "salaries.csv")))
    print(daf)
    print(daf["Name"])
    print(daf["Salary"])
    print(daf[["Name", "Salary"]])
    print(daf["Age"])
    print(daf["Age"].mean())
    print(daf["Age"] > 30)
    age_filter = daf['Age'] > 30
    print(daf[age_filter])
    print(daf[daf['Age'] > 30])


if __name__ == "__main__":
    wrapper()
