"""
TensorFlow Basics
"""
from tensorflow import (
    constant,
    fill,
    zeros,
    ones,
    matmul,
    Session,
    random_normal,
    random_uniform,
    InteractiveSession,
)


def wrapper():
    """
    wrapper function
    """
    hello = constant("Hello")
    print(type(hello))
    world = constant("World")
    result = hello + world
    print(result)
    print(type(result))

    with Session() as sess:
        result = sess.run(hello + world)
    print(result)
    tensor_1 = constant(1)
    tensor_2 = constant(2)
    print(type(tensor_1))
    print(tensor_1 + tensor_2)
    const = constant(10)
    fill_mat = fill((4, 4), 10)
    myzeros = zeros((4, 4))
    myones = ones((4, 4))
    myrandn = random_normal((4, 4), mean=0, stddev=0.5)
    myrandu = random_uniform((4, 4), minval=0, maxval=1)
    my_ops = [const, fill_mat, myzeros, myones, myrandn, myrandu]

    sess = InteractiveSession()
    for ops in my_ops:
        print(ops.eval())
        print("\n")
    ada = constant([[1, 2], [3, 4]])
    print(ada.get_shape())
    bda = constant([[10], [100]])
    print(bda.get_shape())
    result = matmul(ada, bda)
    print(result.eval())


if __name__ == "__main__":
    wrapper()
