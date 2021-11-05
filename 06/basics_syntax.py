"""
TensorFlow Basics
"""
from tensorflow import __version__, compat, constant, fill, zeros, ones, matmul


def main():
    """
    main function
    """
    print(__version__)
    with compat.v1.Session() as sess:
        result = sess.run(constant("Hello") + constant("World"))
    print(result)
    tensor_1 = constant(1)
    tensor_2 = constant(2)
    print(type(tensor_1))
    print(tensor_1 + tensor_2)
    const = constant(10)
    fill_mat = fill((4, 4), 10)
    myzeros = zeros((4, 4))
    myones = ones((4, 4))
    myrandn = compat.v1.random_normal(shape=(4, 4), mean=0, stddev=0.5)
    myrandu = compat.v1.random_uniform((4, 4), 0, 1)
    my_ops = [const, fill_mat, myzeros, myones, myrandn, myrandu]
    sess = compat.v1.InteractiveSession()
    for ops in my_ops:
        print(ops.eval())
        print("\n")
    a_var = constant([[1, 2], [3, 4]])
    print(a_var.get_shape())
    b_var = constant([[10], [100]])
    print(b_var.get_shape())
    result = matmul(a_var, b_var)
    print(result.eval())


if __name__ == "__main__":
    compat.v1.disable_v2_behavior()
    main()
