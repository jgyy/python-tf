"""
Variables and Placeholders
"""
from tensorflow import Variable, compat


def main():
    """
    main function
    """
    sess = compat.v1.InteractiveSession()
    my_tensor = compat.v1.random_uniform((4, 4), 0, 1)
    my_var = Variable(initial_value=my_tensor)
    print(my_var)
    init = compat.v1.global_variables_initializer()
    init.run()
    print(my_var.eval())
    print(sess.run(my_var))


if __name__ == "__main__":
    compat.v1.disable_v2_behavior()
    main()
