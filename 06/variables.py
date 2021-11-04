"""
Variables and Placeholders
"""
from tensorflow import (
    random_uniform,
    Variable,
    InteractiveSession,
    global_variables_initializer,
    placeholder,
    float64,
    float32,
    int32,
)


def wrapper():
    """
    wrapper function
    """
    sess = InteractiveSession()
    my_tensor = random_uniform((4, 4), 0, 1)
    my_var = Variable(initial_value=my_tensor)
    print(my_var)
    init = global_variables_initializer()
    init.run()
    print(my_var.eval())
    print(sess.run(my_var))
    placeholder(float64)
    placeholder(int32)
    placeholder(float32, shape=(None, 5))


if __name__ == "__main__":
    wrapper()
