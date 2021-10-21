"""
Manual Neural Network
"""
from numpy import array, exp, linspace
from matplotlib.pyplot import plot, show, figure, scatter
from sklearn.datasets import make_blobs


class SimpleClass:
    """
    simple class
    """

    def __init__(self, str_input):
        print("SIMPLE " + str_input)

    def __call__(self, *args, **kwdsy):
        print(args, kwdsy)

    def __delattr__(self, name):
        return name


class ExtendedClass(SimpleClass):
    """
    extended class
    """

    def __init__(self):
        super().__init__("My String")
        print("EXTENDED")

    def __call__(self, *args, **kwdsy):
        print(args, kwdsy)

    def __delattr__(self, name):
        return name


class Operation:
    """
    An Operation is a node in a "Graph". TensorFlow will also use this concept of a Graph.
    This Operation class will be inherited by other classes that actually compute the specific
    operation, such as adding or matrix multiplication.
    """

    def __init__(self, input_nodes=None):
        """
        Intialize an Operation
        """
        if not input_nodes:
            input_nodes = []
        self.inputs = []
        self.input_nodes = input_nodes
        self.output_nodes = []
        for node in input_nodes:
            node.output_nodes.append(self)

    def __delattr__(self, name):
        return name

    def compute(self, x_var, y_var):
        """
        This is a placeholder function. It will be overwritten by the actual specific operation
        that inherits from this class.
        """
        self.inputs = [x_var, y_var]
        return x_var + y_var


class Add(Operation):
    """
    add operation class
    """

    def __init__(self, x, y):
        super().__init__([x, y])

    def __delattr__(self, name):
        return name

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class Multiply(Operation):
    """
    multiply operation class
    """

    def __init__(self, a, b):
        super().__init__([a, b])

    def __delattr__(self, name):
        return name

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


class Matmul(Operation):
    """
    matmul operation class
    """

    def __init__(self, a, b):
        super().__init__([a, b])

    def __delattr__(self, name):
        return name

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)


class Placeholder:
    """
    A placeholder is a node that needs to be provided a value for computing the output in the Graph.
    """

    def __init__(self):
        self.output_nodes = []

    def __call__(self, *args, **kwdsy):
        print(args, kwdsy, end="\r")

    def __delattr__(self, name):
        return name


class Variable:
    """
    This variable is a changeable parameter of the Graph.
    """

    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []

    def __call__(self, *args, **kwdsy):
        print(args, kwdsy)

    def __delattr__(self, name):
        return name


class Graph:
    """
    graph class
    """

    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def __delattr__(self, name):
        return name

    def set_as_default(self):
        """
        Sets this Graph instance as the Global Default Graph
        """
        _default_graph = self
        print(_default_graph, end="\r")


class Session:
    """
    session class
    """

    def __delattr__(self, name):
        return name

    @staticmethod
    def run(operation, feed_dict=None):
        """
        operation: The operation to compute
        feed_dict: Dictionary mapping placeholders to input values (the data)
        """
        if not feed_dict:
            feed_dict = {}
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            if isinstance(node, Placeholder):
                node.output = feed_dict[node]
            elif isinstance(node, Variable):
                node.output = node.value
            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
            if isinstance(node.output, list):
                node.output = array(node.output)
        return operation.output


class Sigmoid(Operation):
    """
    sigmoid operation class
    """

    def __init__(self, z):
        """
        a is the input node
        """
        super().__init__([z])

    def __delattr__(self, name):
        return name

    def compute(self, x_var, y_var=None):
        print(y_var)
        return 1 / (1 + exp(-x_var))


def traverse_postorder(operation):
    """
    PostOrder Traversal of Nodes. Basically makes sure computations are done in
    the correct order (Ax first , then Ax + b). Feel free to copy and paste this code.
    It is not super important for understanding the basic fundamentals of deep learning.
    """
    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


def wrapper():
    """
    wrapper function
    """
    ExtendedClass()
    gra = Graph()
    gra.set_as_default()
    ada = Variable(10)
    bda = Variable(1)
    xda = Placeholder()
    yda = Multiply(ada, xda)
    zda = Add(yda, bda)
    sess = Session()
    result = sess.run(operation=zda, feed_dict={xda: 10})
    print(result)
    print(10 * 10 + 1)

    gra = Graph()
    gra.set_as_default()
    ada = Variable([[10, 20], [30, 40]])
    bda = Variable([1, 1])
    xda = Placeholder()
    yda = Matmul(ada, xda)
    zda = Add(yda, bda)
    sess = Session()
    result = sess.run(operation=zda, feed_dict={xda: 10})
    print(result)
    sigmoid = lambda z: 1 / (1 + exp(-z))
    sample_z = linspace(-10, 10, 100)
    sample_a = sigmoid(sample_z)

    figure()
    plot(sample_z, sample_a)
    data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
    print(data)

    figure()
    features = data[0]
    scatter(features[:, 0], features[:, 1])

    figure()
    labels = data[1]
    scatter(features[:, 0], features[:, 1], c=labels, cmap="coolwarm")

    figure()
    xda = linspace(0, 11, 10)
    yda = 5 - xda
    scatter(features[:, 0], features[:, 1], c=labels, cmap="coolwarm")
    plot(xda, yda)
    print(array([1, 1]).dot(array([[8], [10]])) - 5)
    print(array([1, 1]).dot(array([[4], [-10]])) - 5)

    example()


def example():
    """
    example session graph
    """
    gra = Graph()
    gra.set_as_default()
    xda = Placeholder()
    wda = Variable([1, 1])
    bda = Variable(-5)
    zda = Add(Matmul(wda, xda), bda)
    ada = Sigmoid(zda)
    sess = Session()
    print(sess.run(operation=ada, feed_dict={xda: [8, 10]}))
    print(sess.run(operation=ada, feed_dict={xda: [0, -10]}))


if __name__ == "__main__":
    wrapper()
    show()
