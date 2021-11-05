"""
TensorFlow Graphs
"""
from tensorflow import Graph, compat, constant


def main():
    """
    main function
    """
    n_1 = constant(1)
    n_2 = constant(2)
    n_3 = n_1 + n_2
    with compat.v1.Session() as sess:
        result = sess.run(n_3)
    print(result)
    print(compat.v1.get_default_graph())
    gra = Graph()
    print(gra)
    graph_one = compat.v1.get_default_graph()
    graph_two = Graph()
    print(graph_one is compat.v1.get_default_graph())
    print(graph_two is compat.v1.get_default_graph())
    graph_two.as_default()
    print(graph_two is compat.v1.get_default_graph())


if __name__ == "__main__":
    compat.v1.disable_v2_behavior()
    main()
