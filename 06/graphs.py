"""
TensorFlow Graphs
"""
from tensorflow import constant, Graph, Session, get_default_graph


def wrapper():
    """
    wrapper function
    """
    nu1 = constant(1)
    nu2 = constant(2)
    nu3 = nu1 + nu2
    with Session() as sess:
        result = sess.run(nu3)
    print(result)
    print(get_default_graph())
    gra = Graph()
    print(gra)
    graph_one = get_default_graph()
    graph_two = Graph()
    print(graph_one is get_default_graph())
    print(graph_two is get_default_graph())
    with graph_two.as_default():
        print(graph_two is get_default_graph())
    print(graph_two is get_default_graph())


if __name__ == "__main__":
    wrapper()
