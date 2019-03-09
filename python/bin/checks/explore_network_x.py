import networkx as nx
import matplotlib.pyplot as plt

G=nx.Graph()

for i in range(8):
    G.add_node(i+1)

G.add_edge(1,2)
G.add_edge(1,3)
G.add_edge(2,3)


G.add_edge(4,5)
G.add_edge(4,6)

G.add_edge(5,6)

G.add_edge(1,6)
G.add_edge(2,6)
G.add_edge(3,6)

G.add_edge(7,8)

nx.draw(G)

plt.show()

cliques = list(nx.find_cliques(G))
print("Number", nx.graph_number_of_cliques(G))

for clique in cliques:
    print("Hello, world!")
    print(cliques)
