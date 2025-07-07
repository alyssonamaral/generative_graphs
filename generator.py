import torch
from cycles import simplegraph_to_nx
import matplotlib.pyplot as plt
import networkx as nx

model = torch.load("model.pth", weights_only=False)
model.eval()

for i in range(5):
    g = model()
    nx_g = simplegraph_to_nx(g)
    plt.figure(figsize=(4,4))
    nx.draw_circular(nx_g, with_labels=True)
    plt.title(f"Grafo gerado #{i+1}")
    plt.show()
