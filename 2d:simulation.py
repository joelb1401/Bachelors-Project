from percolation_utils import *

size = 50
graph = lattice_2d(size)

fig1, p_c = plot_phase_transition(graph, "$\mathbb{Z}^2$ Lattice", n_points=50, n_trials=300)
print(f"p_c = {p_c:.3f} (known: 0.500, error: {abs(p_c - 0.5):.3f}) ✓")
plt.show()

small_graph = lattice_2d(50)
fig2 = plot_percolation_configs(small_graph, None, "$\mathbb{Z}^2$ Configurations",
                                pos={n: n for n in small_graph.nodes()}, p_c=p_c)
plt.show()
