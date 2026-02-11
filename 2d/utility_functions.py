import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def bond_percolation(graph, p, n_trials=500):
    """
    Run bond percolation simulation on a 2D lattice.

    Args:
        graph: 2D lattice NetworkX graph
        p: Probability of keeping each edge (0 to 1)
        n_trials: Number of Monte Carlo trials to run

    Returns:
        (spanning cluster probability psi(p))
    """
    percolate_count = 0
    n = len(graph.nodes())

    # 2D lattice bounds
    max_x = max(node[0] for node in graph.nodes())
    max_y = max(node[1] for node in graph.nodes())

    for _ in range(n_trials):
        # Randomly keep edges with probability p
        open_edges = [e for e in graph.edges() if np.random.random() < p]

        # Create graph with open edges only
        g = nx.Graph()
        g.add_nodes_from(graph.nodes())
        g.add_edges_from(open_edges)

        # Find connected components
        components = list(nx.connected_components(g))

        # Check if any component spans
        spans = any(_check_spanning_2d(comp, max_x, max_y) for comp in components)

        if spans:
            percolate_count += 1

    return percolate_count / n_trials


def _check_spanning_2d(component, max_x, max_y):
    """
    Check if a component spans the 2D lattice.

    Args:
        component: Set of nodes in the component
        max_x: Maximum x coordinate
        max_y: Maximum y coordinate

    Returns:
        True if component spans, False otherwise
    """
    if not component:
        return False

    # Get all coordinates in component
    x_coords = [node[0] for node in component]
    y_coords = [node[1] for node in component]

    # Spans if touches both x-boundaries OR both y-boundaries
    return ((min(x_coords) == 0 and max(x_coords) == max_x) or
            (min(y_coords) == 0 and max(y_coords) == max_y))


def find_threshold(graph, n_trials=500):
    """
    Find percolation threshold p_c using binary search.

    Args:
        graph: 2D lattice NetworkX graph
        n_trials: Number of trials per probability value

    Returns:
        Estimated p_c rounded to 3 decimal places
    """
    p_min, p_max = 0.0, 1.0

    while p_max - p_min > 0.001:
        p_mid = (p_min + p_max) / 2
        prob = bond_percolation(graph, p_mid, n_trials)

        if prob < 0.5:
            p_min = p_mid
        else:
            p_max = p_mid

    return round((p_min + p_max) / 2, 3)


def lattice_2d(size):
    """
    Create a 2D square lattice graph.

    Args:
        size: Dimension of the lattice (creates size x size grid)

    Returns:
        NetworkX grid graph with nodes at (x, y) coordinates
    """
    return nx.grid_2d_graph(size, size, periodic=False)


def plot_phase_transition(graph, title, n_points=15, n_trials=300):
    """
    Plot percolation phase transition curve.

    Args:
        graph: 2D lattice NetworkX graph
        title: Overall title for the figure
        n_points: Number of p values to test
        n_trials: Monte Carlo trials per p value

    Returns:
        (matplotlib figure, estimated p_c)
    """
    # Find critical probability using binary search
    p_c = find_threshold(graph, n_trials)

    p_vals = np.linspace(0, 1, n_points)
    perc_probs = []

    for p in p_vals:
        perc_prob = bond_percolation(graph, p, n_trials)
        perc_probs.append(perc_prob)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=300)

    # Plot percolation probability
    ax.plot(p_vals, perc_probs, 'o-', linewidth=2, markersize=5,
            color='#2E86AB', markerfacecolor='#2E86AB',
            markeredgecolor='white', markeredgewidth=0.5,
            label='Percolation probability')

    # Threshold line at 0.5
    ax.axhline(0.5, color='#E63946', linestyle='--', linewidth=1.5,
               alpha=0.8, label='Percolation threshold')

    # Critical probability line
    ax.axvline(p_c, color='#06A77D', linestyle=':', linewidth=2,
               alpha=0.8, label=f'$p_c$ = {p_c:.3f}')

    # Axis labels
    ax.set_xlabel('Edge probability $p$', fontsize=13, fontweight='normal')
    ax.set_ylabel('Probability of Existence of Spanning Cluster $\\psi(p)$', fontsize=13, fontweight='normal')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Set axis limits
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(fontsize=11, framealpha=0.95, loc='best', edgecolor='gray', fancybox=False)

    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, direction='in',
                   top=True, right=True, length=5)

    plt.tight_layout()

    return fig, p_c


def plot_percolation_configs(graph, p_values, title, pos=None, p_c=None):
    """
    Visualize percolation configurations at different edge probabilities.

    Args:
        graph: 2D lattice NetworkX graph
        p_values: List of p values to show (if None, uses p_c ± 0.05)
        title: Figure title
        pos: Node positions (if None, uses actual lattice positions)
        p_c: Critical probability (used to auto-generate p_values)

    Returns:
        matplotlib figure
    """
    # Auto-generate p_values if not provided
    if p_values is None and p_c is not None:
        p_values = [p_c - 0.05, p_c, p_c + 0.05]

    n_configs = len(p_values)
    fig, axes = plt.subplots(1, n_configs, figsize=(5.5 * n_configs, 5.5), dpi=300)

    if n_configs == 1:
        axes = [axes]

    # Default to lattice positions
    if pos is None:
        pos = {node: node for node in graph.nodes()}

    max_x = max(node[0] for node in graph.nodes())
    max_y = max(node[1] for node in graph.nodes())

    # Create visualization for each p value
    for ax, p in zip(axes, p_values):
        # Generate random configuration
        open_edges = [e for e in graph.edges() if np.random.random() < p]
        g = nx.Graph()
        g.add_nodes_from(graph.nodes())
        g.add_edges_from(open_edges)

        # Find spanning cluster
        components = list(nx.connected_components(g))
        spanning_cluster = set()
        for component in components:
            if _check_spanning_2d(component, max_x, max_y):
                spanning_cluster = component
                break

        node_colors = ['#C41E3A' if node in spanning_cluster else '#CCCCCC'
                       for node in graph.nodes()]

        # Draw closed edges
        closed_edges = [e for e in graph.edges() if e not in open_edges]
        nx.draw_networkx_edges(graph, pos, edgelist=closed_edges,
                               edge_color='#E0E0E0', width=0.2, alpha=0.3, ax=ax)

        # Draw open edges
        nx.draw_networkx_edges(g, pos, edgelist=open_edges,
                               edge_color='#2C2C2C', width=1.2, alpha=0.7, ax=ax)

        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                               node_size=50, alpha=0.95, linewidths=0, ax=ax)

        ax.set_title(f'$p = {p:.3f}$', fontsize=13, fontweight='bold', pad=5)

        # Equal aspect ratio and remove axes
        ax.set_aspect('equal')
        ax.axis('off')

    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig
