import torch.nn as nn
import graphviz
import numpy as np


def mlp(
        input_size,
        layer_sizes,
        output_size,
        output_activation=nn.Identity,
        activation=nn.ReLU,
        momentum=0.1,
        init_zero=False,
):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                       act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return nn.Sequential(*layers)


def plot_node(V_est, traj_nodes, leaf_node, graph: graphviz.Digraph, parent_node, index, min_max_stats, parent_name=None):
    def node_name(node, index):
        node_name = f'{index}\n' \
                    f'V = {min_max_stats.normalize(node.mean_value()) if node.num_visits > 0 else 0.0:.3f}\n' \
                    f'N = {node.num_visits}'

        if node == leaf_node:
            node_name += f'\nV_est={V_est:.3f}'
        return node_name

    def edge_label(from_node, to_node):
        action = to_node.action
        return f'a = {to_node.action}\n' \
               f'r = {to_node.reward}\n' \
               f'PUCT = {from_node.puct_scores(min_max_stats)[action]:.3f}\n' \
               f'P = {from_node.child_priors[action]:.3f}'

    if parent_name is None:
        parent_name = node_name(parent_node, index)

    if parent_node in traj_nodes:
        graph.node(parent_name, color='red')
    elif parent_node.terminal:
        graph.node(parent_name, color='blue')
    else:
        graph.node(parent_name)
    child_names = []
    for i, (_, child_node) in enumerate(parent_node.children.items()):
        child_name = node_name(child_node, index + i + 1)
        child_names.append(child_name)
        graph.node(child_name)
        graph.edge(parent_name, child_name, label=edge_label(parent_node, child_node))

    index += len(parent_node.children.keys()) + 1

    for i, (_, child_node) in enumerate(parent_node.children.items()):
        child_name = child_names[i]
        index = plot_node(V_est, traj_nodes, leaf_node, graph, child_node, index, min_max_stats, parent_name=child_name)

    return index


def plot_tree(root_node, leaf_node, V_est, min_max_stats):
    g = graphviz.Digraph('g', filename='tree.gv', node_attr={'shape': 'circle'})
    nodes = []
    parent = leaf_node
    while parent is not None:
        nodes.append(parent)
        parent = parent.parent_traversed
    plot_node(V_est, nodes, leaf_node, g, root_node, 0, min_max_stats)
    g.view()


class MinMaxStats:
    # See: https://arxiv.org/pdf/1911.08265.pdf, p.12
    def __init__(self):
        self.max_delta = 0.01
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value: float):
        if value is None:
            raise ValueError

        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        delta = self.maximum - self.minimum
        if delta < self.max_delta:   # See: EfficientZero implementation
            value_norm = (value - self.minimum) / self.max_delta
        else:
            value_norm = (value - self.minimum) / delta
        return value_norm


class DiscreteSupport:
    def __init__(self, min: int, max: int, delta=1.0):
        assert min < max
        self.min = min
        self.max = max
        self.range = np.arange(min, max + delta, delta)
        self.size = len(self.range)
        self.delta = delta
