import csv
from copy import deepcopy

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from shtmbss2.common.config import *
from shtmbss2.core.data import get_last_experiment_num, get_experiment_folder
from shtmbss2.core.logging import log


class LabelTypes(NamedStorage):
    LETTERS = "letters"
    NUMBERS = "numbers"
    COORDINATES = "coordinates"


EDGE_COLORS = {
    DendriteState.INACTIVE: Colors.GREY,
    DendriteState.WEAK: Colors.LIGHT_BLUE,
    DendriteState.PREDICTIVE: Colors.DARK_BLUE,
    DendriteState.DUPLICATE: Colors.PURPLE,
}

NODE_COLORS = {
    SomaState.INACTIVE: Colors.BLACK,
    SomaState.ACTIVE: Colors.RED,
}


class Map:
    def __init__(self, map_name, sequences, save_history=False, add_default_map_to_history=False):

        self.map_name = map_name
        self.sequences = sequences
        self.save_history = save_history

        self.grid_map = None
        self.size_x = None
        self.size_y = None
        self.symbol_nodes = None

        self.graph: nx.Graph = None
        self.graph_history = None
        self.nodes_default = None
        self.edges_default = None
        self.initialized = False

        self.init_map(add_default_map_to_history=add_default_map_to_history)

    def load_map(self, file_path):
        """
        Loads a 2D map from a csv file. The map is composed of characters separated by commas, defining the individual
        cells of the map. Each line constitutes a row and each new character a column in the map. An empty cell is
        defined by a 0, a traversable cell is defined by a letter of the alphabet.

        :param file_path: The path to the source csv file containing the map.
        :type file_path: str
        :return: Returns the 2D map as a numpy array.
        :rtype: np.ndarray
        """
        with open(file_path, 'r') as f:
            csv_reader = csv.reader(f)
            lines = list(csv_reader)

        i_line = 0
        while i_line < len(lines):
            try:
                line_arr = np.array(lines[i_line], dtype=np.int8)
            except ValueError as e:
                i_line += 1
                continue

            if (line_arr < 0).all():
                lines.pop(i_line)
            else:
                i_line += 1

        # set dimensions of map from number of rows (x), cols (y)
        self.size_x = len(lines)
        self.size_y = len(lines[0])

        return np.array(lines)

    def generate_edges(self, connections=None, default_edge_color="grey"):
        """
        Generates the edges for all connections defined in the sequences with the default edge color.
        If additional connections are specified, than the edge color is set based on the value in the connections dict.

        :param connections: Optional connections dictionary including the start/target of an edge and its edge color.
        :type connections: dict
        :param default_edge_color: The default edge color.
        :type default_edge_color: str
        :return:
        :rtype: None
        """
        # remove all existing edges from graph
        self.graph = nx.create_empty_copy(self.graph)
        self.symbol_nodes = dict()
        self.edges_default = dict()

        if connections is None:
            connections = dict()

        created = list()
        for sequence in self.sequences:
            for i_sym in range(len(sequence) - 1):
                sym_a = sequence[i_sym]
                sym_b = sequence[i_sym + 1]
                if (sym_a, sym_b) in connections.keys():
                    edge_color = connections[(sym_a, sym_b)]
                else:
                    edge_color = default_edge_color

                # check if symbols are already contained in symbol nodes
                if sequence[i_sym] in self.symbol_nodes.keys():
                    a_ids = self.symbol_nodes[sequence[i_sym]]
                else:
                    a_ids = np.where(self.grid_map == sequence[i_sym])
                    a_ids = list(zip(a_ids[0], a_ids[1]))
                    self.symbol_nodes[sequence[i_sym]] = a_ids
                if sequence[i_sym + 1] in self.symbol_nodes.keys():
                    b_ids = self.symbol_nodes[sequence[i_sym + 1]]
                else:
                    b_ids = np.where(self.grid_map == sequence[i_sym + 1])
                    b_ids = list(zip(b_ids[0], b_ids[1]))
                    self.symbol_nodes[sequence[i_sym + 1]] = b_ids
                # loop over all nodes that represent symbols a and b
                for x_a, y_a in a_ids:
                    for x_b, y_b in b_ids:
                        if np.linalg.norm(np.array([x_a, y_a]) - np.array([x_b, y_b])) >= 2:
                            continue
                        if (x_a, y_a, x_b, y_b) in created:
                            continue
                        log.debug(f"Adding edge {(x_a, y_a), (x_b, y_b)} with color={edge_color}")
                        self.graph.add_edge((x_a, y_a), (x_b, y_b), color=edge_color)
                        created.append((x_a, y_a, x_b, y_b))
                        self.edges_default[(sym_a, sym_b)] = DendriteState.INACTIVE

    def init_map(self, add_default_map_to_history=False):
        """
        Load the data and initialize the graph for the given map.

        :param add_default_map_to_history: Whether to add the initial default map to the start of the history or not.
        :type add_default_map_to_history: bool
        :return:
        :rtype: None
        """
        # retrieve file path for map name
        file_path = join(PATH_MAPS, f'map_{self.map_name}.csv')
        # retrieve 2d grid-map from file
        self.grid_map = self.load_map(file_path)
        # create directional graph (DiGraph) with a grid structure (m x n)
        self.graph = nx.grid_2d_graph(self.size_x, self.size_y, create_using=nx.DiGraph)
        # reset the color for all nodes to black
        self.graph.add_nodes_from(self.graph.nodes, color='black')
        # generate all default edges for potential connections
        self.generate_edges()
        # generate all nodes
        self.nodes_default = {k: SomaState.INACTIVE for k in self.symbol_nodes.keys()}
        # reset history
        if self.save_history:
            self.graph_history = list()
            if add_default_map_to_history:
                self.graph_history.append(deepcopy(self.graph))
        # mark initialization as final
        self.initialized = True

    def get_labels(self, pos, empty_label=""):
        """
        Determines traversable and empty nodes in the gridmap, generates a list of those and returns it together with
        the labels for the respective nodes.

        :param pos: The positions of the nodes in the graph.
        :type pos: dict
        :param empty_label: The character or string to be used for empty labels.
        :type empty_label: str
        :return: Dictionaries for the labels, traversable and empty nodes.
        :rtype: tuple
        """
        labels = dict()
        empty_pos = dict()
        actual_pos = dict()
        for key_i, pos_i in pos.items():
            symbol = self.grid_map[key_i[0]][key_i[1]].lstrip()
            if symbol.isnumeric():
                # we found empty space = 0
                empty_pos[key_i] = pos_i
                if empty_label is None:
                    symbol = int(symbol)
                else:
                    symbol = empty_label
            else:
                actual_pos[key_i] = pos_i
            labels[key_i] = symbol

        return labels, actual_pos, empty_pos

    def plot_graph(self, graph=None, only_traversable=True, arrows=False, label_type=LabelTypes.LETTERS,
                   empty_label="", ax=None, show_plot=True):
        """
        Plot the currently saved graph or a sub-graph of the traversable nodes.

        :param only_traversable: If true, a sub-graph with only traversable nodes shown is plotted.
        :type only_traversable: bool
        :param arrows: Whether to display arrows at the end of the edger or not.
        :type arrows: bool
        :param label_type: The type of the label, i.e. letters, numbers or coordinates.
        :type label_type: LabelTypes
        :param empty_label: The displayed string for an empty label.
        :type empty_label: str
        :return:
        :rtype: None
        """
        if graph is None:
            graph = self.graph
        # create positions (coordinates) for nodes of graph
        pos = {(x, y): (y, -x) for x, y in self.graph.nodes()}
        # retrieve all traversable nodes of the map
        labels_letters, actual_pos, empty_pos = self.get_labels(pos, empty_label=empty_label)
        if label_type == LabelTypes.LETTERS:
            labels = labels_letters
        elif label_type == LabelTypes.NUMBERS:
            labels = {(x, y): labels_letters[(x, y)] for x, y in self.graph.nodes()}
        elif label_type == LabelTypes.COORDINATES:
            labels = {(x, y): (x, y) for x, y in self.graph.nodes()}
        else:
            log.warning(f"Label type {label_type} does not exist. Aborting graph plotting.")
            return
        # set pos and graph depending on only_traversable
        if only_traversable:
            graph = graph.subgraph(actual_pos.keys())
            pos = actual_pos

        # retrieve edge colors for edges from subgraph
        edge_colors = nx.get_edge_attributes(graph, 'color').values()
        node_colors = nx.get_node_attributes(graph, 'color').values()
        # arrows = [True if c == "blue" else False for c in edge_colors]

        # create plot
        if ax is None:
            plt.figure(figsize=(self.size_y * 2, self.size_x * 2))

        # draw graph
        nx.draw(graph, pos=pos,
                node_color='white',
                node_size=4000,
                edge_color=edge_colors,
                width=4,
                edgecolors=node_colors,
                linewidths=4,
                with_labels=True,
                labels={k: labels[k] for k in pos.keys()},
                font_size=30,
                font_weight='bold',
                arrows=arrows,
                arrowstyle='->',
                arrowsize=30,
                ax=ax,
                )

        if show_plot:
            plt.show()

    def plot_graph_history(self, network, history_range=None, fps=1, only_traversable=True, arrows=False,
                           label_type=LabelTypes.LETTERS, empty_label="", title="Frame", show_plot=False,
                           experiment_num=None, save_plot=True):
        if history_range is None:
            history_range = list(range(len(self.graph_history)))

        fig, ax = plt.subplots(figsize=(self.size_y * 2, self.size_x * 2))

        def animate_graph(frame):
            if frame in history_range:
                ax.clear()
                self.plot_graph(graph=self.graph_history[frame], only_traversable=only_traversable, arrows=arrows,
                                label_type=label_type, empty_label=empty_label, ax=ax, show_plot=False)
                ax.set_title(f'{title.capitalize()} {frame}', fontsize=30)

        ani = FuncAnimation(fig, animate_graph, frames=len(self.graph_history), repeat=False)

        if save_plot:
            if experiment_num is None:
                experiment_num = (get_last_experiment_num(network, network.p.experiment.id, network.p.experiment.type)
                                  + 1)
            experiment_folder = get_experiment_folder(network, network.p.experiment.type, network.p.experiment.id,
                                                      experiment_num)

            if not os.path.exists(experiment_folder):
                os.makedirs(experiment_folder)

            file_path = join(experiment_folder,
                             f'graph_{network.network_state}.gif'
                             )
            ani.save(file_path, writer='pillow', fps=fps)

        if not show_plot:
            plt.close()

    def update_edge(self, start, target, **argv):
        """
        Function for updating an edge of a graph. The parameters can be arbitrarily defined by argv and are not checked.

        :param start: The start node of the edge.
        :type start: tuple
        :param target: The end node of the edge.
        :type target: tuple
        :param argv: The parameters to be updated.
        :type argv: dict
        :return:
        :rtype: None
        """
        if argv is None:
            log.warning(f"No parameters specified for updating edge {start} -> {target}.")
            return

        for param_name, param_value in argv.items():
            self.graph[start][target][param_name] = param_value

    def update_graph(self, nodes, edges):
        """
        Updates the saved graph based on the provided nodes and edges. This function does not remove anything, it only
        updates the colors (states) of the respective nodes and edges.

        :param nodes: The nodes (keys) to be updated with their new states (values).
        :type nodes: dict
        :param edges: The edges (keys) to be updated with their new states (values).
        :type edges: dict
        :return: None
        :rtype:
        """
        nodes = {**self.nodes_default, **nodes}
        edges = {**self.edges_default, **edges}

        # create a
        connections = dict()
        for edge_name, edge_state in edges.items():
            # for node_a_x, node_a_y in self.symbol_nodes[edge_name[0]]:
            #     for node_b_x, node_b_y in self.symbol_nodes[edge_name[1]]:
            connections[edge_name] = EDGE_COLORS[edge_state]
        self.generate_edges(connections=connections)

        # reset the color for all nodes to black
        self.graph.add_nodes_from(self.graph.nodes, color='black')
        # update the color for nodes in the provided dict
        for node_name, node_state in nodes.items():
            for node_x, node_y in self.symbol_nodes[node_name]:
                self.graph.nodes[(node_x, node_y)]["color"] = NODE_COLORS[node_state]

        if self.save_history:
            self.graph_history.append(deepcopy(self.graph))

    def reset_graph_history(self):
        self.graph_history = list()