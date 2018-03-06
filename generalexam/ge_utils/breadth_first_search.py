"""Search algorithms."""

import Queue

WHITE_COLOUR = 'white'
GREY_COLOUR = 'grey'
BLACK_COLOUR = 'black'
VALID_COLOURS = [WHITE_COLOUR, GREY_COLOUR, BLACK_COLOUR]


class BfsNode:
    """Node for breadth-first search."""

    DEFAULT_DISTANCE_FROM_ROOT = int(1e12)
    DEFAULT_ADJACENT_KEYS = []

    def __init__(
            self, colour=WHITE_COLOUR,
            distance_from_root=DEFAULT_DISTANCE_FROM_ROOT, parent_key=None,
            adjacent_keys=DEFAULT_ADJACENT_KEYS):
        """Creates new search node.

        :param colour: Colour (must be a string in `VALID_COLOURS`).  Indicates
            the exploration status of the node.
        :param distance_from_root: Distance from root node.
        :param parent_key: Key of parent node.
        :param adjacent_keys: 1-D list with keys of adjacent nodes.
        """

        self.colour = colour
        self.distance_from_root = distance_from_root
        self.parent_key = parent_key
        self.adjacent_keys = adjacent_keys

    def set_colour(self, colour):
        """Sets colour of node.

        :param colour: See documentation for constructor.
        """

        self.colour = colour

    def get_colour(self):
        """Returns colour.

        :return: colour: See documentation for constructor.
        """

        return self.colour

    def set_distance_from_root(self, distance_from_root):
        """Sets distance from root node.

        :param distance_from_root: Distance from root node.
        """

        self.distance_from_root = distance_from_root

    def get_distance_from_root(self):
        """Gets distance from root node.

        :return: distance_from_root: Distance from root node.
        """

        return self.distance_from_root

    def set_parent(self, parent_key):
        """Sets parent of node.

        :param parent_key: Key of parent node.
        """

        self.parent_key = parent_key

    def get_parent(self):
        """Returns parent of node.

        :return: parent_key: Key of parent node.
        """

        return self.parent_key

    def set_adjacency_list(self, adjacent_keys):
        """Sets adjacency list.

        :param adjacent_keys: 1-D list with keys of adjacent nodes.
        """

        self.adjacent_keys = adjacent_keys

    def get_adjacency_list(self):
        """Returns adjacency list.

        :return: adjacent_keys: 1-D list with keys of adjacent nodes.
        """

        return self.adjacent_keys


def run_bfs(bfs_node_dict, start_node_key, end_node_key):
    """Runs BFS.

    If BFS cannot reach the end node, returns None.

    :param bfs_node_dict: Dictionary, where each key is a string ID and each
        value is an instance of `BfsNode`.
    :param start_node_key: String ID for start node.
    :param end_node_key: String ID for end node.
    :return: visited_node_keys: 1-D list with keys of visited nodes (in order
        from start to end).
    """

    if start_node_key == end_node_key:
        return [start_node_key]

    bfs_node_dict[start_node_key].set_colour(GREY_COLOUR)
    bfs_node_dict[start_node_key].set_distance_from_root(0)

    node_queue = Queue.Queue()
    node_queue.put(start_node_key)

    while (bfs_node_dict[end_node_key].get_parent() is None
           and not node_queue.empty()):
        first_key = node_queue.get_nowait()
        these_adjacent_keys = bfs_node_dict[first_key].get_adjacency_list()

        for second_key in these_adjacent_keys:
            if second_key == end_node_key:
                bfs_node_dict[second_key].set_distance_from_root(
                    bfs_node_dict[first_key].get_distance_from_root() + 1)
                bfs_node_dict[second_key].set_parent(first_key)
                break

            if bfs_node_dict[second_key].get_colour() != WHITE_COLOUR:
                continue

            bfs_node_dict[second_key].set_colour(GREY_COLOUR)
            bfs_node_dict[second_key].set_distance_from_root(
                bfs_node_dict[first_key].get_distance_from_root() + 1)
            bfs_node_dict[second_key].set_parent(first_key)
            node_queue.put(second_key)

        bfs_node_dict[first_key].set_colour(BLACK_COLOUR)

    if bfs_node_dict[end_node_key].get_parent() is None:
        return None

    visited_node_keys = [end_node_key]
    while True:
        this_key = bfs_node_dict[visited_node_keys[-1]].get_parent()
        if this_key is None:
            break

        visited_node_keys.append(this_key)

    return visited_node_keys[::-1]
