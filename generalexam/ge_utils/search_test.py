"""Unit tests for search.py."""

import copy
import unittest
from generalexam.ge_utils import search

NODE_KEYS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K']
BFS_NODE_DICT_PATH_TO_GOAL = {}

for this_key in NODE_KEYS:
    BFS_NODE_DICT_PATH_TO_GOAL.update({this_key: search.BfsNode()})

BFS_NODE_DICT_PATH_TO_GOAL['A'].set_adjacency_list(['B', 'C', 'J'])
BFS_NODE_DICT_PATH_TO_GOAL['B'].set_adjacency_list(['A', 'C', 'H', 'K'])
BFS_NODE_DICT_PATH_TO_GOAL['C'].set_adjacency_list(['A', 'B', 'D', 'F'])
BFS_NODE_DICT_PATH_TO_GOAL['D'].set_adjacency_list(['C', 'E', 'G', 'H'])
BFS_NODE_DICT_PATH_TO_GOAL['E'].set_adjacency_list(['D', 'F'])
BFS_NODE_DICT_PATH_TO_GOAL['F'].set_adjacency_list(['C', 'E', 'H', 'J'])
BFS_NODE_DICT_PATH_TO_GOAL['G'].set_adjacency_list(['D', 'H'])
BFS_NODE_DICT_PATH_TO_GOAL['H'].set_adjacency_list(['B', 'D', 'F', 'G', 'K'])
BFS_NODE_DICT_PATH_TO_GOAL['J'].set_adjacency_list(['A', 'F'])
BFS_NODE_DICT_PATH_TO_GOAL['K'].set_adjacency_list(['B', 'H'])

START_NODE_KEY = 'A'
END_NODE_KEY = 'G'
EXPECTED_VISITED_KEYS = ['A', 'B', 'H', 'G']

BFS_NODE_DICT_NO_PATH_TO_GOAL = copy.deepcopy(BFS_NODE_DICT_PATH_TO_GOAL)
BFS_NODE_DICT_NO_PATH_TO_GOAL['H'].set_adjacency_list(['B', 'D', 'F', 'K'])
BFS_NODE_DICT_NO_PATH_TO_GOAL['D'].set_adjacency_list(['C', 'E', 'H'])


class SearchTests(unittest.TestCase):
    """Each method is a unit test for search.py."""

    def test_breadth_first_search_path_to_goal(self):
        """Ensures correct output from breadth_first_search.

        In this case there is a path to the goal, which BFS is expected to find.
        """

        these_visited_keys = search.breadth_first_search(
            bfs_node_dict=BFS_NODE_DICT_PATH_TO_GOAL,
            start_node_key=START_NODE_KEY, end_node_key=END_NODE_KEY)
        self.assertTrue(these_visited_keys == EXPECTED_VISITED_KEYS)

    def test_breadth_first_search_no_path_to_goal(self):
        """Ensures correct output from breadth_first_search.

        In this case there is no path to the goal.
        """

        these_visited_keys = search.breadth_first_search(
            bfs_node_dict=BFS_NODE_DICT_NO_PATH_TO_GOAL,
            start_node_key=START_NODE_KEY, end_node_key=END_NODE_KEY)
        self.assertTrue(these_visited_keys is None)


if __name__ == '__main__':
    unittest.main()
