from smyAlphaZero.go import GameState
from smyAlphaZero.mcts import MCTS,TreeNode
from operator import itemgetter
import numpy as np
import unittest
class TestTreeNode(unittest.TestCase):
    def setUp(self):
        self.gs = GameState()
        self.node = TreeNode(None,1.0)

    def test_selection(self):
        self.node.expand_and_evaluate(dummy_policy((self.gs)))
        action,next_node = self.node.select()
        self.assertEqual(action,(18,18))
        self.assertIsNone(next_node)
        self.assertIsNotNone(next_node)

dummy_distribution = np.arange(361,dtype=np.float)
dummy_distribution = dummy_distribution/dummy_distribution.sum()


def dummy_policy(state):
    moves = state.get_legal_moves(include_eyes=False)
    return zip(moves,dummy_distribution)



