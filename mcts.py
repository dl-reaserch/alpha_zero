import numpy as np
import az.preprocessing.preprocessing as pr
from smyAlphaZero import go
import sys


class TreeNode(object):
    def __init__(self, parent, prior_p):
        sys.setrecursionlimit(1000000)
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = prior_p
        self._p = prior_p


    def is_leaf(self):
        """
            Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None

    def get_value(self):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
               this node's prior adjusted for its visit count, u
        """
        return self._Q + self._u

    # def get_transed_matrix(state,isChange=False):
    #     if not isChange:
    #         return state
    #     b = np.fliplr(state)  # 左右翻转 [3 2 1 6 5 4 9 8 7]
    #     c = np.flipud(state)  # 上下翻转 [7 8 9 4 5 6 1 2 3]
    #     d = np.rot90(state)  # 90 [3 6 9 2 5 8 1 4 7]
    #     e = np.rot90(np.rot90(state))  # 180 [9 8 7 6 5 4 3 2 1]
    #     f = np.rot90(np.rot90(np.rot90(state)))  # 270 [7 4 1 8 5 2 9 6 3]
    #     g = np.rot90(c)  # 左右翻转再旋转[1 4 7 2 5 8 3 6 9]
    #     h = np.rot90(d)  # 上下翻转再旋转 [9 6 3 8 5 2 7 4 1]
    #     map = {1: state, 2: b, 3: c, 4: d, 5: e, 6: f, 7: g, 8: h}
    #     key = np.random.randint(1, 9)
    #     return map[key]

    def expand_and_evaluate(self, s_t, value_fn, policy_fn, sensiable_moves):
        """Expand tree by creating new children.
        Arguments:
        action_priors -- output from policy function - a list of tuples of actions and their prior
            probability according to the policy function.

        Returns:
        None
        """
        # state = self.get_transed_matrix(state,True)
        _y = policy_fn([s_t])

        p = np.reshape(_y,-1)
        pass_prob, _p = p[-1], p[:-1]
        _p = np.reshape(_p, [go.size, go.size])
        self._children[(go.size,go.size)] = TreeNode(self, pass_prob)
        for x, probs in enumerate(_p):
            for y, prob in enumerate(probs):
                if (x,y) in sensiable_moves:
                    self._children[(x,y)] = TreeNode(self, prob)
                    # if self.is_root():
                    #     print("add_root_childe:","--action:",(x,y),"--prob",prob)
        return value_fn([s_t])

    def select(self):
        """Select action among children that gives maximum action value, Q plus bonus u(P).
            Returns:
            A tuple of (action, next_node)
        """
        if self.is_root():
            # a = 0.25 * np.random.dirichlet(0.03)
            # np.random.d
            noise = np.random.normal(loc=0.0, scale=0.03, size=82)
            self._p = 0.75 * self._p + 0.25 * noise
        # act_node: <smyAlphaZero.mcts.TreeNode object at 0x00000000120B6080> act_node_value: Tensor("add_5:0", shape=(82,), dtype=float32)
        # if self.is_root():
        #     for k,v in self._children.items():
        #         print("root_child_action:",k,"--v:",v.get_value())
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value())

    def update_recursive(self, leaf_value, c_puct):
        if self._parent:
            self._parent.update_recursive(leaf_value, c_puct)
        self.update(leaf_value, c_puct)

    def update(self, leaf_value, c_puct):
        """Update node values from leaf evaluation.
        Arguments:
        leaf_value -- the value of subtree evaluation from the current player's perspective.
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.

        Returns:
        None
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += (leaf_value - self._Q) / self._n_visits
        if not self.is_root():
            self._u = c_puct * self._p * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)


class MCTS(object):
    def __init__(self, value_fn, policy_fn, c_puct=5, playout_depth=2, n_playout=2, player_color=go.BLACK):
        """Arguments:
                value_fn -- a function that takes in a state and ouputs a score in [-1, 1], i.e. the
                    expected value of the end game score from the current player's perspective.
                policy_fn -- a function that takes in a state and outputs a list of (action, probability)
                    tuples for the current player.
                c_puct -- a number in (0, inf) that controls how quickly exploration converges to the
                    maximum-value policy, where a higher value means relying on the prior more, and
                    should be used only in conjunction with a large value for n_playout.
                """
        self._root = TreeNode(None, 1.0)
        self._value_fn = value_fn
        self._policy_fn = policy_fn
        self._c_puct = c_puct
        self._L = playout_depth
        self._n_playout = n_playout
        self.player_color = player_color

    def get_move(self, state, temperature,sensiable_moves):

        #print("self._root:", self._root)
        """Runs all playouts sequentially and returns the most visited action.
            Arguments:
            state -- the current state, including both game state and the current player.

            Returns:
            the selected action
        """
        leaf_values = []
        for n in range(self._n_playout):
            #print("_playout_",n)
            state_copy = state.copy()
            leaf_values.append(self._playout(state_copy, self._L,sensiable_moves))
        # chosen action is the *most visited child*, not the highest-value one
        # (they are the same as self._n_playout gets large).
        beta = 1.0 / temperature
        _children = {}
        for key,val in self._root._children.items():
            if key in sensiable_moves:
                _children[key] = val
        if len(_children) == 0:
            return 1,go.PASS_MOVE
        else:
            max_leaf_value,move = max(leaf_values), \
                                  max(_children.items(),
                                         key=lambda act_node: np.power(act_node[1]._n_visits, beta) / np.power(
                                             self._root._n_visits, beta))[0]
            move = go.PASS_MOVE if move ==(go.size,go.size) else move
            return max_leaf_value,move

    def _playout(self, state, leaf_depth,sensiable_moves):
        """Run a single playout from the root to the given depth, getting a value at the leaf and
            propagating it back through its parents. State is modified in-place, so a copy must be
            provided.

            Arguments:
            state -- a copy of the state.
            leaf_depth -- after this many moves, leaves are evaluated.

            Returns:
            None
        """

        leaf_value = 0
        node = self._root
        # init
        init_planes = pr.get_board_new(state)
        s_t_planes = []
        for i in range(8):
            s_t_planes.append(init_planes[0])
            s_t_planes.append(init_planes[1])
        new_s_t_planes = s_t_planes
        for i in range(leaf_depth):
            #print("leaf_depth_",i)
            # Only expand node if it has not already been done. Existing nodes already know their
            # prior.
            new_planes = pr.get_board_new(state)
            new_s_t_planes = new_s_t_planes[2:len(new_s_t_planes)]
            new_s_t_planes = np.append(new_s_t_planes, [new_planes[0]], axis=0)
            new_s_t_planes = np.append(new_s_t_planes, [new_planes[1]], axis=0)
            if state.get_current_player() == self.player_color:
                s_t = np.append(new_s_t_planes, [np.ones([state.size, state.size])], axis=0)
            else:
                s_t = np.append(new_s_t_planes, [np.zeros([state.size, state.size])], axis=0)
            s_t = np.stack(s_t, axis=0)
            s_t = np.reshape(s_t, [go.size, go.size, 17])

            if node.is_leaf():
                # action_probs = self._policy(state)

                leaf_value = node.expand_and_evaluate(s_t, self._value_fn, self._policy_fn, sensiable_moves)
                #print("leaf_depth_",i,"--leaf_value:",leaf_value)

                # Check for end of game.
            # Greedily select next move.

            action, node = node.select()
            #print("leaf_depth_",i,"select:",action,"--value:",node.get_value())
            if state.is_legal(action):
                state.do_move(action)
        # Update value and visit count of nodes in this traversal.

        node.update_recursive(leaf_value, self._c_puct)
        return leaf_value

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about the subtree, assuming
            that get_move() has been called already. Siblings of the new root will be garbage-collected.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)
