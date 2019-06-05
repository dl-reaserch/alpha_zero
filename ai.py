from smyAlphaZero import go
from smyAlphaZero import mcts


class MCTSPlayer(object):
    def __init__(self, value_function, policy_function, player_flag, playout_depth=100, n_playout=50,c_puct=3):
        self.player_flag = player_flag
        self.mcts = mcts.MCTS(value_function, policy_function, c_puct, playout_depth, n_playout)

    def get_move(self, state, temperature, moved,suicides):
        #current_player = state.get_current_player()
        sensiable_moves = [move for move in state.get_legal_moves(include_eyes=False)]
        # TODO
        if len(sensiable_moves) > 0:
            max_search_value, move = self.mcts.get_move(state, temperature, sensiable_moves)
            if state.is_legal(move):
                self.mcts.update_with_move(move)
                # print("legal action,!",move,"max_search_value:",max_search_value)
                return max_search_value, move
            else:

                # key = str(move) + "_" + str(current_player)
                # print("suicide-key:", key, "-- move:", move)
                #
                # if key in suicides.keys():
                #     value = suicides[key]
                #     print("not first suicide: action:", key, "--times:", value)
                #     if value == 5:
                #         print("current_player:", current_player,
                #               "--suicide: action overlimit max times ,choose Pass")
                #         return max_search_value, go.PASS_MOVE
                #     else:
                #         suicides[key] = value + 1
                # else:
                #     suicides[key] = 1
                #     print("first suicide: action:", key, "--times:", 1)

                return self.get_move(state, temperature, sensiable_moves,suicides)
        # No 'sensible' moves available, so do pass move
        return 1, go.PASS_MOVE
