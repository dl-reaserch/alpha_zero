# from smyAlphaZero import go
# import  numpy as np
# import az.preprocessing.preprocessing as pr
#
# def trans2move_matrix(board_size, move):
#     move_matrix = np.zeros([board_size * board_size + 1])
#     move_index = board_size * board_size if move == (board_size, board_size) else (move[0]) * board_size + move[1]
#     move_matrix[move_index] = 1
#     return move_matrix
#
# move_limit =18
# board_size = 9
# batch_xyzs = []
# is_end_of_game = False
# for game_batch_i in range(2):
#     state = go.GameState(size=board_size)
#     player_color = go.BLACK if game_batch_i % 2 == 0 else go.WHITE
#     opponent_color = -player_color
#
#     z = np.zeros([3])
#     xyzs = []
#     init_planes = pr.get_board_new(state)
#     s_t_planes = []
#     for i in range(8):
#         s_t_planes.append(init_planes[0])
#         s_t_planes.append(init_planes[1])
#
#     s_t = np.stack((init_planes[0], init_planes[1], init_planes[0], init_planes[1], init_planes[0], init_planes[1],
#                     init_planes[0], init_planes[1],
#                     init_planes[0], init_planes[1], init_planes[0], init_planes[1], init_planes[0], init_planes[1],
#                     init_planes[0], init_planes[1], np.ones([board_size, board_size])),
#                    axis=2)
#     moved = {}
#
#
#     max_leaf_value, move = 0.6,(5,5)
#     move_matrix = trans2move_matrix(board_size, move)
#
#     xyzs.append([s_t, move_matrix])
#     new_s_t_planes = s_t_planes
#     # print("game_batch_", game_batch_i, "--step_", 0, "--current_player:",current_player.player_flag ,"--choosed-position:", move,"--max_leaf_value:",max_leaf_value)
#     moved[move] = 1
#
#
#     for step_i in range(1, move_limit):
#         new_planes = pr.get_board_new(state)
#         new_s_t_planes = new_s_t_planes[2:len(new_s_t_planes)]
#
#         new_s_t_planes = np.append(new_s_t_planes, [new_planes[0]], axis=0)
#         new_s_t_planes = np.append(new_s_t_planes, [new_planes[1]], axis=0)
#
#         if state.get_current_player() == player_color:
#             # print("next_player:",current_player.player_flag,"--TTT:", 1)
#             s_t = np.append(new_s_t_planes, [np.ones([board_size, board_size])], axis=0)
#         else:
#             # print("next_player:", current_player.player_flag,"--TTT:", 0)
#             s_t = np.append(new_s_t_planes, [np.zeros([board_size, board_size])], axis=0)
#         s_t = np.stack(s_t, axis=2)
#         temperature = 1 if step_i <= 30 else 0.00001
#         max_leaf_value, move = 0.6,(5,5)
#         move_matrix = trans2move_matrix(board_size, move)
#         # print("game_batch_", game_batch_i, "--step_", step_i, "--current_player:", current_player.player_flag,"--choosed-position:", move,"--max_leaf_value:",max_leaf_value)
#
#         xyzs.append([s_t, move_matrix])
#
#         if max_leaf_value < 0.005:
#             if state.get_current_player() == player_color:
#                 z[2] = 1
#             else:
#                 z[0] = 1
#             break
#
#         else:
#             is_end_of_game = False
#             if step_i == move_limit - 1 or is_end_of_game:
#
#                 if state.get_winner() == player_color:
#                     z[0] = 1
#                 elif state.get_winner() == opponent_color:
#                     z[2] = 1
#                 else:
#                     z[1] = 1
#                 for s in xyzs:
#                     s.append(z)
#                 break
#
#     # 18,3
#
#
#     batch_xyzs.append(xyzs)
#
#
# # batch_xyzs = np.stack(batch_xyzs, axis=0)
# #
# # s_batch_xyzs = np.reshape(batch_xyzs, [-1, 3])
# # print("s_batch_xyzs_shape:",np.shape(s_batch_xyzs))
# # np.random.shuffle(s_batch_xyzs)
# # batch_size = 32
# # n_batchs = len(s_batch_xyzs) // batch_size
# ############
#
# print("s_batch_xyzs_shape1:",np.shape(batch_xyzs))
# s_batch_xyzs = np.reshape(batch_xyzs, (-1, 3))
# print("s_batch_xyzs_shape2:",np.shape(s_batch_xyzs))
# np.random.shuffle(s_batch_xyzs)
# batch_size = 32
# n_batchs = len(s_batch_xyzs) // batch_size
#
#
# def get_next_batches():
#     for batch_i in range(n_batchs + 1):
#         train_xyzs = s_batch_xyzs[batch_i * batch_size:(batch_i + 1) * batch_size]
#         yield list(zip(*train_xyzs))
#
#
# batch_gen = get_next_batches()
# # for batch_i in range(n_batchs + 1):
# #     next_batches = [*next(batch_gen)]
# #     print("train_next_batch_shape:", np.shape(next_batches[0]))
#     #pvnet.process(next_batches, model_json)
#
# # def get_next_batches():
# #     for batch_i in range(n_batchs + 1):
# #         train_xyzs = s_batch_xyzs[batch_i * batch_size:(batch_i + 1) * batch_size]
# #         yield list(zip(*train_xyzs))
# #
# #
# # batch_gen = get_next_batches()
# #
# for batch_i in range(n_batchs + 1):
#     next_batches = [*next(batch_gen)]
#     print("next_batches_shape:",np.shape(next_batches[0]))
#     print("next_batches_shape:", np.shape(next_batches[1]))
#     print("next_batches_shape:", np.shape(next_batches[2]))
#


