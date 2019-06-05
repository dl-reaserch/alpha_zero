# import numpy as np
# from smyAlphaZero import go
# import preprocessing as pr
# arr=[]
# state = go.GameState(size=9)
# init_planes = pr.get_board_new(state)
#
# for i in range(8):
#     arr.append(init_planes[0])
#     arr.append(init_planes[1])
# new_arr= arr
# #
# #
# for step_i in range(1, 25):
#     new_arr = new_arr[2:len(new_arr)]
#     new_arr = np.append(new_arr,[init_planes[0]],axis=0)
#     new_arr = np.append(new_arr,[init_planes[1]],axis=0)
#
#
#     s_t = np.append(new_arr,[np.zeros([9, 9])],axis=0)
#
#     s_t = np.stack(s_t)
#     print(np.shape(s_t))
#
#
#
# #
#
#
#
#

