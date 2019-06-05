import  os
import  numpy as np

# DS =[]
# D =[]
#
#
# for i in range(100):
#     if i >= 15:
#         if i % 2 ==1 :
#             D.append((i-1,i,i-3,i-2,i-5,i-4,i-7,i-6,i-9,i-8,i-11,i-10,i-13,i-12,i-15,i-14,"BLACK"))
#             DS.append(D)
#             D =[]
#
# ds = [D for D in DS]
# print(ds)

states = []
DS = []
D = []
collect_i = 0
for step_i in range(9*9*2):

    if step_i >= 8:
        new_states = states[collect_i:]
        for new_state in new_states:
            D.append(new_state[0])
            D.append(new_state[1])

        # if step_i % 2 == 0:
        #     D.append("1")
        # else:
        #     D.append("0")
        DS.append(D)
        D=[]
        collect_i =collect_i + 2
    # 落子
    states.append(("Black_" + str(step_i), "White_" + str(step_i)))

for D in DS:
    print(D)
