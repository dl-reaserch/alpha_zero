import numpy as np
#
# a = np.matrix('1,2,3;4,5,6;7,8,9')
# c = np.fliplr(a) # 左右翻转 [3 2 1 6 5 4 9 8 7]
# d = np.flipud(a) # 上下翻转 [7 8 9 4 5 6 1 2 3]
# e = np.rot90(a)  # 90 [3 6 9 2 5 8 1 4 7]
# f = np.rot90(np.rot90(a)) #180 [9 8 7 6 5 4 3 2 1]
# g = np.rot90(np.rot90(np.rot90(a))) #270 [7 4 1 8 5 2 9 6 3]
# h = np.rot90(c) #左右翻转再旋转[1 4 7 2 5 8 3 6 9]
# i = np.rot90(d) #上下翻转再旋转 [9 6 3 8 5 2 7 4 1]

np.random.randint(1,9)
planes = np.zeros((1, 3,3))

a = np.matrix('1,-1,1;-1,0,0;0,0,0')

planes[0,:,:] = a == -1


print(planes)
