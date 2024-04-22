# 卞庆朝
# 开发时间：2023/7/3 10:53
# import numpy as np
# # 从 .npy 文件中加载数组
# array = np.load('G:\pycharm\chap06_刀具磨损预测\PHM_2010\c1(31575000).npy')
# array1 = np.load('G:\pycharm\chap06_刀具磨损预测\PHM_2010\c1(31575000).npy')
# array2 = np.load('G:\pycharm\chap06_刀具磨损预测\PHM_2010\c1(31575000).npy')
# # 合并数组
# merged_array = np.concatenate((array,array1,array2), axis=0)
# # 打印数组
# print(type(array))
# print(array.shape)
# # 合并后并且交换第二维和三维的顺序
# # 交换第二和第三维度
# new_arr = np.transpose(merged_array, (0, 2, 1))
# print(new_arr.shape)
# # 保存数组
# # np.save('new_arr.npy', new_arr)

import numpy as np

# 创建一个形状为 (315, 1) 的示例数组
arr = np.array([[1], [2], [3], [4], [5]])
print(arr.shape)
print(arr)
# 使用 numpy.squeeze() 去除维度为1的维度
new_arr = np.squeeze(arr)

# 打印新数组的形状
print(new_arr.shape)  # 输出 (315,)
print(new_arr)

