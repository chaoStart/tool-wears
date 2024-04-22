import numpy
import pandas
from matplotlib import pyplot as plt
import os, sys
from sklearn.preprocessing import MinMaxScaler
data_path_c1 = 'G:\pycharm\chap06_刀具磨损预测\c1\c1'
data_path_c4 = 'G:\pycharm\chap06_刀具磨损预测\c4\c4'
data_path_c6 = 'G:\pycharm\chap06_刀具磨损预测\c6\c6'
# get file list
path_list_c1 = os.listdir(data_path_c1)
path_list_c4 = os.listdir(data_path_c4)
path_list_c6 = os.listdir(data_path_c6)
# 使用降采样
def get_downsample_data(path_list, data_path):
    sample_num = 5000  # 取样点数
    data_cn = numpy.zeros((315, 7, 5000))
    for i, file in enumerate(path_list):
        path = os.path.join(data_path, file)
        data_pd = pandas.read_csv(path)
        interval = len(data_pd.values) // sample_num  # 取样间隔
        data_np = data_pd.values[::interval][:sample_num]
        scaler = MinMaxScaler()
        data_np = scaler.fit_transform(data_np)  # (5000,7)
        data_cn[i, :, :] = data_np.transpose()  # (7,5000)
        print('data_cn返回的数据类型:', data_cn.shape)
        print(file)
    return data_cn
# 从中间裁剪
def get_clip_data(path_list, data_path):
    data_cn = numpy.zeros((315, 7, 5000))
    for i, file in enumerate(path_list):
        path = os.path.join(data_path, file)
        data = pandas.read_csv(path)
        start = int(len(data) * 0.5)
        data = data.iloc[start:start + 5000, :].values
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        data_cn[i, :, :] = data.transpose()
        print(file)
        print('data_cn返回的数据类型:',data_cn.shape)
    return data_cn
def sample1():
    data_c1 = get_clip_data(path_list_c1, data_path_c1)
    data_c4 = get_clip_data(path_list_c4, data_path_c4)
    data_c6 = get_clip_data(path_list_c6, data_path_c6)
    numpy.save('G:\pycharm\chap06_刀具磨损预测\PHM_2010', data_c1)
    numpy.save('G:\pycharm\chap06_刀具磨损预测\PHM_2010', data_c4)
    numpy.save('G:\pycharm\chap06_刀具磨损预测\PHM_2010', data_c6)
def sample2():
    data_c1 = get_downsample_data(path_list_c1, data_path_c1)
    data_c4 = get_downsample_data(path_list_c4, data_path_c4)
    data_c6 = get_downsample_data(path_list_c6, data_path_c6)
    numpy.save('G:\pycharm\chap06_刀具磨损预测\PHM_2010\c1(31575000).npy', data_c1)
    numpy.save('G:\pycharm\chap06_刀具磨损预测\PHM_2010\c4(31575000).npy', data_c4)
    numpy.save('G:\pycharm\chap06_刀具磨损预测\PHM_2010\c6(31575000).npy', data_c6)
if __name__ == '__main__':
    # sample1()
    sample2()