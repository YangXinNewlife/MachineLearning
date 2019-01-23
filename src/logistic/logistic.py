# -*- coding:utf-8 -*-
__author__ = 'yangxin_ryan'

import numpy as np
import matplotlib.pyplot as plt


"""
logistic算法实现
"""
class Logistic(object):
    """
    加载数据集
    ：return：返回两个数组，普通数组
        data_arr -- 原始数据的特征
        label_arr -- 原始数据的标签，也就是每条样本对应的类别
    """
    def load_data_set(self):
        data_arr = []
        label_arr = []
        f = open("/xxx/TestSet.txt", 'r')
        for line in f.readlines():
            line_arr = line.strip().split()
            data_arr.append([1.0, np.float(line_arr[0]), np.float(line_arr[1])])
            label_arr.append(int(line_arr[2]))
        return data_arr, label_arr

    """
    处理数据溢出问题
    """
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


    """
    梯度上升法，其实就是因为使用了极大似然估计
    ：param data_arr:传入的就是一个普通的数组，当你传入一个二维的ndarray也行
    ：param class_labels: class_labels是类别标签，它是一个1 * 100 的行向量，
    为了便于矩阵计算，需要将行向量转换为列向量，做法是将原有的向量转置，再将它赋值给label_mat
    """
    def grad_ascent(self, data_arr, class_labels):
        data_mat = np.mat(data_arr)
        label_mat = np.mat(class_labels).transpose()
        m, n = np.shape(data_mat)
        alpha = 0.001
        max_cycles = 500
        weights = np.ones((n, 1))
        for k in range(max_cycles):
            h = self.sigmoid(data_mat * weights)
            error = label_mat - h
            weights = weights + alpha * data_mat.transpose() * error
        return weights

    """
    实现可视化部分
    :param weights
    """
    def plot_best_fit(self, weights):
        data_mat, label_mat = self.load_data_set()
        data_arr = np.array(data_mat)
        n = np.shape(data_mat)[0]
        x_cord1 = []
        y_cord1 = []
        x_cord2 = []
        y_cord2 = []
        for i in range(n):
            if int(label_mat[i]) == 1:
                x_cord1.append(data_arr[i, 1])
                y_cord1.append(data_arr[i, 2])
            else:
                x_cord2.append(data_arr[i, 1])
                y_cord2.append(data_arr[i, 2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
        ax.scatter(x_cord2, y_cord2, s=30, color='red', marker='s')
        x = np.arange(-3.0, 3.0, 0.1)
        y = (-weights[0] - weights * x) / weights[2]
        ax.plot(x, y)
        plt.xlabel('x1')
        plt.ylabel('y1')
        plt.show()

    """
    随机梯度上升算法，只是用一个样本点来更新回归系数
    :param data_mat: 输入数据的数据特征（除去最后一列），ndarray
    :param class_labels: 输入数据的列别标签（最后一列数据）
    """
    def stoc_grad_ascent0(self, data_mat, class_labels):
        m, n = np.shape(data_mat)
        alpha = 0.01
        weights = np.ones(n)
        for i in range(m):
            h = self.sigmoid(sum(data_mat[i] * weights))
            error = class_labels[i] - h
            weights = weights + alpha * error * data_mat[i]
        return weights


    """
    改进版本的随机梯度上升，使用随机的一个样本来更新回归系数
    :param data_mat:输入数据的数据特征（除去最后一列），ndarray
    :param class_labels: 输入数据的列别标签（最后一列数据）
    :param num_iter: 得带次数
    """
    def stoc_grad_ascent1(self, data_mat, class_labels, num_iter=150):
        m, n = np.shape(data_mat)
        weights = np.ones(n)
        for j in range(num_iter):
            data_index = list(range(m))
            for i in range(m):
                alpha = 4 / (1.0 + j + i) + 0.1
                rand_index = int(np.random.uniform(0, len(data_index)))
                h = self.sigmoid(np.sum(data_mat[data_index[rand_index]] * weights))
                error = class_labels[data_index[rand_index]] - h
                weights = weights + alpha * error * data_mat[data_index[rand_index]]
                del(data_index[rand_index])
        return weights

    """
    logistic算法应用
    从疝气病预测病马的死亡率
    最终的分类函数，根据回归系数和特征向量来计算Sigmod的值，大雨0.5函数返回1，否则返回0
    :param in_x: 特征向量
    :param weights:根据梯度下降/随机梯度下降 计算得倒的回归系数
    """
    def classify_vector(self, in_x, weights):
        prob = self.sigmoid(np.sum(in_x * weights))
        if prob > 0.5:
            return 1.0
        return 0.0

    """
    打卡测试集和训练集，并对数据进行格式化处理，其实最主要的部分，比如缺失值的补充
    """
    def colic_test(self):
        f_train = open("/../HorseColicTraning.txt", 'r')
        f_test = open("/../HorseColicTest.txt", 'r')
        training_set = []
        training_labels = []
        for line in f_train.readlines():
            curr_line = line.strip().split("\t")
            if len(curr_line) == 1:
                continue
            line_arr = [float(curr_line[i]) for i in range(21)]
            training_set.append(line_arr)
            training_labels.append(float(curr_line[21]))
        train_weights = self.stoc_grad_ascent1(np.array(training_set), training_labels, 500)
        error_count = 0
        num_test_vec = 0.0
        for line in f_test.readlines():
            num_test_vec += 1
            curr_line = line.strip().split("\t")
            if len(curr_line) == 1:
                continue
            line_arr = [float(curr_line[i]) for i in range(21)]
            if int(self.classify_vector(np.array(line_arr), train_weights)) != int(curr_line[21]):
                error_count += 1
        error_rate = error_count / num_test_vec
        print('the error rate is {}'.format(error_rate))
        return error_rate

    """
    调用 colicTest() 10次并求结果的平均值
    :return: nothing
    """
    def multi_test(self):
        num_tests = 10
        error_sum = 0
        for k in range(num_tests):
            error_sum += self.colic_test()
        print ("after {} iteration the average error rate is {}".format(num_tests, error_sum / num_tests))


if __name__ == '__main__':
    logistic = Logistic()
    logistic.colic_test()
    logistic.multi_test()


