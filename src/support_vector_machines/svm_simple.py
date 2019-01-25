# -*- coding:utf-8 -*-
from numpy import *
import random
import matplotlib.pyplot as plt


class SVMSimple(object):

    """
    对文件进行逐行解析，从而得到第行的累标签和整体特征矩阵
    """
    def load_data_set(self, file_name):
        data_mat = []
        label_mat = []
        fr = open(file_name)
        for line in fr.readlines():
            line_arr = line.strip().split("\t")
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[2]))
        return data_mat, label_mat

    """
    随机选择一个整数
    """
    def select_jrand(self, i, m):
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j

    """
    aj 目标值
    h 最大值
    l 最小值
    """
    def clip_alpha(self, aj, h, l):
        if aj > h:
            aj = h
        if l > aj:
            aj = l
        return aj

    """
    data_mat_in 数据集
    class_label 类别标签
    c 松弛变亮
    toler 错误率
    max_iter 推出前最大的循环次数
    """
    def smo_simple(self, data_mat_in, class_labels, c, toler, max_iter):
        data_matrix = mat(data_mat_in)
        label_mat = mat(class_labels).transpose()
        m, n = shape(data_matrix)
        b = 0
        alphas = mat(zeros((m, 1)))
        iter = 0
        while (iter < max_iter):
            alpha_pairs_changed = 0
            for i in range(m):
                fxi = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
                ei = fxi - float(label_mat[i])
                if ((label_mat[i] * ei < -toler) and (alphas[i] < c)) or ((label_mat[i] * ei > toler) and (alphas[i] > 0)):
                    j = self.select_jrand(i, m)
                    fxj = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                    rej = fxj - float(label_mat[j])
                    ej = fxj - float(label_mat[j])
                    alpha_iold = alphas[i].copy()
                    alpha_jold = alphas[j].copy()
                    if label_mat[i] != label_mat[j]:
                        l = max(0, alphas[j] - alphas[i])
                        h = min(c, c + alphas[j] - alphas[i])
                    else:
                        l = max(0, alphas[j] + alphas[i] - c)
                        h = min(c, alphas[j] + alphas[i])
                    if l == j:
                        print("L==H")
                        continue
                    eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - data_matrix[i, :] * data_matrix[i, :].T - data_matrix[j, :] * data_matrix[j, :].T
                    if eta >= 0:
                        print("eta >= 0")
                        continue

                    alphas[j] -= label_mat[j] * (ei - ej) / eta
                    alphas[j] = self.clipAlpha(alphas[j], h, l)
                    if abs(alphas[j] - alpha_jold) < 0.00001:
                        print("j not mobing enough")
                        continue
                    alphas[i] += label_mat[j] * label_mat[i] * (alpha_jold - alphas)
                    b1 = b - ei - label_mat[i] * (alphas[i] - alpha_iold) * data_matrix[i, :] * data_matrix[i, :].T - label_mat[j] * (alphas[j] - alpha_jold) * data_matrix[i, :] * data_matrix[j, :].T
                    b2 = b - ej - label_mat[i] * (alphas[i] - alpha_iold) * data_matrix[i, :] * data_matrix[j, :].T - label_mat[j] * (alphas[j] - alpha_jold) * data_matrix[j, :] * data_matrix[j, :].T
                    if (0 < alphas[i]) and (c > alphas[i]):
                        b = b1
                    elif (0 < alphas[j]) and (c > alphas[j]):
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alpha_pairs_changed += 1
                    print("iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
            if 0 == alpha_pairs_changed:
                iter += 1
            else:
                iter = 0
            print("iteration number: %d" % iter)
        return b, alphas

    """
    基于alpha计算W值
    """
    def calc_ws(self, alphas, data_arr, class_labels):
        x = mat(data_arr)
        label_mat = mat(class_labels).transpose()
        m, n = shape(x)
        w = zeros((n, 1))
        for i in range(m):
            w += multiply(alphas[i] * label_mat[i], x[i, :].T)
        return w

    def plotfig_svm(self, x_mat, y_mat, ws, b, alphas):
        y_mat = mat(x_mat)
        y_mat = mat(y_mat)
        b = array(b)[0]
        fig = plt.figure()
        ax = fig.add_subploy(111)
        ax.scatter(x_mat[:, 0].flatten().a[0], x_mat[:, 1].flatten().A[0])
        x = arange(-1, 10.0, 0.1)
        y = (-b-ws[0, 0] * x / ws[1, 0])
        ax.plot(x, y)
        for i in range(100):
            if alphas[i] > 0.0:
                ax.plot(x_mat[i, 0], x_mat[i, 1], 'cx')
            else:
                ax.plot(x_mat[i, 0], x_mat[i, 1], 'kp')

        for i in range(100):
            if alphas[i] > 0.0:
                ax.plot(x_mat[i, 0], x_mat[i, 1], 'ro')
        plt.show()

if __name__ == "__main__":
    # 获取特征和目标变量
    svm_simple = SVMSimple()
    dataArr, labelArr = svm_simple.load_data_set('../../../input/6.SVM/testSet.txt')
    # print(labelArr)

    # b是常量值， alphas是拉格朗日乘子
    b, alphas = svm_simple.smo_simple(dataArr, labelArr, 0.6, 0.001, 40)
    print('/n/n/n')
    print('b=', b)
    print('alphas[alphas>0]=', alphas[alphas > 0])
    print('shape(alphas[alphas > 0])=', shape(alphas[alphas > 0]))
    for i in range(100):
        if alphas[i] > 0:
            print(dataArr[i], labelArr[i])
    # 画图
    ws = svm_simple.calc_ws(alphas, dataArr, labelArr)
    svm_simple.plotfig_svm(dataArr, labelArr, ws, b, alphas)




