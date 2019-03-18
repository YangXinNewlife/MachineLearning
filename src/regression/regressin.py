# -*- codingutf-8 -*-
__author__ = 'yangxin_ryan'
from numpy import *
import matplotlib.pylab as plt


class Regression(object):

    def load_data_set(self, file_name):
        num_feat = len(open(file_name).readline().split('\t')) - 1
        data_mat = []
        label_mat = []
        fr = open(file_name)
        for line in fr.readlines():
            line_arr = []
            cur_line = line.strip().split("\t")
            for i in range(num_feat):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            label_mat.append(float(cur_line[-1]))
        return data_mat, label_mat

    def stand_regres(self, x_arr, y_arr):
        x_mat = mat(x_arr)
        y_mat = mat(y_arr).T
        xTx = x_mat.T * x_mat
        if linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = xTx.I * (x_mat.T * y_mat)
        return ws

    def lwlr(self, test_point, x_arr, y_arr, k=1.0):
        x_mat = mat(x_arr)
        y_mat = mat(y_arr).T
        m = shape(x_mat)[0]
        weights = mat(eye((m)))
        for j in range(m):
            diff_mat = test_point - x_mat[j:]
            weights[j, j] = exp(diff_mat * diff_mat / (-2.0 * k ** 2))
        xTx = x_mat * (weights * x_mat)
        if linalg.det(xTx) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = xTx.I * (x_mat.T * (weights * y_mat))
        return test_point * ws

    def lwlr_test(self, test_arr, x_arr, y_arr, k=1.0):
        m = shape(test_arr)[0]
        y_mat = zeros(m)
        for i in range(m):
            y_mat[i] = self.lwlr(test_arr[i], x_arr, y_arr, k)
        return y_mat

    def lwlr_test_plot(self, x_arr, y_arr, k=1.0):
        y_hat = zeros(shape(y_arr))
        x_copy = mat(x_arr)
        x_copy.sort(0)
        for i in range(shape(x_arr)[0]):
            y_hat[i] = self.lwlr(x_copy[i], x_arr, y_arr, k)
        return y_hat, x_copy

    def rss_error(self, y_arr, y_hat_arr):
        return ((y_arr - y_hat_arr) ** 2).sum()

    def ridge_regres(self, x_mat, y_mat, lam=0.2):
        xTx = x_mat.T * x_mat
        denom = xTx + eye(shape(x_mat)[1]) * lam
        if linalg.det(denom) == 0.0:
            print("This matrix is singular, cannot do inverse")
            return
        ws = denom.I * (x_mat.T * y_mat)
        return ws

    def ridge_test(self, x_arr, y_arr):
        x_mat = mat(x_arr)
        y_mat = mat(y_arr).T
        y_mean = mean(y_mat, 0)
        y_mat = y_mat - y_mean
        x_means = mean(x_mat, 0)
        x_var = var(x_mat, 0)
        x_mat = (x_mat - x_means) / x_var
        num_test_pts = 30
        w_mat = zeros((num_test_pts, shape(x_mat)[1]))
        for i in range(num_test_pts):
            # exp() 返回 e^x
            ws = self.ridge_regres(x_mat, y_mat, exp(i - 10))
            w_mat[i, :] = ws.T
        return w_mat

    def regularize(self, x_mat):  # 按列进行规范化
        in_mat = x_mat.copy()
        in_means = mean(in_mat, 0)  # 计算平均值然后减去它
        in_var = var(in_mat, 0)  # 计算除以Xi的方差
        in_mat = (in_mat - in_means) / in_var
        return in_mat

    def stageWise(self, x_arr, y_arr, eps=0.01, numIt=100):
        x_mat = mat(x_arr)
        y_mat = mat(y_arr).T
        y_mean = mean(y_mat, 0)
        y_mat = y_mat - y_mean  # 也可以规则化ys但会得到更小的coef
        x_mat = self.regularize(x_mat)
        m, n = shape(x_mat)
        return_mat = zeros((numIt, n))  # 测试代码删除
        ws = zeros((n, 1))
        ws_test = ws.copy()
        ws_max = ws.copy()
        for i in range(numIt):
            print(ws.T)
            lowest_error = inf
            for j in range(n):
                for sign in [-1, 1]:
                    ws_test = ws.copy()
                    ws_test[j] += eps * sign
                    y_test = x_mat * ws_test
                    rss_e = self.rss_error(y_mat.A, y_test.A)
                    if rss_e < lowest_error:
                        lowest_error = rss_e
                        ws_max = ws_test
            ws = ws_max.copy()
            return_mat[i, :] = ws.T
        return return_mat

    def regression1(self):
        x_arr, y_arr = self.load_data_set("data/8.Regression/data.txt")
        x_mat = mat(x_arr)
        y_mat = mat(y_arr)
        ws = self.stand_regres(x_arr, y_arr)
        fig = plt.figure()
        ax = fig.add_subplot(111)  # add_subplot(349)函数的参数的意思是，将画布分成3行4列图像画在从左到右从上到下第9块
        ax.scatter([x_mat[:, 1].flatten()], [y_mat.T[:, 0].flatten().A[0]])  # scatter 的x是xMat中的第二列，y是yMat的第一列
        x_copy = x_mat.copy()
        x_copy.sort(0)
        y_hat = x_copy * ws
        ax.plot(x_copy[:, 1], y_hat)
        plt.show()


if __name__ == '__main__':
    regression = Regression()
    regression.regression1()