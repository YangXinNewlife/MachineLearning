# -*- coding:utf-8 -*-
__author__='yangxin_ryan'
import numpy as np
import matplotlib.pyplot as plt


class AdaBoost(object):

    # 导入测试数据
    # data_arr feature 对应的数据集
    # label_arr feature 对应的分类标签
    def load_sim_data(self):
        data_mat = np.matrix([[1.0, 2.1],
                              [2.0, 1.1],
                              [1.3, 1.0],
                              [1.0, 1.0],
                              [2.0, 1.0]])
        class_labels = [1.0, 1.0, -1.0, 1.0]
        return data_mat, class_labels

    # 加载数据
    def load_data_set(self, file_name):
        num_feat = len(open(file_name).readline().split('\t'))
        data_arr = []
        label_arr = []
        fr = open(file_name)
        for line in fr.readlines():
            line_arr = []
            cur_line = line.strip().split("\t")
            for i in range(num_feat - 1):
                line_arr.append(float(cur_line[i]))
            data_arr.append(line_arr)
            label_arr.append(float(cur_line[-1]))
        return np.matrix(data_arr), label_arr

    # 将数据集，按照feature列的value进行 二分法切分比较来赋值分类
    # data_mat[:, dimen] 表示数据集中第dimen列的所有值
    # thresh_ineq == 'lt'表示修改左边的值，gt表示修改右边的值
    def stump_classify(self, data_mat, dimen, thresh_val, thres_ineq):
        ret_array = np.ones((np.shape(data_mat)[0], 1))
        if thres_ineq == 'lt':
            ret_array[data_mat[:, dimen] <= thresh_val] = -1.0
        else:
            ret_array[data_mat[:, dimen] > thresh_val] = - 1.0
        return ret_array

    # 得到决策树的模型
    def build_stump(self, data_arr, class_labels, D):
        data_mat = np.mat(data_arr)
        label_mat = np.mat(class_labels).T
        m, n = np.shape(data_mat)
        num_steps = 10.0
        best_stump = {}
        best_class_est = np.mat(np.zeros((m, 1)))
        # 无穷大
        min_err = np.inf
        for i in range(n):
            range_min = data_mat[:, i].min()
            range_max = data_mat[:, i].max()
            step_size = (range_max - range_min) / num_steps
            for j in range(-1, int(num_steps) + 1):
                for inequal in ['lt', 'gt']:
                    thresh_val = (range_min + float(j) * step_size)
                    predicted_vals = self.stump_classify(data_mat, i, thresh_val, inequal)
                    err_arr = np.mat(np.ones((m, 1)))
                    err_arr[predicted_vals == label_mat] = 0
                    # 这里是矩阵乘法
                    weighted_err = D.T * err_arr
                    if weighted_err < min_err:
                        min_err = weighted_err
                        best_class_est = predicted_vals.copy()
                        best_stump['dim'] = i
                        best_stump['thresh'] = thresh_val
                        best_stump['ineq'] = inequal
        return best_stump, min_err, best_class_est

    # adaBoost训练过程放大
    def ada_boost_train_ds(self, data_arr, class_labels, num_it=40):
        weak_class_arr = []
        m = np.shape(data_arr)[0]
        D = np.mat(np.ones((m, 1)) / m)
        agg_class_est = np.mat(np.zeros((m, 1)))
        for i in range(num_it):
            best_stump, error, class_est = self.build_stump(data_arr, class_labels, D)
            alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
            best_stump['alpha'] = alpha
            weak_class_arr.append(best_stump)
            expon = np.multiply(-1 * alpha * np.mat(class_labels).T, class_est)
            D = np.multiply(D, np.exp(expon))
            D = D / D.sum()
            agg_class_est += alpha * class_est
            agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T,
                                     np.ones((m, 1)))
            error_rate = agg_errors.sum() / m
            if error_rate == 0.0:
                break
        return weak_class_arr, agg_class_est

    # 通过刚刚上面那个函数得到的弱分类器的集合进行预测
    def ada_classify(self, data_to_class, classifier_arr):
        data_mat = np.mat(data_to_class)
        m = np.shape(data_mat)[0]
        agg_class_est = np.mat(np.zeros((m, 1)))
        for i in range(len(classifier_arr)):
            class_est = self.stump_classify(
                data_mat, classifier_arr[i]['dim'],
                classifier_arr[i]['thresh'],
                classifier_arr[i]['ineq']
            )
            agg_class_est += classifier_arr[i]['alpha'] * class_est
            print(agg_class_est)
        return np.sign(agg_class_est)

    # 通过去计算AUC
    def plot_roc(self, pred_strengths, class_labels):
        y_sum = 0.0
        num_pos_class = np.sum(np.array(class_labels) == 1.0)
        y_step = 1 / float(num_pos_class)
        x_step = 1 / float(len(class_labels) - num_pos_class)
        sorted_indicies = pred_strengths.argsort()
        fig = plt.figure()
        fig.clf()
        ax = plt.subplot(111)
        cur = (1.0, 1.0)
        for index in sorted_indicies.tolist()[0]:
            if class_labels[index] == 1.0:
                del_x = 0
                del_y = y_step
            else:
                del_x = x_step
                del_y = 0
                y_sum += cur[1]
            ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], c='b')
            cur = (cur[0] - del_x, cur[1] - del_y)
        ax.plot([0, 1], [0, 1], 'b--')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve for AdaBoost horse colic detection system')
        ax.axis([0, 1, 0, 1])
        plt.show()
        print("the Area Under the Curve is: ", y_sum * x_step)

    def test(self):
        data_mat, class_labels = self.load_data_set('xxxx1.txt')
        print(data_mat.shape, len(class_labels))
        weak_class_arr, agg_class_est = self.ada_boost_train_ds(data_mat, class_labels, 40)
        self.plot_roc(agg_class_est, class_labels)
        data_arr_test, label_arr_test = self.load_data_set("xxxx2.txt")
        m = np.shape(data_arr_test)[0]
        predicting10 = self.ada_classify(data_arr_test, weak_class_arr)
        err_arr = np.mat(np.ones((m, 1)))
        print(m, err_arr[predicting10 != np.mat(label_arr_test).T].sum(), err_arr[predicting10 != np.mat(label_arr_test).T].sum() / m)


if __name__ == '__main__':
    ada_boost = AdaBoost()
    ada_boost.test()