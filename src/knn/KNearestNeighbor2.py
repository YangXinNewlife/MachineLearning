# -*- coding:utf-8 -*-
from numpy import *
import operator
__author__ = 'yangxin'


"""
分析需求

我在约会网站看到的人，分为三类：

不喜欢的
一般喜欢的
非常喜欢的
我希望分类算法可以实现替我区分这三类人。
而现在我有一群人的下面的数据：

每年的飞行里程
玩视频游戏消耗时间的百分比
每周吃的冰淇淋（公升）
如何根据这些数据对这群人进行分类？这就是我的需求。


代码原理

根据首先将数据分为训练数据以及测试数据。然后根据测试数据的每一条数据去计算训练数据中每个数据的距离，然后根据K值找距离最短的 K 个。

就是每次K个最邻近的结果。
"""
class KNearestNeighbor2(object):

    def __init__(self):
        print "[INFO]:Welcome Dating site！"

    @staticmethod
    def file_to_matrix(file_name):
        file_data = open(file_name)
        array_lines = file_data.readlines()
        number_lines = len(array_lines)
        mat = zeros((number_lines, 3))
        label_vector = []
        index = 0
        for line in array_lines:
            line = line.strip()
            list_from_line = line.split('\t')
            mat[index, :] = list_from_line[0:3]
            label_vector.append(int(list_from_line[-1]))
            index += 1
        return mat, label_vector

    @staticmethod
    def normalized(data_set):
        # 数据归一化
        min_vals = data_set.min(0)
        max_vals = data_set.max(0)
        ranges = max_vals - min_vals
        m = data_set.shape[0]
        norm_data_set = data_set - tile(min_vals, (m, 1))
        norm_data_set = norm_data_set / tile(ranges, (m, 1))
        return norm_data_set, ranges, min_vals

    @staticmethod
    def classify(in_x, data_set, labels, k):
        #  分类
        data_set_size = data_set.shape[0]
        diff_mat = tile(in_x, (data_set_size, 1)) - data_set
        sq_diff_mat = diff_mat ** 2
        sq_distances = sq_diff_mat.sum(axis=1)
        distances = sq_distances ** 0.5
        sorted_dist_indicies = distances.argsort()
        class_count = {}
        for i in range(k):
            vote_ilabel = labels[sorted_dist_indicies[i]]
            class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
        sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    def run(self, k):
        ho_ratio = 0.10
        dating_data_mat, dating_labels = self.file_to_matrix('../data/test_data_2.txt')
        norm_mat, ranges, min_vals = self.normalized(dating_data_mat)
        total_lines = norm_mat.shape[0]
        test_lines = int(total_lines * ho_ratio)
        error_count = 0.0
        for i in range(test_lines):
            classifier_result = self.classify(norm_mat[i, :],
                                              norm_mat[test_lines:total_lines, :],
                                              dating_labels[test_lines:total_lines],
                                              k)

            if classifier_result != dating_labels[i]:
                print "[ERROR]:test_data: %s"%norm_mat[test_lines:total_lines, :]
                error_count += 1.0
            print "[INFO]:classifier result: %d, real answer is: %d" % (classifier_result, dating_labels[i])
        print "[INFO]:error number: " + str(error_count)
        print "[INFO]:the total error rate is: %f" % (error_count / float(test_lines))


if __name__ == '__main__':
    k = 7
    k_nearest_neighbor_obj = KNearestNeighbor2()
    k_nearest_neighbor_obj.run(k)






