# -*- coding:utf-8 -*-
__author__ = 'yangxin_ryan'
from numpy import *
from collections import Counter
import operator
"""
分析需求

我在交友网站看到的人，分为三类：

不喜欢的
一般喜欢的
非常喜欢的
我希望分类算法可以实现替我区分这三类人。
而现在我有一群人的下面的数据：

* 每年的飞行里程
* 玩视频游戏消耗时间的百分比
* 每周吃的冰淇淋

如何根据这些数据对这群人进行分类？这就是我的需求。


代码原理

根据首先将数据分为训练数据以及测试数据。然后根据测试数据的每一条数据去计算训练数据中每个数据的距离，然后根据K值找距离最短的 K 个。

就是每次K个最邻近的结果。
"""


class InterestsSimilar(object):

    def __init__(self):
        print("Welcome, 兴趣相似的盆友算法!")

    def classify_1(self, in_x, data_set, labels, k):
        data_set_size = data_set.shape[0]
        diff_mat = tile(in_x, (data_set_size, 1)) - data_set
        sq_diff_mat = diff_mat ** 2
        sq_distances = sq_diff_mat.sum(axis=1)
        distances = sq_distances ** 0.5
        sorted_dist_indicies = distances.argsort()
        class_count = {}
        for i in range(k):
            vote_i_label = labels[sorted_dist_indicies[i]]
            class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count

    def classify_2(self, in_x, data_set, labels, k):
        dist = np.sum((in_x - data_set) ** 2, axis=1) ** 0.5
        k_labels = [labels[index] for index in dist.argsort()[0 : k]]
        label = Counter(k_labels).most_common(1)[0][0]
        return label

    def file_to_matrix(self, file_name):
        fr = open(file_name)
        number_of_lines = len(fr.readlines())
        return_mat = zeros((number_of_lines, 3))
        class_label_vector = []
        fr = open(file_name)
        index = 0
        for line in fr.readlines():
            line = line.strip()
            list_from_line = line.split("\t")
            return_mat[index, :] = list_from_line[0 : 3]
            class_label_vector.append(int(list_from_line[-1]))
            index += 1
        return return_mat, class_label_vector

    """
    归一化特征，消除属性之间量级不同导致的影响
    归一化公式：Y = （X - Xmin） / （Xmax - Xmin）
    """
    def auto_norm(self, data_set):
        min_vals, max_vals = data_set.min(0), data_set.max(0)
        ranges = max_vals - min_vals
        norm_data_set = (data_set - min_vals) / ranges
        return norm_data_set, ranges, min_vals

    def run(self, file_path, k):
        test_rate = 0.2

        dating_data_mat, dating_labels = self.file_to_matrix(file_path)

        norm_mat, ranges, min_vals = self.auto_norm(dating_data_mat)

        length = norm_mat.shape[0]

        num_test_vecs = int(length * test_rate)
        print('num_test_vecs=', num_test_vecs)

        error_count = 0
        count = 0

        for i in range(num_test_vecs):
            classifier_result = self.classify_1(norm_mat[i], norm_mat[num_test_vecs : length], dating_labels[num_test_vecs : length], k)

            if classifier_result != dating_labels[i]:
                error_count += 1
            else:
                count += 1
        print("the total error rate is: %f" % (error_count / num_test_vecs))
        print(count / num_test_vecs)
        print(error_count)


if __name__ == '__main__':
    interest_similar = InterestsSimilar()
    interest_similar.run("/Users/yangxin_ryan/PycharmProjects/MachineLearning/data/knn/interests_friends.txt", 3)
