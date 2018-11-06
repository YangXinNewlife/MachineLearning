# -*- coding:utf-8 -*-
import math
import csv
import random
import operator
__author__ = 'yangxin'
"""
问题：

这里我构造了一个150*5的矩阵，分别代表三类数据。每行的前四个值代表数据的特征，第五个值代表数据的类别

这三类数据分别属于apple、banana、orange

原理：

根据首先将数据分为训练数据以及测试数据。然后根据测试数据的每一条数据去计算训练数据中每个数据的距离，然后根据K值找距离最短的 K 个。

就是每次K个最邻近的结果。
"""


class KNearestNeighbor(object):

    def __init__(self):
        print "Welcome, 水果分类算法!"

    @staticmethod
    def load_data_set(filename, split, training_set, test_set):
        # 对 test_set, train_set 赋值
        with open(filename, 'r') as data_file:
            lines = csv.reader(data_file)
            data_set = list(lines)
            for x in range(len(data_set) - 1):
                for y in range(4):
                    data_set[x][y] = float(data_set[x][y])
                if random.random() < split:
                    training_set.append(data_set[x])
                else:
                    test_set.append(data_set[x])

    @staticmethod
    def calculate_distance(test_data, train_data, length):
        # 计算距离每个测试样本对所有训练样本的距离, 欧氏公式
        distance = 0
        for x in range(length):
            distance += pow((test_data[x] - train_data[x]), 2)
        return math.sqrt(distance)

    def get_neighbors(self, training_set, test_instance, k):
        # 返回最近的k个值
        distances = []
        length = len(test_instance) - 1
        for x in xrange(len(training_set)):
            dist = self.calculate_distance(test_instance, training_set[x], length)
            print "test: %s   -->   train: %s   =    dist: %s" % (test_instance, training_set[x], dist)
            distances.append((training_set[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
            return neighbors

    @staticmethod
    def get_label(neighbors):
        # 根据少数服从多数，决定归类到哪一类
        class_votes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1
        sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_votes[0][0]

    @staticmethod
    def get_accuracy(test_set, predictions):
        # 准确率计算
        correct = 0
        for x in range(len(test_set)):
            if test_set[x][-1] == predictions[x]:
                correct += 1
        print "correct: %s, total: %s" % (correct, len(test_set))
        return (correct / float(len(test_set))) * 100.0

    def run(self, k):
        training_set = []
        test_set = []
        split = 0.6   # train / test percentage
        self.load_data_set(r'../data/test_data_1.txt', split, training_set, test_set)
        print('Total train set: ' + str(len(training_set)))
        print('Total test set: ' + str(len(test_set)))
        predictions = []
        for x in range(len(test_set)):
            neighbors = self.get_neighbors(training_set, test_set[x], k)
            result = self.get_label(neighbors)
            predictions.append(result)
        accuracy = self.get_accuracy(test_set, predictions)
        print('Accuracy: ' + repr(accuracy) + '%')


if __name__ == '__main__':
    k = 3
    k_nearest_neighbor_obj = KNearestNeighbor()
    k_nearest_neighbor_obj.run(k)