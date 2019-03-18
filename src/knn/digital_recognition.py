# -*- coding:utf-8 -*-
__author__ = 'yangxin_ryan'
from numpy import *
from os import listdir
from collections import Counter
import operator
"""
图片的输入为 32 * 32的转换为 1 * 1024的向量
"""


class DigitalRecognition(object):

    def __init__(self):
        print("Welcome, 手写数字识别算法!")

    """
    1.距离计算
    tile生成和训练样本对应的矩阵，并与训练样本求差
    取平方
    将矩阵的每一行相加
    开方
    根据距离从小到大的排序，并返回对应的索引位置
    2.选择距离最小的k个值
    3.排序并返回出现最多的那个类型
    """
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

    """         
    1.计算距离
    2.k个最近的标签
    3.出现次数最多的标签即为最终类别
    """
    def classify_2(self, in_x, data_set, labels, k):
        dist = np.sum((in_x - data_set) ** 2, axis=1) ** 0.5
        k_labels = [labels[index] for index in dist.argsort()[0:k]]
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
            return_mat[index, :] = list_from_line[0:3]
            class_label_vector.append(int(list_from_line[-1]))
            index += 1
        return return_mat, class_label_vector

    """
    将图片转换为向量
    图片的输入为 32 * 32的，将图像转换为向量，该函数创建 1 * 1024 的Numpy数组
    """
    def img_to_vector(self, file_name):
        return_vector = zeros((1, 1024))
        fr = open(file_name, 'r')
        for i in range(32):
            line_str = fr.readline()
            for j in range(32):
                return_vector[0, 32 * i + j] = int(line_str[j])
        return return_vector

    def run(self, train_file_path, test_file_path, k):
        labels = []
        training_file_list = listdir(train_file_path)
        train_len = len(training_file_list)
        training_mat = zeros((train_len, 1024))
        for i in range(train_len):
            file_name_str = training_file_list[i]
            file_str = file_name_str.split(".")[0]
            class_num_str = int(file_str.split("_")[0])
            labels.append(class_num_str)
            img_file = train_file_path + file_name_str
            print(img_file)
            training_mat[i] = self.img_to_vector(img_file)

        test_file_list = listdir(test_file_path)
        error_count = 0.0
        test_len = len(test_file_list)
        for i in range(test_len):
            file_name_str = test_file_list[i]
            file_str = file_name_str.split(".")[0]
            class_num_str = int(file_str.split("_")[0])
            test_file_img = test_file_path + file_name_str
            vector_under_test = self.img_to_vector(test_file_img)
            classifier_result = self.classify_1(vector_under_test, training_mat, labels, k)
            if classifier_result != class_num_str:
                print(file_name_str)
                error_count += 1.0
        print("\nthe total number of errors is: %d" % error_count)
        print("\nthe total error rate is: %f" % (error_count / float(test_len)))


if __name__ == '__main__':
    digital_recognition = DigitalRecognition()
    digital_recognition.run("/Users/yangxin_ryan/PycharmProjects/MachineLearning/data/knn/trainingDigits/",
                            "/Users/yangxin_ryan/PycharmProjects/MachineLearning/data/knn/testDigits/",
                            6)