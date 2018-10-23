# -*- coding:utf-8 -*-
import math
__author__ = 'yangxin'

"""
一条信息的信息量大小和它的不确定性有直接的关系。比如说，我们要搞清楚一件非常非常不确定的事，或是我们一无所知的事情，
就需要了解大量的信息。相反，如果我们对某件事已经有了较多的了解，我们不需要太多的信息就能把它搞清楚。
所以，从这个角度，我们可以认为，信息量的度量就等于不确定性的多少。
"""


class ShannonEntropy(object):

    # 计算给定数据集的香农墒的函数
    def calc_shannon_ent(self, data_set):
        # 求list的长度，表示计算参与训练的数据量
        num_entries = len(data_set)
        # 计算分类标签label出现的次数
        label_counts = {}
        # the number of unique elements and their occurance
        for featVec in data_set:
            # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
            current_label = featVec[-1]
            # 为所有可能的分类创建字典，如果当前的健值不存在，则扩展字典并将当前健值加入
            if current_label not in label_counts.keys():
                label_counts[current_label] = 0
                label_counts[current_label] += 1
        # 对于label标签的占比，求出label标签的香农墒
        shannon_ent = 0.0
        for key in label_counts:
            # 使所有类标签的发生频率计算类别出现的概率
            prob = float(label_counts[key]) / num_entries
            # 计算香农熵，以 2 为底求对数
            shannon_ent -= prob * math.log(prob, 2)
        return shannon_ent

