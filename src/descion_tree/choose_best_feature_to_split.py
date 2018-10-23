# -*- coding:utf-8 -*-
__author__ = 'yangxin'
from src.descion_tree.shannon_entropy import ShannonEntropy
from src.descion_tree.split_dataset import SplitDataset
"""
choose_best_feature_to_split选择最好的特征
    Args:
        data_set     数据集
    Returns:
        best_feature  最优的特征列
"""


class ChooseBestFeatureToSplit(object):
    # 根据计算香农熵，选择最优的特征
    def choose_best_feature_to_split(self, data_set):
        # 计算第一行有多少列的feature, 最后一列是标签列
        num_features = len(data_set[0]) - 1
        # 计算数据集的原始信息熵
        base_entropy = ShannonEntropy.calc_shannon_ent(data_set)
        # 最优的信息增益值，和最优的特征编号
        best_info_gain, best_feature = 0.0, -1
        for i in range(num_features):
            # 获取对应的feature下的所有数据
            feat_list = [example[i] for example in data_set]
            # 对特征list去重
            unique_vals = set(feat_list)
            # 定义一个新的信息熵
            new_entropy = 0.0
            # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据，计算数据集的信息熵
            for value in unique_vals:
                sub_data_set = SplitDataset.split_data_set(data_set, i, value)
                # 信息熵计算公式
                prob = len(sub_data_set) / float(len(data_set))
                new_entropy += prob * ShannonEntropy.calc_shannon_ent(sub_data_set)
            # 划分前后数据集的信息增益，获取信息熵最大的值
            info_gain = base_entropy - new_entropy
            print('info_gain=', info_gain, 'best_feature=', i, base_entropy, new_entropy)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
            return best_feature
