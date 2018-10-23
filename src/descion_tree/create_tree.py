# -*- coding:utf-8 -*-
__author__ = 'yangxin'
from src.descion_tree.choose_best_feature_to_split import ChooseBestFeatureToSplit
from src.descion_tree.split_dataset import SplitDataset
from src.descion_tree.majority_cnt import MajortyCnt
"""
递归创建树
"""


class CreateTree(object):

    def create_tree(self, data_set, labels):
        class_list = [example[-1] for example in data_set]
        # 如果数据集最后一列的第一个值出现的次数 = 整个集合的数量也就是一个类别，就只直接返回结果就行
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if len(data_set[0]) == 1:
            return MajortyCnt.majority_cnt(class_list)
        # 选取最优的列，得到最优列对应的label含义
        best_feat = ChooseBestFeatureToSplit.choose_best_feature_to_split(data_set)
        # 获取label的名称
        best_feat_label = labels[best_feat]
        my_tree = {best_feat: {}}
        # labels列表是可变对象，在Python函数中作为参数时传址引用，能够被全局修改
        del(labels[best_feat])
        feat_values = [example[best_feat] for example in data_set]
        unique_valus = set(feat_values)
        for value in unique_valus:
            # 求出剩余的标签label
            sub_labels = labels[:]
            # 递归遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数creat_tree()
            my_tree[best_feat_label][value] = self.create_tree(SplitDataset.split_data_set(data_set, best_feat, value),
                                                               sub_labels)
        return my_tree


