# -*- coding:utf-8 -*-
from __future__ import print_function
import operator
from math import log
import src.descion_tree.decision_tree_plot as dtPlot
__author__ = 'yangxin'


class DescionTreeApp1(object):
    # 创建数据集
    def create_data_set(self):
        data_set = [[1, 1, 'yes'],
                    [1, 1, 'yes'],
                    [1, 0, 'no'],
                    [0, 1, 'no'],
                    [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return data_set, labels

    # 数据的计算信息熵
    def calc_shannon_ent(self, data_set):
        num_entries = len(data_set)
        label_counts = {}
        for feat_vec in data_set:
            # 获取当前数据的标签
            current_label = feat_vec[-1]
            if current_label not in label_counts.keys():
                label_counts[current_label] = 0
            label_counts[current_label] += 1
        shannon_ent = 0.0
        for key in label_counts:
            # 计算Pi(每个数据的均值)
            prob = float(label_counts[key]) / num_entries
            # 计算Pi * logPi （均值 * (Pi开方）
            shannon_ent -= prob * log(prob, 2)
        return shannon_ent

    # 划分数据集，根据特征列划分
    def split_data_set(self, data_set, index, value):
        ret_data_set = []
        for feat_vec in data_set:
            if feat_vec[index] == value:
                reduced_feat_vec = feat_vec[:index]
                reduced_feat_vec.extend(feat_vec[index + 1:])
                ret_data_set.append(reduced_feat_vec)
        return ret_data_set

    # 划分数据集 此处的函数式写法是split_data_set同样的方式
    def split_data_set1(self, data_set, index, value):
        ret_data_set = [data[:index] + data[index + 1:] for data in data_set for i, v in enumerate(data) if i == index and v == value]
        return ret_data_set

    # 选择切分数据集的最佳特征
    def choose_best_feature_to_split(self, data_set):
        num_features = len(data_set[0]) - 1
        # 计算信息熵
        base_entropy = self.calc_shannon_ent(data_set)
        #best_info_gain, best_feature = 0.0, -1
        for i in range(num_features):
            # 收集特征的值的集合
            feat_list = [example[i] for example in data_set]
            # 去重
            unique_vals = set(feat_list)
            new_entropy = 0.0
            for value in unique_vals:
                sub_data_set = self.split_data_set(data_set, i, value)
                # 计算均值Pi
                prob = len(sub_data_set) / float(len(data_set))
                # 计算信息增益
                new_entropy += prob * self.calc_shannon_ent(sub_data_set)
            info_gain = base_entropy - new_entropy
            print('infoGain=', info_gain, 'bestFeature=', i, base_entropy, new_entropy)
            # 比较信息增益
            if (info_gain > best_info_gain):
                best_info_gain = info_gain
                bestFeature = i
        return bestFeature

    # 选择出现次数最多的一个类别
    def majority_cnt(self, class_list):
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    def createTree(self, data_set, labels):
        class_list = [example[-1] for example in data_set]
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if len(data_set[0]) == 1:
            return self.majority_cnt(class_list)
        best_feat = self.choose_best_feature_to_split(data_set)
        best_feat_label = labels[best_feat]
        myTree = {best_feat_label: {}}
        del (labels[best_feat])
        feat_values = [example[best_feat] for example in data_set]
        unique_vals = set(feat_values)
        for value in unique_vals:
            subLabels = labels[:]
            myTree[best_feat_label][value] = self.create_tree(self.split_data_set(data_set, best_feat, value), subLabels)
        return myTree

    def classify(self, input_tree, feat_labels, test_vec):
        first_str = input_tree.keys()[0]
        second_dict = input_tree[first_str]
        feat_index = feat_labels.index(first_str)
        key = test_vec[feat_index]
        value_of_feat = second_dict[key]
        print('+++', first_str, 'xxx', second_dict, '---', key, '>>>', value_of_feat)
        if isinstance(value_of_feat, dict):
            class_label = self.classify(value_of_feat, feat_labels, test_vec)
        else:
            class_label = value_of_feat
        return class_label

    def store_tree(self, input_tree, file_name):
        import pickle
        fw = open(file_name, 'wb')
        pickle.dump(input_tree, fw)
        fw.close()
        with open(file_name, 'wb') as fw:
            pickle.dump(input_tree, fw)

    def grab_tree(self, file_name):
        import pickle
        fr = open(file_name, 'rb')
        return pickle.load(fr)


    def fish_test(self):
        myDat, labels = self.create_data_set()
        import copy
        myTree = self.create_tree(myDat, copy.deepcopy(labels))
        print(myTree)
        print(self.classify(myTree, labels, [1, 1]))
        print(self.get_tree_height(myTree))
        dtPlot.createPlot(myTree)


    def contact_lenses_test(self):
        fr = open('db/3.DecisionTree/lenses.txt')
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lensesTree = self.create_tree(lenses, lensesLabels)
        print(lensesTree)
        dtPlot.createPlot(lensesTree)


    def get_tree_height(self, tree):
        if not isinstance(tree, dict):
            return 1
        child_trees = tree.values()[0].values()
        max_height = 0
        for child_tree in child_trees:
            child_tree_height = self.get_tree_height(child_tree)
            if child_tree_height > max_height:
                max_height = child_tree_height
        return max_height + 1


if __name__ == "__main__":
    descion_tree_app1 = DescionTreeApp1()
    descion_tree_app1.fishTest()


