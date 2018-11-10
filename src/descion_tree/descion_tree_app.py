# -*- coding:utf-8 -*-
__author__ = 'yangxin_ryan'

import operator
from math import log
from src.descion_tree.decision_tree_plot import DecisionTreePlot as dtPlot
import pickle
import copy


class DescionTreeApp(object):

    def create_data_set(self):
        data_set = [[1, 1, 'yes'],
                   [1, 1, 'yes'],
                   [1, 0, 'no'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return data_set, labels

    def calc_shannon_ent(self, data_set):
        num_entries = len(data_set)
        label_counts = {}
        for feat_vec in data_set:
            current_label = feat_vec[-1]
            if current_label not in label_counts.keys():
                label_counts[current_label] = 0
            label_counts[current_label] += 1
        shannon_ent = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / num_entries
            shannon_ent -= prob * log(prob, 2)
        return shannon_ent

    def split_data_set(self, data_set, index, value):
        ret_data_set = []
        for feat_vec in data_set:
            if feat_vec[index] == value:
                reduced_feat_vec = feat_vec[:index]
                reduced_feat_vec.extend(feat_vec[index+1:])
                ret_data_set.append(reduced_feat_vec)
        return ret_data_set

    def choose_best_feature_to_split(self, data_set):
        num_features = len(data_set[0]) - 1
        base_entropy = self.calc_shannon_ent(data_set)
        best_info_gain, best_feature = 0.0, -1
        for i in range(num_features):
            feat_list = [example[i] for example in data_set]
            unique_vals = set(feat_list)
            new_entropy = 0.0
            for value in unique_vals:
                sub_data_set = self.split_data_set(data_set, i, value)
                prob = len(sub_data_set)/float(len(data_set))
                new_entropy += prob * self.calc_shannon_ent(sub_data_set)
            info_gain = base_entropy - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
        return best_feature

    def majority_cnt(self, class_list):
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    def create_tree(self, data_set, labels):
        class_list = [example[-1] for example in data_set]
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if len(data_set[0]) == 1:
            return self.majority_cnt(class_list)
        best_feat = self.choose_best_feature_to_split(data_set)
        best_feat_label = labels[best_feat]
        my_tree = {best_feat_label: {}}
        del(labels[best_feat])
        feat_values = [example[best_feat] for example in data_set]
        unique_vals = set(feat_values)
        for value in unique_vals:
            sub_labels = labels[:]
            my_tree[best_feat_label][value] = self.create_tree(self.split_data_set(data_set, best_feat, value), sub_labels)
        return my_tree

    def classify(self, input_tree, feat_labels, test_vec):
        first_str = list(input_tree.keys())[0]
        second_dict = input_tree[first_str]
        feat_index = feat_labels.index(first_str)
        key = test_vec[feat_index]
        value_of_feat = second_dict[key]
        if isinstance(value_of_feat, dict):
            class_label = self.classify(value_of_feat, feat_labels, test_vec)
        else:
            class_label = value_of_feat
        return class_label

    def store_tree(self, input_tree, filename):
        fw = open(filename, 'wb')
        pickle.dump(input_tree, fw)
        fw.close()
        with open(filename, 'wb') as fw:
            pickle.dump(input_tree, fw)

    def grab_tree(self, filename):
        fr = open(filename, 'rb')
        return pickle.load(fr)

    # 应用测试一、判断鱼类与非鱼类
    def app_fish(self):
        my_dat, labels = self.create_data_set()
        my_tree = self.create_tree(my_dat, copy.deepcopy(labels))
        dtPlot.create_plot(my_tree)

    # 应用测试二、判断隐形眼镜的类型
    def app_contact_lenses(self):
        fr = open('')
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lenses_tree = self.create_tree(lenses, lenses_labels)
        dtPlot.create_plot(lenses_tree)


if __name__ == "__main__":
    app = DescionTreeApp()
    app.app_contact_lenses()
