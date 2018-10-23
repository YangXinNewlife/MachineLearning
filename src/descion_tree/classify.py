# -*- coding:utf-8 -*-
__author__ = 'yangxin'
"""
分类
"""


class Classify(object):

    def classify(self, input_tree, feat_labels, test_vec):
        first_str = input_tree.keys()[0]
        second_dict = input_tree[first_str]
        fest_index = feat_labels.index(first_str)
        key = test_vec[fest_index]
        value_off_feat = second_dict[key]
        print('+++', first_str,'xxx', second_dict, '---', key, '>>>', value_off_feat)
        if isinstance(value_off_feat, dict):
            class_label = self.classify(value_off_feat, feat_labels, test_vec)
        else:
            class_label = value_off_feat
        return class_label

