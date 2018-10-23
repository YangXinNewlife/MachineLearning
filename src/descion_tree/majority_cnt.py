# -*- coding:utf-8 -*-
__author__ = 'yangxin'
import operator
"""
函数使用分类名称的列表，然后创建键值为class_list中唯一值的数据字典，字典存储了class_list中每个类标签出现的频率，
最后利用operator操作键值排序字典，并返回出现次数最多的分类名称
"""


class MajortyCnt(object):

    def majority_cnt(self, class_list):
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
                class_count[vote] += 1
        # 逆序排列
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]
