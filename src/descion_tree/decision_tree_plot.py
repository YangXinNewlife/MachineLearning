# -*- coding:utf-8 -*-
__author__ = 'yangxin_ryan'

import matplotlib.pyplot as plt
"""
定义文本框 和 箭头格式 
【 sawtooth 波浪方框, round4 矩形方框 , fc表示字体颜色的深浅 0.1~0.9 依次变浅，没错是变浅】
"""
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


class DecisionTreePlot(object):

    def get_num_leafs(self, my_tree):
        num_leafs = 0
        first_str = my_tree.keys()[0]
        second_dict = my_tree[first_str]
        for key in second_dict.keys():
            if type(second_dict[key]) is dict:
                num_leafs += self.get_num_leafs(second_dict[key])
            else:
                num_leafs += 1
        return num_leafs

    def get_tree_depth(self, my_tree):
        max_depth = 0
        first_str = my_tree.keys()[0]
        second_dict = my_tree[first_str]
        for key in second_dict.keys():
            if type(second_dict[key]) is dict:
                this_depth = 1 + self.get_tree_depth(second_dict[key])
            else:
                this_depth = 1
            if this_depth > max_depth:
                max_depth = this_depth
        return max_depth

    def plot_node(self, node_txt, center_pt, parent_pt, node_type):
        self.create_plot.ax1.annotate(node_txt, xy=parent_pt,  xycoords='axes fraction', xytext=center_pt,
                                textcoords='axes fraction', va="center", ha="center", bbox=node_type,
                                arrowprops=arrow_args)

    def plot_mid_text(self, cntr_pt, parent_pt, txt_string):
        x_mid = (parent_pt[0] - cntr_pt[0]) / 2.0 + cntr_pt[0]
        y_mid = (parent_pt[1] - cntr_pt[1]) / 2.0 + cntr_pt[1]
        self.create_plot.ax1.text(x_mid, y_mid, txt_string, va="center", ha="center", rotation=30)

    def plot_tree(self, my_tree, parent_pt, node_txt):
        num_leafs = self.get_num_leafs(my_tree)
        cntr_pt = (self.plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / self.plot_tree.totalW, self.plot_tree.yOff)
        self.plot_mid_text(cntr_pt, parent_pt, node_txt)
        first_str = my_tree.keys()[0]
        self.plot_node(first_str, cntr_pt, parent_pt, decisionNode)
        second_dict = my_tree[first_str]
        self.plot_tree.yOff = self.plot_tree.yOff - 1.0 / self.plot_tree.totalD
        for key in second_dict.keys():
            if type(second_dict[key]) is dict:
                self.plot_tree(second_dict[key], cntr_pt, str(key))
            else:
                self.plot_tree.xOff = self.plot_tree.xOff + 1.0 / self.plot_tree.totalW
                self.plot_node(second_dict[key], (self.plot_tree.xOff, self.plot_tree.yOff), self.cntr_pt, self.leaf_node)
                self.plot_mid_text((self.plot_tree.xOff, self.plot_tree.yOff), self.cntr_pt, str(key))
        self.plot_tree.yOff = self.plot_tree.yOff + 1.0 / self.plot_tree.totalD

    def create_plot(self, in_tree):
        fig = plt.figure(1, facecolor='green')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
        self.plot_tree.totalW = float(self.get_num_leafs(in_tree))
        self.plot_tree.totalD = float(self.get_tree_depth(in_tree))
        self.plot_tree.xOff = -0.5 / self.plot_tree.totalW
        self.plot_tree.yOff = 1.0
        self.plot_tree(in_tree, (0.5, 1.0), '')
        plt.show()

    def retrieve_tree(self, i):
        list_of_trees = [
            {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
            {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
        ]
        return list_of_trees[i]
