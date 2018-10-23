# -*- coding:utf-8 -*-
__author__ = 'yangxin'

"""
处理步骤：
* 收集数据，可以使用任何方法
* 准备数据，树构造算法只适用于标称型数据，因此数据必须离散化
* 分析数据：可以使用任何方法，构造树完成之后，我们应该检查徒刑是否符合预期、按照指定的特征划分数据集
* 训练算法：构造树的数据结构
* 测试算法：使用决策树进行分类
* 实用算法：此步骤可以适用与任何监督学习任务，而使用决策树可以更好地理解数据的内在含义
"""


class CreateDataSet(object):
    """
    创建数据集
    """
    def create_data_set(self):
        data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return data_set, labels





























