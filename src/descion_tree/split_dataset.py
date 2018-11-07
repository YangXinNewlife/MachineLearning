# -*- coding:utf-8 -*-
__author__ = 'yangxin'

"""
    将指定特征的特征值等于value的行剩下列作为子数据集
    split_data_set（通过遍历data_set数据集，求出index对应的column列的值为value的行）
    就是一句index列进行分类，如果index列的数据等于value的时候，就要将index划分到我们创建的新的数据集中
    Args:
        data  数据集                   待划分的数据集
        index 表示每一行的index列       划分数据集的特征
        value 表示index列对应的value值  需要返回的特征的值
    Returns:
        index列为value的数据集【该数据集需要排除index列】

    处理思路：
    e.g:
    [[1, 1, ‘yes’], [1, 1, ‘yes’], [1, 0, ‘no’], [0, 1, ‘no’], [0, 1, ‘no’]]
    这个是我们的数据集。 如果我们选取第一个特征值也就是需不需要浮到水面上才能生存来划分我们的数据，这里生物有两种可能，1就是需要，0就是不需要。那么第一个特征的取值就是两种。
    如果我们按照第一个特征的第一个可能的取值来划分数据也就是当所有的样本的第一列取1的时候满足的样本，那就是如下三个：
    [1, 1, ‘yes’], [1, 1, ‘yes’], [1, 0, ‘no’]
    可以理解为这个特征为一条分界线，我们选取完这个特征之后这个特征就要从我们数据集中剔除，因为要把他理解

    测试用例：
    In [1]: import trees
    In [2]: reload(trees)
    Out[2]: <module 'trees' from 'trees.pyc'>
    In [3]: myDat,labels=trees.createDataSet()
    In [4]: myDat
    Out[4]: [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    In [5]: trees.splitDataSet(myDat,0,1)
    Out[5]: [[1, 'yes'], [1, 'yes'], [0, 'no']]
    In [6]: trees.splitDataSet(myDat,0,0)
    Out[6]: [[1, 'no'], [1, 'no']]

"""


class SplitDataset(object):
    # 划分数据集
    def split_data_set(self, data_set, index, value):
        ret_data_set = []
        for feat_vec in data_set:
            # index列为value的数据集【该数据集需要排除index列】
            # 判断index列的值是否为value
            if feat_vec[index] == value:
                # [:index]表示前index行，即若index 为 2，就是取 feat_vec的前index行
                reduced_feat_vec = feat_vec[:index]
                reduced_feat_vec.extend(feat_vec[index + 1:])
                # [index + 1:] 表示从跳过 index 的 index + 1 行，取接下来的数据
                # 收集结果值 index列为value的行【该行需要排除index列】
                ret_data_set.append(reduced_feat_vec)
        return ret_data_set






