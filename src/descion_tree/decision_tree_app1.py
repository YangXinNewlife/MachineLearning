# -*- coding:utf-8 -*-
from __future__ import print_function
import operator
from math import log
import src.descion_tree.decision_tree_plot as dtPlot
__author__ = 'yangxin'
"""

"""


class DecisionTreeApp1(object):

    """
    基础数据，返回数据机和对应的label标签
    dataSet = [['yes'],['yes'],['no'],['no'],['no']]
    labels  露出水面   脚蹼
    """
    def create_data_set(self):
        data_set = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return data_set, labels

    """
    香农公式
    每一组feature下的某个分类下，香农熵的信息期望
    将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
    为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
    对于label标签的占比，求出label标签的香农熵
    使用所有类标签的发生频率计算类别出现的概率
    计算香农熵，以 2 为底求对数
    """
    def calc_shannon_ent(self, data_set):
        num_entries = len(data_set)
        label_counts = {}
        for featVec in data_set:
            current_label = featVec[-1]
            if current_label not in label_counts.keys():
                label_counts[current_label] = 0
            label_counts[current_label] += 1
        shannon_ent = 0.0
        for key in label_counts:
            prob = float(label_counts[key]) / num_entries
            shannon_ent -= prob * log(prob, 2)
        return shannon_ent

    """
    splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
        就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
    Args:
        dataSet 数据集                 待划分的数据集
        index 表示每一行的index列        划分数据集的特征
        value 表示index列对应的value值   需要返回的特征的值。
    Returns:
        index列为value的数据集【该数据集需要排除index列】
    """
    def split_data_set(self, data_set, index, value):
        ret_data_set = []
        for featVec in data_set:
            # index列为value的数据集【该数据集需要排除index列】
            # 判断index列的值是否为value
            if featVec[index] == value:
                # chop out index used for splitting
                # [:index]表示前index行，即若 index 为2，就是取 featVec 的前 index 行
                reducedFeatVec = featVec[:index]
                '''
                请百度查询一下： extend和append的区别
                list.append(object) 向列表中添加一个对象object
                list.extend(sequence) 把一个序列seq的内容添加到列表中
                1、使用append的时候，是将new_media看作一个对象，整体打包添加到music_media对象中。
                2、使用extend的时候，是将new_media看作一个序列，将这个序列和music_media序列合并，并放在其后面。
                result = []
                result.extend([1,2,3])
                print result
                result.append([4,5,6])
                print result
                result.extend([7,8,9])
                print result
                结果：
                [1, 2, 3]
                [1, 2, 3, [4, 5, 6]]
                [1, 2, 3, [4, 5, 6], 7, 8, 9]
                '''
                reducedFeatVec.extend(featVec[index + 1:])
                # [index+1:]表示从跳过 index 的 index+1行，取接下来的数据
                # 收集结果值 index列为value的行【该行需要排除index列】
                ret_data_set.append(reducedFeatVec)
        return ret_data_set


    def choose_best_feature_to_split(self, data_set):
        """chooseBestFeatureToSplit(选择最好的特征)

        Args:
            dataSet 数据集
        Returns:
            bestFeature 最优的特征列
        """

        # -----------选择最优特征的第一种方式 start------------------------------------
        # 求第一行有多少列的 Feature, 最后一列是label列嘛
        num_features = len(data_set[0])  - 1
        # label的信息熵
        base_entropy = self.calc_shannon_ent(data_set)
        # 最优的信息增益值, 和最优的Featurn编号
        best_info_gain, best_feature = 0.0, -1
        # iterate over all the features
        for i in range(num_features):
            # create a list of all the examples of this feature
            # 获取每一个实例的第i+1个feature，组成list集合
            feat_list = [example[i] for example in data_set]
            # get a set of unique values
            # 获取剔重后的集合，使用set对list数据进行去重
            unique_vals = set(feat_list)
            # 创建一个临时的信息熵
            new_entropy = 0.0
            # 遍历某一列的value集合，计算该列的信息熵
            # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
            for value in unique_vals:
                subDataSet = self.split_data_set(data_set, i, value)
                prob = len(subDataSet) / float(len(data_set))
                new_entropy += prob * self.calc_shannon_ent(subDataSet)
            # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
            # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
            infoGain = base_entropy - new_entropy
            print('infoGain=', infoGain, 'bestFeature=', i, base_entropy, new_entropy)
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature
        # -----------选择最优特征的第一种方式 end------------------------------------

        # # -----------选择最优特征的第二种方式 start------------------------------------
        # # 计算初始香农熵
        # base_entropy = calcShannonEnt(dataSet)
        # best_info_gain = 0
        # best_feature = -1
        # # 遍历每一个特征
        # for i in range(len(dataSet[0]) - 1):
        #     # 对当前特征进行统计
        #     feature_count = Counter([data[i] for data in dataSet])
        #     # 计算分割后的香农熵
        #     new_entropy = sum(feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) \
        #                    for feature in feature_count.items())
        #     # 更新值
        #     info_gain = base_entropy - new_entropy
        #     print('No. {0} feature info gain is {1:.3f}'.format(i, info_gain))
        #     if info_gain > best_info_gain:
        #         best_info_gain = info_gain
        #         best_feature = i
        # return best_feature
        # # -----------选择最优特征的第二种方式 end------------------------------------


    def majority_cnt(self, class_list):
        """majorityCnt(选择出现次数最多的一个结果)

        Args:
            classList label列的集合
        Returns:
            bestFeature 最优的特征列
        """
        # -----------majorityCnt的第一种方式 start------------------------------------
        class_count = {}
        for vote in class_list:
            if vote not in class_count.keys():
                class_count[vote] = 0
                class_count[vote] += 1
        # 倒叙排列classCount得到一个字典集合，然后取出第一个就是结果（yes/no），即出现次数最多的结果
        sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
        # print 'sortedClassCount:', sortedClassCount
        return sorted_class_count[0][0]
        # -----------majorityCnt的第一种方式 end------------------------------------

        # # -----------majorityCnt的第二种方式 start------------------------------------
        # major_label = Counter(classList).most_common(1)[0]
        # return major_label
        # # -----------majorityCnt的第二种方式 end------------------------------------


    def createTree(self, data_set, labels):
        class_list = [example[-1] for example in data_set]
        # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
        # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
        # count() 函数是统计括号中的值在list中出现的次数
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
        # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
        if len(data_set[0]) == 1:
            return self.majority_cnt(class_list)

        # 选择最优的列，得到最优列对应的label含义
        best_feat = self.choose_best_feature_to_split(data_set)
        # 获取label的名称
        best_feat_label = labels[best_feat]
        # 初始化myTree
        myTree = {best_feat_label: {}}
        # 注：labels列表是可变对象，在PYTHON函数中作为参数时传址引用，能够被全局修改
        # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
        del (labels[best_feat])
        # 取出最优列，然后它的branch做分类
        feat_values = [example[best_feat] for example in data_set]
        unique_vals = set(feat_values)
        for value in unique_vals:
            # 求出剩余的标签label
            subLabels = labels[:]
            # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
            myTree[best_feat_label][value] = self.create_tree(self.split_data_set(data_set, best_feat, value), subLabels)
            # print 'myTree', value, myTree
        return myTree


    def classify(self, input_tree, feat_labels, test_vec):
        """classify(给输入的节点，进行分类)

        Args:
            inputTree  决策树模型
            featLabels Feature标签对应的名称
            testVec    测试输入的数据
        Returns:
            classLabel 分类的结果值，需要映射label才能知道名称
        """
        # 获取tree的根节点对于的key值
        first_str = input_tree.keys()[0]
        # 通过key得到根节点对应的value
        second_dict = input_tree[first_str]
        # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
        feat_index = feat_labels.index(first_str)
        # 测试数据，找到根节点对应的label位置，也就知道从输入的数据的第几位来开始分类
        key = test_vec[feat_index]
        value_of_feat = second_dict[key]
        print('+++', first_str, 'xxx', second_dict, '---', key, '>>>', value_of_feat)
        # 判断分枝是否结束: 判断valueOfFeat是否是dict类型
        if isinstance(value_of_feat, dict):
            class_label = self.classify(value_of_feat, feat_labels, test_vec)
        else:
            class_label = value_of_feat
        return class_label

    def store_tree(self, input_tree, file_name):
        import pickle
        # -------------- 第一种方法 start --------------
        fw = open(file_name, 'wb')
        pickle.dump(input_tree, fw)
        fw.close()
        # -------------- 第一种方法 end --------------

        # -------------- 第二种方法 start --------------
        with open(file_name, 'wb') as fw:
            pickle.dump(input_tree, fw)
        # -------------- 第二种方法 start --------------


    def grab_tree(self, file_name):
        import pickle
        fr = open(file_name, 'rb')
        return pickle.load(fr)


    def fish_test(self):
        # 1.创建数据和结果标签
        myDat, labels = self.create_data_set()
        # print myDat, labels

        # 计算label分类标签的香农熵
        # calcShannonEnt(myDat)

        # # 求第0列 为 1/0的列的数据集【排除第0列】
        # print '1---', splitDataSet(myDat, 0, 1)
        # print '0---', splitDataSet(myDat, 0, 0)

        # # 计算最好的信息增益的列
        # print chooseBestFeatureToSplit(myDat)

        import copy
        myTree = self.create_tree(myDat, copy.deepcopy(labels))
        print(myTree)
        # [1, 1]表示要取的分支上的节点位置，对应的结果值
        print(self.classify(myTree, labels, [1, 1]))

        # 获得树的高度
        print(self.get_tree_height(myTree))

        # 画图可视化展现
        dtPlot.createPlot(myTree)


    def ContactLensesTest(self):
        """
        Desc:
            预测隐形眼镜的测试代码
        Args:
            none
        Returns:
            none
        """

        # 加载隐形眼镜相关的 文本文件 数据
        fr = open('db/3.DecisionTree/lenses.txt')
        # 解析数据，获得 features 数据
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        # 得到数据的对应的 Labels
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
        lensesTree = self.create_tree(lenses, lensesLabels)
        print(lensesTree)
        # 画图可视化展现
        dtPlot.createPlot(lensesTree)


    def get_tree_height(self, tree):
        """
         Desc:
            递归获得决策树的高度
        Args:
            tree
        Returns:
            树高
        """

        if not isinstance(tree, dict):
            return 1

        child_trees = tree.values()[0].values()

        # 遍历子树, 获得子树的最大高度
        max_height = 0
        for child_tree in child_trees:
            child_tree_height = self.get_tree_height(child_tree)

            if child_tree_height > max_height:
                max_height = child_tree_height

        return max_height + 1


if __name__ == "__main__":
    decison_tree = DecisionTreeApp1()
    decison_tree.fishTest()


