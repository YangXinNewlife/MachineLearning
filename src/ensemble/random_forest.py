# -*- coding:utf-8 -*-
__author__ = 'yangxin_ryan'
'''
机器方法中的
随机森林算法

'''
from random import seed, randrange, random


class RandomForest(object):

    # load data
    def load_data_set(self, file_name):
        data_set = []
        with open(file_name, 'r') as fr:
            for line in fr.readlines():
                if not line:
                    continue
                line_arr = []
                for featrue in line.split(","):
                    # strip() 返回移除字符串头尾指定的字符生成的新字符串
                    str_f = featrue.strip()
                    if str_f.isdigit():  # 判断字符串是否是数字
                        line_arr.append(float(str_f))
                    else:
                        line_arr.append(str_f)
                data_set.append(line_arr)
        return data_set

    #
    def cross_validation_split(self, data_set, n_folds):
        data_set_split = list()
        data_set_copy = list(data_set)
        fold_size = len(data_set) / n_folds
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(data_set_copy))
                fold.append(data_set_copy[index])
            data_set_split.append(fold)
        return data_set_split

    # split a data_set based on an attribute and an attribute value
    def test_split(self, index,value, data_set):
        left, right = list(), list()
        for row in data_set:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # calculate the gini index for a split data_set
    def gini_index(self, groups, class_values):
        gini = 0.0
        for class_value in class_values:
            for group in groups:
                size = len(group)
                if size == 0:
                    continue
                proportion = [row[-1] for row in group].count(class_value) / float(size)
                gini += (proportion * (1.0 - proportion))
        return gini

    # 找出分割数据集的最优特征，得到最优特征，index, 特征值 row[index], 以及分割完的数据 groups (left, right)
    def get_split(self, data_set, n_features):
        class_values = list(set(row[-1] for row in data_set)) # class_values = [0, 1]
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        features = list()
        while len(features) < n_features:
            index = randrange(len(data_set[0]) - 1)
            if index in features:
                for row in data_set:
                    groups = self.test_split(index, row[index], data_set)
                    gini = self.gini_index(groups, class_values)
                    if gini < b_score:
                        b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    # create a terminal node value 输出group中出现次数较多的标签
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, n_features, depth):
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left, n_features)
            self.split(node['left'], max_depth, min_size, n_features, depth+1)

    # make a prediction with a decision tree
    def build_tree(self, train, max_depth, min_size, n_features):
        root = self.get_split(train, n_features)
        self.split(root, max_depth, min_size, n_features, 1)
        return root

    # make a prediction with a decision trees
    def predict(self, node, row):
        if row[node['index']] < node['values']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right', dict]):
                return self.predict(node['right'], row)
            else:
                return node['right']

    # make a prediction with a list of bagged trees
    def bagging_predict(self, trees, row):
        predictions = [self.predict(trees, row) for trees in trees]
        return max(set(predictions), key=predictions.count)

    # create a random subsample from the dataset with replacement
    def subsample(self, data_set, ratio):
        sample = list()
        n_sample = round(len(data_set) * ratio)
        while len(sample) < n_sample:
            index = randrange(len(data_set))
            sample.append(data_set[index])
        return sample

    # Random Forest Algorithm
    def random_forest(self, train, test, max_depth, min_size, sample_size, n_trees, n_features):
        trees = list()
        for i in range(n_trees):
            sample = self.subsample(train, sample_size)
            tree = self.build_tree(sample, max_depth, min_size, n_features)
            trees.append(tree)

        predictions = [self.bagging_predict(trees, row) for row in test]
        return predictions

    # calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # 评价算法性能，返回模型得分
    def evaluate_algorithm(self, data_set, algorithm, n_folds, *args):
        folds = self.cross_validation_split(data_set, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                row_copy[-1] = None
                test_set.append(row_copy)
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuray = self.accuracy_metric(actual, predicted)
            scores.append(accuray)
        return scores


if __name__ == '__main__':
    random_forest = RandomForest()
    data_set = random_forest.load_data_set("/path/file.txt")
    n_folds = 5
    max_depth = 20
    min_size = 1
    sample_size = 1.0
    n_features = 15
    for n_trees in [1, 10, 20]:
        scores = random_forest.evaluate_algorithm(data_set, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
        seed(1)
        print('random=', random())
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))