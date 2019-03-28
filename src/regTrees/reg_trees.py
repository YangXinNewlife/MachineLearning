# -*- coding:utf-8 -*-
__author__ = 'yangxin_ryan'
from numpy import *
"""
树回归
"""

class RegTees(object):

    def load_data_set(self, file_name):
        data_mat = []
        fr = open(file_name)
        for line in fr.readlines():
            cur_line = line.strip().split("\t")
            flt_line = [float(x) for x in cur_line]
            data_mat.append(flt_line)
        return data_mat

    def bin_split_data_set(self, data_set, feature, value):
        mat0 = data_set[nonzero(data_set[:, feature] <= value)[0], :]
        mat1 = data_set[nonzero(data_set[:, feature] > value)[0], :]
        return mat0, mat1

    def reg_left(self, data_set):
        return mean(data_set[:, -1])

    def reg_err(self, data_set):
        return var(data_set[:, -1]) * shape(data_set)[0]

    def choose_best_split(self, data_set, leaf_type, err_type, ops=(1, 4)):
        tol_s = ops[0]
        tol_n = ops[1]
        if len(set(data_set[:, -1].T.tolist()[0])) == 1:
            return None, leaf_type(data_set)
        m, n = shape(data_set)
        s = err_type(data_set)
        best_s, best_index, best_value = inf, 0, 0
        for feat_index in range(n - 1):
            for split_val in set(data_set[:, feat_index].T.tolist()[0]):
                mat0, mat1 = self.bin_split_data_set(data_set, feat_index, split_val)
                if (shape(mat0)[0] < tol_n) or (shape(mat1)[0] < tol_n):
                    continue
                new_s = err_type(mat0) + err_type(mat1)
                if new_s < best_s:
                    best_index = feat_index
                    best_value = split_val
                    best_s = new_s
        if (s - best_s) < tol_s:
            return None, leaf_type(data_set)
        mat0, mat1 = self.bin_split_data_set(data_set, best_index, best_value)
        if (shape(mat0)[0] < tol_n) or (shape(mat1)[0] < tol_n):
            return None, leaf_type(data_set)
        return best_index, best_value

    def create_tree(self, data_set, leaf_type, err_type, ops=(1, 4)):
        feat, val = self.choose_best_split(data_set, leaf_type, err_type, ops)
        if feat is None:
            return val
        ret_tree = {}
        ret_tree['spInd'] = feat
        ret_tree['spVal'] = val
        l_set, r_set = self.bin_split_data_set(data_set, feat, val)
        ret_tree['left'] = self.create_tree(l_set, leaf_type, err_type, ops)
        ret_tree['right'] = self.create_tree(r_set, leaf_type, err_type, ops)
        return ret_tree

    def is_tree(self, obj):
        return (type(obj).__name__ == 'dict')

    def get_mean(self, tree):
        if self.is_tree(tree['right']):
            tree['right'] = self.get_mean(tree['right'])
        if self.is_tree(tree['left']):
            tree['left'] = self.get_mean(tree['left'])
        return (tree['left'] + tree['right']) / 2.0

    def prune(self, tree, test_data):
        l_set, r_set = None, None
        if shape(test_data)[0] == 0:
            return self.get_mean(tree)
        if self.is_tree(tree['right']) or self.is_tree(tree['left']):
            l_set, r_set = self.bin_split_data_set(test_data, tree['spInd'], tree['spVal'])
        if self.is_tree(tree['left']):
            tree['left'] = self.prune(tree['left'], l_set)
        if self.is_tree(tree['right']):
            tree['right'] = self.prune(tree['right'], r_set)
        if not self.is_tree(tree['left']) and not self.is_tree(tree['right']):
            l_set, r_set = self.bin_split_data_set(test_data, tree['spInd'], tree['spVal'])
            error_no_merge = sum(power(l_set[:, -1] - tree['left'], 2)) + sum(power(r_set[:, -1] - tree['right'], 2))
            tree_mean = (tree['left'] + tree['right']) / 2.0
            error_merge = sum(power(test_data[:, -1] - tree_mean, 2))
            if error_merge < error_no_merge:
                return tree_mean
            else:
                return tree
        else:
            return tree

    def model_leaf(self, data_set):
        ws, X, Y = self.linear_solve(data_set)
        return ws

    def model_err(self, data_set):
        ws, X, Y = self.linear_solve(data_set)
        y_hat = X * ws
        return sum(power(Y - y_hat, 2))

    def linear_solve(self, data_set):
        m, n = shape(data_set)
        X = mat(ones((m, n)))
        Y = mat(ones((m, 1)))
        X[:, 1: n] = data_set[:, 0: n - 1]
        Y = data_set[:, -1]
        xTx = X.T * X
        if linalg.det(xTx) == 0.0:
            raise NameError('This matrix is singular, cannot do inverse,\ntry increasing the second value of ops')
        ws = xTx.I * (X.T * Y)
        return ws, X, Y

    def reg_tree_eval(self, model, in_dat):
        return float(model)

    def model_tree_eval(self, model, in_dat):
        n = shape(in_dat)[1]
        X = mat(ones((1, n + 1)))
        X[:, 1: n + 1] = in_dat
        return float(X * model)

    def tree_fore_cast(self, tree, in_data, model_eval):
        if not self.is_tree(tree):
            return model_eval(tree, in_data)
        if in_data[tree['spInd']] <= tree['spVal']:
            if self.is_tree(tree['left']):
                return self.tree_fore_cast(tree['left'], in_data, model_eval)
            else:
                return model_eval(tree['left'], in_data)
        else:
            if self.is_tree(tree['right']):
                return self.tree_fore_cast(tree['right'], in_data, model_eval)
            else:
                return model_eval(tree['right'], in_data)

    def create_fore_cast(self, tree, test_data, model_eval):
        m = len(test_data)
        y_hat = mat(zeros((m, 1)))
        for i in range(m):
            y_hat[i, 0] = self.tree_fore_cast(tree, mat(test_data[i]), model_eval)
        return y_hat


if __name__ == "__main__":
    reg_tress = RegTees()
    train_mat = mat(reg_tress.load_data_set('data/9.RegTrees/bikeSpeedVsIq_train.txt'))
    test_mat = mat(reg_tress.load_data_set('data/9.RegTrees/bikeSpeedVsIq_test.txt'))
    my_tree1 = reg_tress.create_tree(train_mat, ops=(1, 20))
    y_hat1 = reg_tress.create_fore_cast(my_tree1, test_mat[:, 0])
    my_tree2 = reg_tress.create_tree(train_mat, reg_tress.model_leaf, reg_tress.model_err, ops=(1, 20))
    y_hat2 = reg_tress.create_fore_cast(my_tree2, test_mat[:, 0], reg_tress.model_tree_eval)
    ws, X, Y = reg_tress.linear_solve(train_mat)
    m = len(test_mat[:, 0])
    y_hat3 = mat(zeros((m, 1)))
    for i in range(shape(test_mat)[0]):
        y_hat3[i] = test_mat[i, 0] * ws[1, 0] + ws[0, 0]
