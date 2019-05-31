# -*- coding:utf-8 -*-
__author__ = 'yangxin_ryan'
from numpy import *
import matplotlib.pyplot as plt


class PCA(object):

    def load_data_set(self, file_name, delim='\t'):
        fr = open(file_name)
        string_arr = [line.strip().split(delim) for line in fr.readlines()]
        dat_arr = [list(map(float, line)) for line in string_arr]
        return mat(dat_arr)

    def pca(self, data_mat, top_n_feat=9999999):
        mean_vals = mean(data_mat, axis=0)
        mean_removed = data_mat - mean_vals
        cov_mat = cov(mean_removed, rowvar=0)
        eig_vals, eig_vects = linalg.eig(mat(cov_mat))
        eig_val_ind = argsort(eig_vals)
        eig_val_ind = eig_val_ind[:-(top_n_feat + 1): -1]
        red_eig_vects = eig_vects[:, eig_val_ind]
        low_d_data_mat = mean_removed * red_eig_vects
        recon_mat = (low_d_data_mat * red_eig_vects.T) + mean_vals
        return low_d_data_mat, recon_mat

    def replace_nan_with_mean(self):
        dat_mat = self.load_data_set("path/xxxxxx.data", ' ')
        num_feat = shape(dat_mat)[1]
        for i in range(num_feat):
            mean_val = mean(dat_mat[nonzero(~isnan(dat_mat[:, i].A))[0], i])
            dat_mat[nonzero(isnan(dat_mat[:, i].A))[0], i] = mean_val
        return dat_mat

    def show_picture(self, data_mat, recon_mat):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker='^', s=90)
        ax.scatter(recon_mat[:, 0].flatten().A[0], recon_mat[:, 1].flatten().A[0], marker='0', s=50, c='red')
        plt.show()

    def analyse_data(self, data_mat):
        mean_vals = mean(data_mat, axis=0)
        mean_removed = data_mat - mean_vals
        cov_mat = cov(mean_removed, rowvar=0)
        eigvals, eig_vects = linalg.eig(mat(cov_mat))
        eig_val_ind = argsort(eigvals)
        top_n_feat = 20
        eig_val_ind = eig_val_ind[:-(top_n_feat + 1): -1]
        sum_cov_score = 0
        for i in range(0, len(eig_val_ind)):
            line_cov_score = float(eigvals[eig_val_ind[i]])
            sum_cov_score += line_cov_score


if __name__ == "__main__":
    pca = PCA()
    data_mat = pca.replace_nan_with_mean()
    pca.analyse_data(data_mat)
