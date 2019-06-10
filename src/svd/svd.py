# -*- coding:utf-8 -*-
__author__ = 'yangxin_ryan'
from numpy import linalg as la
from numpy import *


class SVD(object):

    def load_ex_data3(self):
        return [[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
                [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
                [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
                [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
                [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
                [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
                [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
                [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]

    def load_ex_data2(self):
        return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
                [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
                [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
                [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
                [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
                [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
                [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
                [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
                [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
                [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

    def load_ex_data(self):
        return [[0, -1.6, 0.6],
                [0, 1.2, 0.8],
                [0, 0, 0],
                [0, 0, 0]]

    def eclud_sim(self, in_a, in_b):
        """
        相似度计算，假定in_a和in_b都是列向量
        基于欧式距离
        :param in_a:
        :param in_b:
        :return:
        """
        return 1.0 / (1.0 + la.norm(in_a - in_b))

    def pears_sim(self, in_a, in_b):
        """
        pears_sim()函数会检查是否存在3哥或更多的点
        :param in_a:
        :param in_b:
        :return:
        """
        if len(in_a) < 3:
            return 1.0
        return 0.5 + 0.5 * corrcoef(in_a, in_b, rowvar=0)[0][1]

    def cos_sim(self, in_a, in_b):
        num = float(in_a.T * in_b)
        denom = la.norm(in_a) * la.norm(in_b)
        return 0.5 + 0.5 * (num / denom)

    def stand_est(self, data_mat, user, sim_meas, item):
        """

        :param data_mat:
        :param user:
        :param sim_meas:
        :param item:
        :return:
        """
        n = shape(data_mat)[1]
        sim_total = 0.0
        rat_sim_total = 0.0
        for j in range(n):
            user_rating = data_mat[user, j]
            if user_rating == 0:
                continue
            over_lap = nonzero(logical_and(data_mat[:, item].A > 0, data_mat[:, j].A > 0))[0]
            if len(over_lap) == 0:
                similarity = 0
            else:
                similarity = sim_meas(data_mat[over_lap, item], data_mat[over_lap, j])
            sim_total += similarity
            rat_sim_total += similarity * user_rating
        if sim_total == 0:
            return 0
        else:
            return rat_sim_total / sim_total

    def svd_est(self, data_mat, user, sim_meas, item):
        n = shape(data_mat)[1]
        sim_total = 0.0
        rat_sim_total = 0.0
        u, sigma, VT = la.svd(data_mat)
        sig4 = mat(eye(4) * sigma[:, 4])
        xformed_items = data_mat.T * u[:, :4] * sig4.I
        print('dataMat', shape(data_mat))
        print('U[:, :4]', shape(u[:, :4]))
        print('Sig4.I', shape(sig4.I))
        print('VT[:4, :]', shape(VT[:4, :]))
        print('xformedItems', shape(xformed_items))

        # 对于给定的用户，for循环在用户对应行的元素上进行遍历
        # 这和standEst()函数中的for循环的目的一样，只不过这里的相似度计算时在低维空间下进行的。
        for j in range(n):
            user_rating = data_mat[user, j]
            if user_rating == 0 or j == item:
                continue
            # 相似度的计算方法也会作为一个参数传递给该函数
            similarity = sim_meas(xformed_items[item, :].T, xformed_items[j, :].T)
            # for 循环中加入了一条print语句，以便了解相似度计算的进展情况。如果觉得累赘，可以去掉
            print('the %d and %d similarity is: %f' % (item, j, similarity))
            # 对相似度不断累加求和
            sim_total += similarity
            # 对相似度及对应评分值的乘积求和
            rat_sim_total += similarity * user_rating
        if sim_total == 0:
            return 0
        else:
            # 计算估计评分
            return rat_sim_total / sim_total

    def recommend(self, data_mat, user, n=3, sim_meas=cos_sim, est_method=stand_est):
        """svdEst( )
        Args:
            dataMat         训练数据集
            user            用户编号
            simMeas         相似度计算方法
            estMethod       使用的推荐算法
        Returns:
            返回最终 N 个推荐结果
        """
        # 寻找未评级的物品
        # 对给定的用户建立一个未评分的物品列表
        unrated_items = nonzero(data_mat[user, :].A == 0)[1]
        # 如果不存在未评分物品，那么就退出函数
        if len(unrated_items) == 0:
            return 'you rated everything'
        # 物品的编号和评分值
        item_scores = []
        # 在未评分物品上进行循环
        for item in unrated_items:
            # 获取 item 该物品的评分
            estimated_score = est_method(data_mat, user, sim_meas, item)
            item_scores.append((item, estimated_score))
        # 按照评分得分 进行逆排序，获取前N个未评级物品进行推荐
        return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[: n]

    def analyse_data(self, sigma, loop_num=20):
        """analyse_data(分析 Sigma 的长度取值)
        Args:
            Sigma         Sigma的值
            loopNum       循环次数
        """
        # 总方差的集合（总能量值）
        sig2 = sigma ** 2
        sigma_sum = sum(sig2)
        for i in range(loop_num):
            sigma_i = sum(sig2[:i + 1])
            '''
            根据自己的业务情况，就行处理，设置对应的 Singma 次数
            通常保留矩阵 80% ～ 90% 的能量，就可以得到重要的特征并取出噪声。
            '''
            print('主成分：%s, 方差占比：%s%%' % (format(i + 1, '2.0f'), format(sigma_i / sigma_sum * 100, '4.2f')))

    # 图像压缩函数
    # 加载并转换数据
    def imgLoadData(self, file_name):
        myl = []
        # 打开文本文件，并从文件以数组方式读入字符
        for line in open(file_name).readlines():
            new_row = []
            for i in range(32):
                new_row.append(int(line[i]))
            myl.append(new_row)
        # 矩阵调入后，就可以在屏幕上输出该矩阵
        my_mat = mat(myl)
        return my_mat

    # 打印矩阵
    def print_mat(self, in_mat, thresh=0.8):
        # 由于矩阵保护了浮点数，因此定义浅色和深色，遍历所有矩阵元素，当元素大于阀值时打印1，否则打印0
        for i in range(32):
            for k in range(32):
                if float(in_mat[i, k]) > thresh:
                    print(1, )
                else:
                    print(0, )
            print('')

    # 实现图像压缩，允许基于任意给定的奇异值数目来重构图像
    def img_compress(self, num_sv=3, thresh=0.8):
        """imgCompress( )
        Args:
            numSV       Sigma长度
            thresh      判断的阈值
        """
        # 构建一个列表
        my_mat = self.imgLoadData(('xxxxx'))
        # 对原始图像进行SVD分解并重构图像e
        self.print_mat(my_mat, thresh)
        # 通过Sigma 重新构成SigRecom来实现
        # Sigma是一个对角矩阵，因此需要建立一个全0矩阵，然后将前面的那些奇异值填充到对角线上。
        u, sigma, vt = la.svd(my_mat)
        # SigRecon = mat(zeros((numSV, numSV)))
        # for k in range(numSV):
        #     SigRecon[k, k] = Sigma[k]
        # 分析插入的 Sigma 长度
        self.analyse_data(sigma, 20)
        sig_recon = mat(eye(num_sv) * sigma[: num_sv])
        recon_mat = u[:, :num_sv] * sig_recon * vt[:num_sv, :]
        print("****reconstructed matrix using %d singular values *****" % num_sv)
        self.print_mat(recon_mat, thresh)


if __name__ == "__main__":
    svd = SVD()
    # 计算相似度的方法
    my_mat = mat(svd.load_ex_data3())
    # print(myMat)
    # 计算相似度的第一种方式
    print(svd.recommend(my_mat, 1, est_method=svd.svd_est))
    # 计算相似度的第二种方式
    print(svd.recommend(my_mat, 1, est_method=svd.svd_est, sim_meas=svd.pears_sim))
    # 默认推荐（菜馆菜肴推荐示例）
    print(svd.recommend(my_mat, 2))

