# -*- coding:utf-8 -*-
__aithor__ = 'yangxin_ryan'
from numpy import *
from matplotlib import pyplot as plt


class K_Means(object):

    def load_data_set(self, file_name):
        data_set = []
        fr = open(file_name)
        for line in fr.readlines():
            cur_line = line.strip().split("\t")
            flt_line = list(map(float, cur_line))
            data_set.append(flt_line)
        return data_set

    def dist_eclud(self, vec_a, vec_b):
        return sqrt(sum(power(vec_a - vec_b, 2)))

    def rand_cent(self, data_mat, k):
        m, n = shape(data_mat)
        centroids = mat(zeros((k, n)))
        for j in range(n):
            min_j = min(data_mat[:, j])
            range_j = float(max(data_mat[:, j]) - min_j)
            centroids[:, j] = mat(min_j + range_j * random.rand(k , 1))
        return centroids

    def k_means(self, data_mat, k, dis_meas=dist_eclud, create_cent=rand_cent):
        m, n = shape(data_mat)
        cluster_assment = mat(zeros(m, 2))
        centroids = create_cent(data_mat, k)
        cluster_changed = True
        while cluster_changed:
            cluster_changed = False
            for i in range(m):
                min_dist = inf
                min_index = -1
                for j in range(k):
                    dist_ji = dis_meas(centroids[:, :], data_mat[i, :])
                    if dist_ji < min_dist:
                        min_dist = dist_ji
                        min_index = j
                if cluster_assment[i:0] != min_index: cluster_changed = True
                cluster_assment[i,:] = min_index, min_dist ** 2
            for cent in range(k):
                pts_in_clust = data_mat[nonzero(cluster_assment[:, 0].A == cent)[0]]
                centroids[cent, :] = mean(pts_in_clust, axis=0)
        return centroids, cluster_assment

    def bi_kmeans(self, data_mat, k, dist_meas):
        """
        在给定数据集,所期望的簇数目和距离计算方法的条件下,函数返回聚类结果
        desc:
        """
        m, n = shape(data_mat)
        # 创建一个矩阵来存储数据集中每个点的簇分配结果及平方误差
        cluster_assment = mat(zeros((m, 2)))
        best_clust_ass = inf
        best_cent_to_split= inf
        best_new_cents = inf
        # 计算整个数据集的质心,并使用一个列表来保留所有的质心
        centroid0 = mean(data_mat, axis=0).tolist()[0]
        cent_list = [centroid0]
        # 遍历数据集中所有点来计算每个点到质心的误差值
        for j in range(m):
            cluster_assment[j, 1] = dist_meas(mat(centroid0), data_mat[j, :]) ** 2
        # 对簇不停的进行划分,直到得到想要的簇数目为止
        while len(cent_list) < k:
            # 初始化最小SSE为无穷大,用于比较划分前后的SSE
            lowest_sse = inf
            # 通过考察簇列表中的值来获得当前簇的数目,遍历所有的簇来决定最佳的簇进行划分
            for i in range(len(cent_list)):
                # 对每一个簇,将该簇中的所有点堪称一个小的数据集
                pts_in_curr_cluster = data_mat[nonzero(cluster_assment[:, 0].A == i)[0], :]
                # 将ptsInCurrCluster输入到函数kMeans中进行处理,k=2,
                # kMeans会生成两个质心(簇),同时给出每个簇的误差值
                centroid_mat, split_clust_ass = self.k_means(pts_in_curr_cluster, 2, dist_meas)
                # 将误差值与剩余数据集的误差之和作为本次划分的误差
                sse_split = sum(split_clust_ass[:, 1])
                sse_not_split = sum(cluster_assment[nonzero(cluster_assment[:, 0].A != i)[0], 1])
                # 如果本次划分的SSE值最小,则本次划分被保存
                if (sse_split + sse_not_split) < lowest_sse:
                    best_cent_to_split = i
                    best_new_cents = centroid_mat
                    best_clust_ass = split_clust_ass.copy()
                    lowest_sse = sse_split + sse_not_split
            # 找出最好的簇分配结果
            # 调用kmeans函数并且指定簇数为2时,会得到两个编号分别为0和1的结果簇
            best_clust_ass[nonzero(best_clust_ass[:, 0].A == 1)[0], 0] = len(cent_list)
            # 更新为最佳质心
            best_clust_ass[nonzero(best_clust_ass[:, 0].A == 0)[0], 0] = best_cent_to_split
            # 更新质心列表
            # 更新原质心list中的第i个质心为使用二分kMeans后bestNewCents的第一个质心
            cent_list[best_cent_to_split] = best_new_cents[0, :].tolist()[0]
            # 添加bestNewCents的第二个质心
            cent_list.append(best_new_cents[1, :].tolist()[0])
            # 重新分配最好簇下的数据(质心)以及SSE
            cluster_assment[nonzero(cluster_assment[:, 0].A == best_cent_to_split)[0], :] = best_clust_ass
        return mat(cent_list), cluster_assment

    def dist_slc(self, vec_a, vec_b):
        '''
        返回地球表面两点间的距离,单位是英里
        给定两个点的经纬度,可以使用球面余弦定理来计算亮点的距离
        '''
        # 经度和维度用角度作为单位,但是sin()和cos()以弧度为输入.
        # 可以将江都除以180度然后再诚意圆周率pi转换为弧度
        a = sin(vec_a[0, 1] * pi / 180) * sin(vec_b[0, 1] * pi / 180)
        b = cos(vec_a[0, 1] * pi / 180) * cos(vec_b[0, 1] * pi / 180) * \
            cos(pi * (vec_b[0, 0] - vec_a[0, 0]) / 180)
        return arccos(a + b) * 6371.0

    def cluster_clubs(self, file_name, img_name, num_clust=5):
        '''
        将文本文件的解析,聚类以及画图都封装在一起
        '''
        # 创建一个空列表
        dat_list = []
        # 打开文本文件获取第4列和第5列,这两列分别对应维度和经度,然后将这些值封装到datList
        for line in open(file_name).readlines():
            line_arr = line.split('\t')
            dat_list.append([float(line_arr[4]), float(line_arr[3])])
        dat_mat = mat(dat_list)
        # 调用biKmeans并使用distSLC函数作为聚类中使用的距离计算方式
        my_centroids, clust_assing = self.bi_kmeans(dat_mat, num_clust, dist_meas=self.dist_slc)
        # 创建一幅图和一个举行,使用该矩形来决定绘制图的哪一部分
        fig = plt.figure()
        rect = [0.1, 0.1, 0.8, 0.8]
        # 构建一个标记形状的列表用于绘制散点图
        scatter_markers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
        axprops = dict(xticks=[], yticks=[])
        ax0 = fig.add_axes(rect, label='ax0', **axprops)
        # 使用imread函数基于一幅图像来创建矩阵
        img_p = plt.imread(img_name)
        # 使用imshow绘制该矩阵
        ax0.imshow(img_p)
        # 再同一幅图上绘制一张新图,允许使用两套坐标系统并不做任何缩放或偏移
        ax1 = fig.add_axes(rect, label='ax1', frameon=False)
        # 遍历每一个簇并将它们一一画出来,标记类型从前面创建的scatterMarkers列表中得到
        for i in range(num_clust):
            pts_in_curr_cluster = dat_mat[nonzero(clust_assing[:, 0].A == i)[0], :]
            # 使用索引i % len(scatterMarkers)来选择标记形状,这意味这当有更多簇时,可以循环使用这标记
            marker_style = scatter_markers[i % len(scatter_markers)]
            # 使用十字标记来表示簇中心并在图中显示
            ax1.scatter(pts_in_curr_cluster[:, 0].flatten().A[0], pts_in_curr_cluster[:, 1].flatten().A[0],
                        marker=marker_style,
                        s=90)
        ax1.scatter(my_centroids[:, 0].flatten().A[0], my_centroids[:, 1].flatten().A[0], marker='+', s=300)
        plt.show()












