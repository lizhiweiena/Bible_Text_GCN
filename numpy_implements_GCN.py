# coding: utf-8
# @Time    : 2019/5/23 9:19
# @Author  : 李志伟
# @Email   : lizhiweiena@163.com

# 原文链接
# https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780
# 中文链接
# https://mp.weixin.qq.com/s/sg9O761F0KHAmCPOfMW_kQ

# GCN 是一种 可直接作用于图并利用其结构信息 的强大神经网络。

# 以下代码可以看到：
# 1）如何通过 GCN 的隐藏层传播。
# 2）GCN 如何聚合来自前一层的信息，以及这种机制如何生成图中节点的有用特征表征。

# 图卷积网络（GCN）是一个对 图数据 进行操作的神经网络。
# 给定图 G = (V, E)，GCN 的输入为：
#   一个输入维度为 N × F⁰ 的特征矩阵 X，其中 N 是图网络中的节点数而 F⁰ 是每个节点的输入特征数。
#   一个图结构的维度为 N × N 的矩阵表征，例如图 G 的邻接矩阵 A。

import numpy as np

# 有向图的邻接矩阵表征
# A = np.matrix([
#     [0, 1, 0, 0],
#     [0, 0, 1, 1],
#     [0, 1, 0, 0],
#     [1, 0, 1, 0]
# ], dtype=float)
# # print(A)
#
# # 解决问题1：增加自环
# I = np.matrix(np.eye(A.shape[0]))
# # print(I)
# # [[1. 0. 0. 0.]
# #  [0. 1. 0. 0.]
# #  [0. 0. 1. 0.]
# #  [0. 0. 0. 1.]]
# A_hat = A + I
# # print(A_hat)
# # [[1. 1. 0. 0.]
# #  [0. 1. 1. 1.]
# #  [0. 1. 1. 0.]
# #  [1. 0. 1. 1.]]
#
#
# # 抽取出特征（基于每个节点的索引为其生成两个整数特征）
# X = np.matrix([
#     [i, -i]
#     for i in range(A.shape[0])
# ], dtype=float)
# # print(X)
# # [[ 0.  0.]
# #  [ 1. -1.]
# #  [ 2. -2.]
# #  [ 3. -3.]]
#
#
# # 我们现在已经建立了一个图，其邻接矩阵为 A，输入特征的集合为 X
# # 应用传播规则之后
# # print(A * X)
# # [[ 1. -1.]
# #  [ 5. -5.]
# #  [ 1. -1.]
# #  [ 2. -2.]]
#
# # 现在，由于每个节点都是自己的邻居，每个节点在对相邻节点的特征求和过程中也会囊括自己的特征！
# # 增加自环之后结果
# # print(A_hat * X)
# # [[ 1. -1.]
# #  [ 6. -6.]
# #  [ 3. -3.]
# #  [ 5. -5.]]
#
# # 每个节点的表征（每一行）现在是其相邻节点特征的和！换句话说，图卷积层将每个节点表示为其相邻节点的聚合
# # 请注意，在这种情况下，如果存在从 v 到 n 的边，则节点 n 是节点 v 的邻居。
#
#
# # 解决问题2：对特征表征进行归一化处理
# # 1. 计算出节点的度矩阵（此处指入度）
# D = np.array(np.sum(A, axis=0))[0]
# D = np.matrix(np.diag(D))
# # print(D)
# # [[1. 0. 0. 0.]
# #  [0. 2. 0. 0.]
# #  [0. 0. 2. 0.]
# #  [0. 0. 0. 1.]]
#
# # 2. 对邻接矩阵进行变换
# # print(D**-1 * A)  # 邻接矩阵中每一行的权重（值）都除以该行对应节点的度
#
# # 变换之前的A
# # [[0, 1, 0, 0],
# #  [0, 0, 1, 1],
# #  [0, 1, 0, 0],
# #  [1, 0, 1, 0]]
# # 变换之后的A
# # [[0.  1.  0.  0. ]
# #  [0.  0.  0.5 0.5]
# #  [0.  0.5 0.  0. ]
# #  [1.  0.  1.  0. ]]
#
# # 3. 对变换后的邻接矩阵应用传播规则
# # print(D**-1 * A * X)
# # [[ 1.  -1. ]
# #  [ 2.5 -2.5]
# #  [ 0.5 -0.5]
# #  [ 2.  -2. ]]
# # 上述结果的原因：（变换后）邻接矩阵的权重对应于相邻节点特征加权和的权重
#
#
# ##################################################################
# # 将自环和归一化技巧结合起来
# # 1. 添加权重
# # D_hat是A_hat的(入)度矩阵
# D_hat = np.matrix([
#     [2, 0, 0, 0],
#     [0, 3, 0, 0],
#     [0, 0, 3, 0],
#     [0, 0, 0, 2]
# ])
# W = np.matrix([
#     [1, -1],
#     [-1, 1]
# ])
# # print(D_hat**-1 * A_hat * X * W)
# # [[ 1. -1.]
# #  [ 4. -4.]
# #  [ 2. -2.]
# #  [ 5. -5.]]
# # 若我们想要减小输出特征表征的维度，我们可以减小权重矩阵W的规模
# W1 = np.matrix([
#     [1],
#     [-1]
# ])
# # print(D_hat**-1 * A_hat * X * W1)
# # [[1.]
# #  [4.]
# #  [2.]
# #  [5.]]
#
# # 2. 添加激活函数
# # 本文选择保持特征表征的维度，并应用ReLU激活函数
# W2 = np.matrix([
#     [1, -1],
#     [-1, 1]
# ])

# print(relu(D_hat**-1 * A_hat * X * W2))

###########################################################
# 在真实场景下的应用

from networkx import to_numpy_matrix, karate_club_graph
import matplotlib.pyplot as plt


# 1. 构建GCN
zkc = karate_club_graph()
order = sorted(list(zkc.nodes()))

A = to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())

A_hat = A + I
D_hat = np.array(np.sum(A_hat, axis=0))[0]
D_hat = np.matrix(np.diag(D_hat))

# 2. 随机初始化权重
W_1 = np.random.normal(loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(loc=0, size=(W_1.shape[1], 2))

# 3. 堆叠 GCN 层。
# 这里，我们只使用单位矩阵作为特征表征，即每个节点被表示为一个 one-hot 编码的类别变量。

def gcn_layer(A_hat, D_hat, X, W):
    return relu(D_hat**-1 * A_hat * X * W)


H_1 = gcn_layer(A_hat, D_hat, I, W_1)
H_2 = gcn_layer(A_hat, D_hat, H_1, W_2)

output = H_2

# 4. 抽取特征表征
feature_representations = {
    node: np.array(output)[node]
    for node in zkc.nodes()
}

plt.plot(feature_representations)






