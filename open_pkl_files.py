# coding: utf-8
# @Time    : 2019/5/29 8:46
# @Author  : 李志伟
# @Email   : lizhiweiena@163.com


import pickle
f = open('./data/df_data.pkl', 'rb')
data = pickle.load(f)
print(data)
