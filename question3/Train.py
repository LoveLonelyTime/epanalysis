import pickle  # pickle模块
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from datetime import date
current_date = date.today()
filename = './dataset/data2.xls'
table = pd.read_excel(filename)
r = table.groupby('用户编号')['缴费日期'].max().reset_index()
r['R'] = (pd.to_datetime(current_date) - r['缴费日期']).dt.days
r = r[['用户编号', 'R']]
# 引入日期标签辅助列
table['日期标签'] = table['缴费日期'].astype(str).str[:10]

# 把单个用户一天内订单合并 主要是将单日的付款日期设置为1，因为一个客户不可能在同一时刻下单多次
dup_f = table.groupby(['用户编号', '日期标签'])['缴费日期'].count().reset_index()

# 对合并后的用户统计频次
f = dup_f.groupby('用户编号')['缴费日期'].count().reset_index()
f.columns = ['用户编号', 'F']
sum_m = table.groupby('用户编号')['缴费金额（元）'].sum().reset_index()
sum_m.columns = ['用户编号', '总支付金额']
com_m = pd.merge(sum_m, f, left_on='用户编号', right_on='用户编号', how='inner')
com_m['M'] = com_m['总支付金额']
com_m['R'] = r['R']
com_m = com_m[["用户编号", 'R', 'F', 'M']]
mean_num1 = com_m['R'].mean()
std_num1 = com_m['R'].std()
mean_num2 = com_m['F'].mean()
std_num2 = com_m['F'].std()
mean_num3 = com_m['M'].mean()
std_num3 = com_m['M'].std()


def func1(val):
    num = (val-mean_num1)/std_num1
    return num


def func2(val):
    num = (val-mean_num2)/std_num2
    return num


def func3(val):
    num = (val-mean_num3)/std_num3
    return num


com_m.R = com_m.apply(lambda x: func1(x.R), axis=1)
com_m.F = com_m.apply(lambda x: func2(x.F), axis=1)
com_m.M = com_m.apply(lambda x: func3(x.M), axis=1)
com_m['CLV'] = 0.2*com_m['R']+0.4*com_m['F']+0.4*com_m['M']
com_m['default'] = 0
Clv = com_m[['用户编号', 'CLV', 'default']]
# KMeans聚类
x = Clv[['CLV', 'default']]
plt.scatter(Clv['CLV'], Clv['default'], marker='o')
plt.show()
model = KMeans(n_clusters=7, max_iter=1000000)
model.fit(x)
predict_y = model.predict(x)
plt.scatter(Clv['CLV'], Clv['default'], c=predict_y)
plt.show()
df = pd.DataFrame(predict_y)
df.columns = ['类别']
com_m['类别'] = df['类别']
X = com_m[['CLV', 'default']]
Y = com_m['类别']
xtrain, xtest, ytrain, ytest = train_test_split(
    X, Y, test_size=1, random_state=420)
gnb = GaussianNB()
gnb.fit(xtrain, ytrain)
# 保存Model(注:model文件夹要预先建立，否则会报错)
with open('model/gnb.pickle', 'wb') as f:
    pickle.dump(gnb, f)
