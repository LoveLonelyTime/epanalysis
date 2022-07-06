import pickle
#加载模型
with open('model/gnb.pickle', 'rb') as f:
    model = pickle.load(f)
import pandas as pd
from datetime import date
current_date = date.today()
filename = './dataset/test.xls' #这里输入需要预测的文件名，格式为xls

#处理数据
table=pd.read_excel(filename)
r=table.groupby('用户编号')['缴费日期'].max().reset_index()
r['R']=(pd.to_datetime(current_date) - r['缴费日期']).dt.days
r=r[['用户编号','R']]
#引入日期标签辅助列
table['日期标签'] = table['缴费日期'].astype(str).str[:10]
#把单个用户一天内订单合并 主要是将单日的付款日期设置为1，因为一个客户不可能在同一时刻下单多次
dup_f = table.groupby(['用户编号','日期标签'])['缴费日期'].count().reset_index()
#对合并后的用户统计频次
f = dup_f.groupby('用户编号')['缴费日期'].count().reset_index()
f.columns = ['用户编号','F']
sum_m = table.groupby('用户编号')['缴费金额（元）'].sum().reset_index()
sum_m.columns = ['用户编号','总支付金额']
com_m = pd.merge(sum_m,f,left_on = '用户编号',right_on = '用户编号',how = 'inner')
com_m['M'] = com_m['总支付金额']
com_m['R'] = r['R']
com_m=com_m[["用户编号",'R','F','M']]
mean_num1=com_m['R'].mean()
std_num1=com_m['R'].std()
mean_num2=com_m['F'].mean()
std_num2=com_m['F'].std()
mean_num3=com_m['M'].mean()
std_num3=com_m['M'].std()
def func1(val):
    num=(val-mean_num1)/std_num1
    return num
def func2(val):
    num=(val-mean_num2)/std_num2
    return num
def func3(val):
    num=(val-mean_num3)/std_num3
    return num
com_m.R=com_m.apply(lambda x: func1(x.R),axis=1)
com_m.F=com_m.apply(lambda x: func2(x.F),axis=1)
com_m.M=com_m.apply(lambda x: func3(x.M),axis=1)
com_m['CLV']=(-0.2)*com_m['R']+0.35*com_m['F']+0.45*com_m['M']
com_m['default']=0
r=com_m.sort_values(by='CLV',ascending=False)
num=r['CLV'].index
num=num[0]
X=com_m[['CLV','default']]
res=model.predict(X)
res=pd.DataFrame(res)
ans=res[0].loc[num]+1#获取类别
ans=str(ans)
p_table=pd.DataFrame(model.predict_proba(X))
p_table.columns=['1','2','3','4','5','6','7']
result=p_table.sort_values(by=ans,ascending=False)
ans=com_m.loc[result.head().index.tolist(),:]
#获取用户编号，展示以及持久化
df=pd.DataFrame(ans,columns=['用户编号'])
df.index=['1','2','3','4','5']
print("最有可能成为高价值客户的TOP5:\n")
print(df)
df.to_csv('居民客户的用电缴费习惯分析 3.csv')