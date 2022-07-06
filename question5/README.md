# 任务五

## 任务需求

根据不同的标准对用户进行集群划分，如某一用户的行为特征、用户基本属性、电器设备使用、用电曲线形态等

## 实现原理
我们收集了某地区2年以来近1000家居民的每日用电量统计表

对于电力客户数据编码，我们通过简单的直方图数量分析，我们发现，大部分用电用户都在10到1000这个范围分布，我们遵循正态分布原理，我们将从最小值到最大值按照20%，20%，20%，20%，20%，切分出5个等长区间。其中10到1000这个区间占中间的60%。我们将这四个区间内的用户编码成5种客户类型：

| 客户类型 | 用电量                | 代表类型                                     |
| -------- | --------------------- | -------------------------------------------- |
| A类      | 用电量在1到13         | 低电量用户，代表个人等微型用电用户           |
| B类      | 用电量在13到153       | 中电量用户，代表家庭、私营店铺等小型用电用户 |
| C类      | 用电量在153到401      | 高电量用电用户，代表学校、小型企业等用电用户 |
| D类      | 用电量在401到1118     | 超高电量用户，代表普通工厂等用户用户         |
| E类      | 用电量在1118到1310016 | 大型用电量用户，代表重型工厂等用电用户       |

### 时间序列数据的特征工程

时间序列的特征工程一般可以分为以下几类。本次案例我们根据实际情况，选用时间戳衍生时间特征。

![image-20220706120739241](https://lzx-figure-bed.obs.dualstack.cn-north-4.myhuaweicloud.com/Figurebed/202207061207308.png)



时间戳虽然只有一列，但是也可以根据这个就衍生出很多很多变量了，具体可以分为三大类：**时间特征、布尔特征，时间差特征**。

本案例首先对日期时间进行时间特征处理，而时间特征包括年、季度、月、周、天(一年、一月、一周的第几天)

```python
i['Year'] = i.Datetime.dt.year
i['Month'] = i.Datetime.dt.month
i['day'] = i.Datetime.dt.day
i['Hour'] = i.Datetime.dt.hour
```



时间戳衍生中，另一常用的方法为布尔特征，即：

- 是否年初/年末
- 是否月初/月末
- 是否周末
- 是否节假日
- 是否特殊日期
- 是否早上/中午/晚上
- 等等

下面判断是否是周末，进行特征衍生的布尔特征转换。首先提取出日期时间的星期几。

```python
train['day of the week'] = train.Datetime.dt.dayofweek
# 返回给定日期时间的星期几
```

再判断`day of the week`是否是周末（星期六和星期日），是则返回1 ，否则返回0

```
def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0


temp = df_initial['record_date']
temp2 = df_initial.record_date.apply(applyer)
df_initial['weekend'] = temp2
df_initial.index = df_initial['record_date']
```



![](https://lzx-figure-bed.obs.dualstack.cn-north-4.myhuaweicloud.com/Figurebed/202207061214616.png)



### 探索性数据分析

首先使用探索性数据分析，从不同时间维度探索分析



分别对年, 月, 日 ,年月, 星期, 数据进行聚合, 绘制图表进行分析

![image-20220706121558167](https://lzx-figure-bed.obs.dualstack.cn-north-4.myhuaweicloud.com/Figurebed/202207061215242.png)

![image-20220706121608717](https://lzx-figure-bed.obs.dualstack.cn-north-4.myhuaweicloud.com/Figurebed/202207061216792.png)

![image-20220706121620311](https://lzx-figure-bed.obs.dualstack.cn-north-4.myhuaweicloud.com/Figurebed/202207061216351.png)

![image-20220706121628529](https://lzx-figure-bed.obs.dualstack.cn-north-4.myhuaweicloud.com/Figurebed/202207061216569.png)

### 时间重采样

◎ **重采样**(resampling)指的是将时间序列从一个频率转换到另一个频率的处理过程；
◎ 将高频率数据聚合到低频率称为**降采样**(downsampling)；
◎ 将低频率数据转换到高频率则称为**升采样**(unsampling)；

## 任务总结

### 技术路线

使用`tslearn`+`python`

数据集来自公开数据

文件说明

| 文件名              | 描述                |
| ------------------- | ------------------- |
| PowerAnalysis.ipynb | 数据处理,及数据展示 |
| requirements.txt    | 依赖包文件          |
| zhenjiang_power.csv | 数据文件            |
| model*.json         | 模型文件            |
| README.md           | 该任务文档          |



## 使用方法

```python
pip install -r requirements.txt
```

运行`PowerAnalysis.ipynb`文件即可