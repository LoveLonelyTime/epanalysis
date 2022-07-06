import xlrd
import csv
# 打开excle
x1 = xlrd.open_workbook('dataset.xls')
# 获取工作表
table = x1.sheets()[0]
# 获取用户id
rows = table.nrows
row = table.col_values(0)
sgl_amt = table.col_values(2)
uer_id = set()
for i in row:
    uer_id.add(i)
num_uer_id = len(uer_id)-1
uers_data = [[0 for x in range(3)]for y in range(num_uer_id)]
nuer_id = 0  # 上一轮循环读取的id
nuer_id_j = -1  # 当前id序号
cir_j = 0  # 当前循环次数（-1）
# 获得用户缴费次数和总金额
for i in row:
    if cir_j != 0:
        if i == nuer_id:
            uers_data[nuer_id_j][1] += 1
            uers_data[nuer_id_j][2] += sgl_amt[cir_j]
        else:
            nuer_id_j += 1
            uers_data[nuer_id_j][0] = i
            uers_data[nuer_id_j][1] += 1
            uers_data[nuer_id_j][2] += sgl_amt[cir_j]
        nuer_id = i
    cir_j += 1
# 计算平均次数和平均金额
uers_ave_fre = 0
uers_ave_amt = 0
for i in uers_data:
    uers_ave_fre += i[1]
    uers_ave_amt += i[2]
uers_ave_fre /= num_uer_id
uers_ave_amt /= num_uer_id
# 输出文件1
with open('居民客户的用电缴费习惯分析1.csv', 'w+', encoding='utf-8', newline='') as f:
    hearders = ['平均次数', '平均金额']
    writer = csv.DictWriter(f, fieldnames=hearders)
    writer.writeheader()
    writer.writerow({'平均次数': str(uers_ave_fre), '平均金额': str(uers_ave_amt)})
# 区分用户类型
uers_lab = [[0, 0]for y in range(num_uer_id)]
for i in range(num_uer_id):
    uers_lab[i][0] = uers_data[i][0]
    if uers_data[i][1] > uers_ave_fre:
        if uers_data[i][2] > uers_ave_amt:
            uers_lab[i][1] = '高价值型客户'
        else:
            uers_lab[i][1] = '大众型客户'
    else:
        if uers_data[i][2] > uers_ave_amt:
            uers_lab[i][1] = '潜力型客户'
        else:
            uers_lab[i][1] = '低价值型客户'
# 输出文件2
with open('居民客户的用电缴费习惯分析2.csv', 'w+', encoding='utf-8', newline='') as f:
    hearders = ['用户id', '用户类型']
    writer = csv.DictWriter(f, fieldnames=hearders)
    writer.writeheader()
    for i in range(num_uer_id):
        writer.writerow(
            {'用户id': str(int(uers_lab[i][0])), '用户类型': str(uers_lab[i][1])})
