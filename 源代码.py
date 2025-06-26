# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:18:09 2024

@author: Hejian
"""

import pandas as pd
import numpy as np  
from scipy.linalg import null_space
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
#第一部分
df1 = pd.read_excel(r"C:\Users\Hejian\Desktop\大作业题目数据文件\数学-插值拟合求解器数据\data.xlsx",sheet_name='Sheet1')
df2 = pd.read_excel(r"C:\Users\Hejian\Desktop\大作业题目数据文件\数学-插值拟合求解器数据\data.xlsx",sheet_name='Sheet2')#导入工作表数据
df3 = pd.concat([df1,df2],ignore_index=True)#合并两个dataframe
df3.sort_values(by=0,axis=1,ascending=True,inplace=True)#按列升序排列
df3.sort_values(by=0,ascending=False,inplace=True)#按行降序排列
column_names = df3.columns.tolist()#将列索引转换为列表
for i in range(1,len(column_names)):
    if column_names[i] >= -0.5:#寻找t_n+1>=0.5时的列索引的整数位置
        break
    
list_0 = []
column_names[0] = 't_n+2/t_n+1'
for j in column_names:
    list_0.append(str(j))#将列名称数据类型转换为字符串
    
df3.columns = list_0#改变列索引
df3 = df3.reset_index(drop=True)#重置行索引
select_df3 = df3[df3['t_n+2/t_n+1']>=0]#筛选出0=<t_n+2<=2的行
df3_0 = select_df3.iloc[:,[0]]#筛选出第一列
df3_1 = select_df3.iloc[:,i:len(column_names)]#筛选出-0.5=<t_n+1<=0的列
df3_2 = pd.concat([df3_0, df3_1], axis=1)#合并两个数表
print(f'行数: {df3_2.shape[0]}, 列数: {df3_2.shape[1]}')
df3_2 = df3_2.reset_index(drop=True)#重置行索引
df3_2.to_excel(r"C:\Users\Hejian\Desktop\大作业题目数据文件\数学-插值拟合求解器数据\data2.xlsx",index=False)#将dataframe写入工作表 
#第二部分(1)
x_k = []
y_l = []
for k in range(0,201):
    x_k.append(float(f'{0.01*k:.4f}'))
    
for l in range(0,151):
    y_l.append(float(f'{0.02*l:.4f}'))

u_p = []
v_q = []#不包含重复数据
u_p_1 = []
v_q_1 = []#包含重复数据
A = np.array([[1,-1,1,0,0,0,0,0,0,0,0,0],
              [0,1,-1,1,0,0,0,0,0,0,0,0],
              [0,0,1,-1,1,0,0,0,0,0,0,0],
              [0,0,0,1,-1,1,0,0,0,0,0,0],
              [0,0,0,0,1,-1,1,0,0,0,0,0],
              [0,0,0,0,0,1,-1,1,0,0,0,0],
              [0,0,0,0,0,0,1,-1,1,0,0,0],
              [0,0,0,0,0,0,0,1,-1,1,0,0],
              [0,0,0,0,0,0,0,0,1,-1,1,0],
              [0,0,0,0,0,0,0,0,0,1,-1,1]])#系数矩阵
rank_A = np.linalg.matrix_rank(A)#求出系数矩阵的秩为10
ns = null_space(A)#求出齐次线性方程组的基础解系
'''基础解系为ns=
[[-0.16895503  0.37164615]
 [-0.40633253  0.03950373]
 [-0.23737749 -0.33214243]
 [ 0.16895503 -0.37164615]
 [ 0.40633253 -0.03950373]
 [ 0.23737749  0.33214243]
 [-0.16895503  0.37164615]
 [-0.40633253  0.03950373]
 [-0.23737749 -0.33214243]
 [ 0.16895503 -0.37164615]
 [ 0.40633253 -0.03950373]
 [ 0.23737749  0.33214243]]'''
B = np.array([[ns[0][0],ns[0][1]],
              [ns[1][0],ns[1][1]]])
'''以上两行的作用是构造新的线性方程组:
k1*s11+k2*s12=x_k,
k1*s21+k2*s22=y_l,
其中s11=ns[0][0],s12=ns[0][1]，以此类推'''  
xy2 = []#创造待插值的点集
for j1 in y_l:
    for i1 in x_k:
        C = np.array([[i1],
                      [j1]])
        D = np.linalg.solve(B,C)#求出k_1和k_2并保存在D中
        a = round(D[0][0]*ns[10][0] + D[1][0]*ns[10][1],4)#得到t_n+1
        u_p_1.append(a)
        if a not in u_p:
            u_p.append(a)
        
        b = round(D[0][0]*ns[11][0] + D[1][0]*ns[11][1],4)#得到t_n+2
        v_q_1.append(b)
        if b not in v_q:
            v_q.append(b)
            
        xy2.append([a,b])
        
p_0 = len(u_p)
q_0 = len(v_q)
print(f'p_0={p_0},q_0={q_0}')#打印p_0,q_0
#第二部分(2)
column_names.pop(0)
t_n_1 = column_names
t_n_2 = df3['t_n+2/t_n+1'].tolist()
for i2 in t_n_1:
    i2 = round(i2,4)#将数字四舍五入到四位小数
for j2 in t_n_2:
    j2 = round(j2,4)#将数字四舍五入到四位小数

z = []#创建z的二维列表
values = []
for i4 in range(0,50):
    values = df3.iloc[i4].tolist()
    values.pop(0)
    z.append(values)
    
new_z = np.array(z).reshape(2500)#将z展开成一维数组
points = []
for i12 in t_n_2:
    for j12 in t_n_1:
        points.append([j12,i12])
        
points = np.array(points)
'''以下为插值'''
X2,Y2 = np.meshgrid(u_p,v_q)#创建网格
f2 = griddata(points,new_z,(X2,Y2),method='cubic')#进行插值
'''以下为拟合'''
f_3 = []#建立与每个点一一对应的函数值的列表
for i13 in xy2:
    d = u_p.index(i13[0])#找出每个点的横坐标在f2中的第二维索引
    e = v_q.index(i13[1])#找出每个点的纵坐标在f2中的第一维索引
    f_3.append(f2[e][d])
    
f_3 = np.array(f_3)
xy = []
for i8 in y_l:
    for j8 in x_k:
        xy.append([j8,i8])
        
xy = np.array(xy)#创建需要拟合的点的点集
popt = []#用于保存拟合函数的系数列表        
sigma = []#用于保存误差值
def poly(order,arr,xy):
    z1 = 0
    x,y = xy[:,0],xy[:,1]#分解二维数组为x和y
    for i18 in range(0,order+1):
        for j18 in range(0,order+1):
            z1 += arr[i18*(order+1)+j18]*x**i18*y**j18#建立拟合函数
    return z1

def polynomial_func(xy,*coefficients):#coefficients为储存系数的列表
    order = int(len(coefficients)**(1/2)-1)#order为多项式阶数
    return poly(order,coefficients,xy)

for i11 in range(1,7):
    popt1,pcov1 = curve_fit(polynomial_func,xy,f_3,p0=[0]*((i11+1)**2))
    '''以上一行进行拟合，分别得到系数数组和协方差，p0的作用为初始化系数列表'''
    popt.append(np.round(popt1,2))
    p_1 = polynomial_func(xy,*popt1)#调用拟合函数得到拟合值
    sigma1 = 0#初始化误差值
    for i9 in range(0,len(p_1)):
        sigma1 += (f_3[i9]-p_1[i9])**2#计算误差值
        
    sigma.append(sigma1)
    print(f'k={i11}时,sigma={sigma1}')

c_rs = np.array(popt[-1]).reshape(i11+1,i11+1)#c_rs为k=k_min时的系数矩阵
print(f'k=k_min时,系数矩阵为c_rs=\n{c_rs}')  
#第三部分
'''绘制热力图'''
z_1 = []
values_1 = []
for i5 in range(0,12):
    values_1 = df3_2.iloc[i5].tolist()
    values_1.pop(0)
    z_1.append(values_1)#从df3_2中创建z_1的二维列表

plt.figure(figsize=(10, 6))#创建画布    
sns.heatmap(z_1,cmap='coolwarm',annot_kws={"size": 7},annot=True,fmt='.4f')
plt.title('figure1')
plt.show()
'''绘制平面散点图'''
'''由于scatter函数的x,y参数和颜色值c之间存在一一对应的关系,既然颜色值c(也就是f(x,y))有2500个值,
那么x,y也需要各有2500个,所以我们首先用for语句将t_n+1和t_n+2扩充为2500个值。'''
x_1 = []
y_1 = []
for i6 in range(0,50):
    for j6 in t_n_1:
        x_1.append(j6)#将t_n_1扩充到2500个值
        
for i7 in t_n_2:
    for j9 in range(0,50):
        y_1.append(i7)#将t_n_2扩充到2500个值
'''创建子图1'''     
plt.subplot(1,2,1)
cmap = plt.get_cmap('jet')#设置颜色条渐变色
plt.scatter(x_1,y_1,c=new_z,s=0.5,marker='.',cmap=cmap)  
plt.colorbar()#添加颜色条 
'''创建子图2'''    
plt.subplot(1,2,2)
camp = plt.get_cmap('jet')#设置颜色条渐变色
plt.scatter(u_p_1,v_q_1,c=f_3,marker='.',cmap=cmap) 
plt.colorbar()#添加颜色条 
plt.show()
'''绘制立体散点图'''
x_k_1 = []
y_l_1 = []
for k1 in range(0,9):
    x_k_1.append(float(f'{0.25*k1:.4f}'))#创造x_i^*的列表并保留四位小数
    
for l1 in range(0,7):
    y_l_1.append(float(f'{0.5*l1:.4f}'))#创造y_j^*的列表并保留四位小数
    
xy1 = []
for i10 in y_l_1:
    for j10 in x_k_1:
        xy1.append([j10,i10])#创造(x_i^*,y_j^*)的点集
        
xy1 = np.array(xy1)
'''以下两行为多项式阶数k=2时的拟合函数p(x,y),以绘制曲面p(x,y)'''
popt2,pcov2 = curve_fit(polynomial_func,xy,f_3,p0=[0]*9)
p_2 = polynomial_func(xy,*popt2)
'''重新定义一个函数,使xy1中的63个点可以通过该函数拟合函数值'''
def p(xy1,c_00,c_01,c_02,c_10,c_11,c_12,c_20,c_21,c_22):
    x,y = xy1[:,0], xy1[:,1]#分解二维数组为x和y
    return c_00+c_01*y+c_02*y**2+c_10*x+c_11*x*y+c_12*x*y**2+c_20*x**2+c_21*x**2*y+c_22*x**2*y**2

p_3 = p(xy1,*popt2)#得到由63个点(x_i^*,y_j^*)拟合出的函数值
xy3 = []
for j14 in y_l_1:
    for i14 in x_k_1:
        E = np.array([[i14],
                      [j14]])
        F = np.linalg.solve(B,E)#求出k_1和k_2并保存在F中
        f = round(F[0][0]*ns[10][0] + F[1][0]*ns[10][1],4)
        g = round(F[0][0]*ns[11][0] + F[1][0]*ns[11][1],4)
        xy3.append([f,g])
        
f_2 = []
for i15 in xy3:
    m = u_p.index(i15[0])
    n = v_q.index(i15[1])
    f_2.append(f2[n][m])
'''绘图'''          
fig = plt.figure()  
ax = fig.add_subplot(111, projection='3d')  
ax.scatter(xy[:,0],xy[:,1],p_2,s=0.02,color='blue')#绘制平面p(x,y)
ax.scatter(xy1[:,0],xy1[:,1],f_2,s=7,color='red')#绘制散点
ax.view_init(azim=20,elev=10)#转换立体散点图的视角
'''以下代码是为了将63个散点与相同横纵坐标的拟合点连线'''
xy1_x = np.array(xy1[:,0].tolist()*2)#将xy1中的点的横坐标分别提取出来，再复制所有横坐标形成列表
xy1_y = np.array(xy1[:,1].tolist()*2)#将xy1中的点的纵坐标分别提取出来，再复制所有纵坐标形成列表
z1 = np.array(p_3.tolist()+f_2)#所有点对应的函数值，包括拟合和准确的函数值
df4 = pd.DataFrame({'xy_x':xy1_x,'xy_y':xy1_y,'z1':z1})
grouped = df4.groupby(['xy_x','xy_y'])
for name,group in grouped:  
    if len(group) > 1:#只有当点的数量大于1时才绘制连接线  
        z1_values = group['z1'].values#提取z坐标值  
        x_value, y_value = name#由于x和y坐标相同可以直接取第一个点的x和y坐标
        ax.plot([x_value]*len(z1_values),[y_value]*len(z1_values),z1_values,color='green',lw=1)#绘制连接线
        
plt.title('figure3')
plt.show()
'''以下代码是为了打印实验要求中的数表'''
xy1_x1 = xy1[:,0].tolist()
xy1_y1 = xy1[:,1].tolist()
p2_2 = p_3.tolist()
df5 = pd.DataFrame({'x':xy1_x1,'y':xy1_y1,'f':f_2,'p':p2_2})
print(f'数表为:\n{df5}')