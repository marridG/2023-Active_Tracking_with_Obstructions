import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import matplotlib;matplotlib.use('tkagg')

plt.rcParams['font.sans-serif']=['SimHei']  # 指定默认字体 SimHei为黑体
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

# 数据１
data1 = []
with open('./example/glass/points.txt', "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().strip('[').strip(']').split(',')
        line = list(map(float,line))
        data1.append(line)
f.close()


data1 = np.array(data1)
x1 = list(data1[:, 0])  # [ 0  3  6  9 12 15 18 21]
y1 = list(data1[:, 1])  # [ 1  4  7 10 13 16 19 22]
z1 = list(data1[:, 2])  # [ 2  5  8 11 14 17 20 23]



# 绘制散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1, y1, z1, c='r', s=5, label='glass')


# 绘制图例
ax.legend(loc='best')

# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

# 展示
plt.show()
