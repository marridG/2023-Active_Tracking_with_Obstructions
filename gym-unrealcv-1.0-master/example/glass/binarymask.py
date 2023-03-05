import numpy as np
import cv2 as cv
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
# import matplotlib;matplotlib.use('tkagg')
import random

# plt.rcParams['font.sans-serif']=['SimHei']  # 指定默认字体 SimHei为黑体
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

class Binarymask():
    def __init__(self, neg_reward, camera_distance=100):

        self.out_fn = './example/glass/out.ply'
        self.trajectary = '/data3/songmingyu/glass/gym-unrealcv-1.0/example/glass/out.csv'
        self.draw_path = './example/glass/draw/draw.jpg'
        self.camera_distacne = camera_distance
        self.threshold_len_min = 10  # change this in .json
        self.threshold_len_max = 1000 # change this in .json
        self.neg_reward = neg_reward
        self.showstep = 0
        self.throshold_near = 5000

        self.ll,self.lr,self.rl,self.rr = [0,0],[0,0],[0,0],[0,0]
        self.ll = np.array(self.ll)
        self.lr = np.array(self.lr)
        self.rl = np.array(self.rl)
        self.rr = np.array(self.rr)
        
    def pair(self, data_map, imgL, imgR, move_side, camera_location, pitch=0, yaw=0):
        imgL = cv.resize(imgL,(720,540))
        imgR = cv.resize(imgR,(720,540))
        imgL, numL = self.largestConnectComponent(imgL/255)
        imgR, numR = self.largestConnectComponent(imgR/255)
        if numL<500 or numR<500:
            print('too small mask!')
            return data_map,[], self.neg_reward

        self.ll[0], self.ll[1], self.lr[0], self.lr[1] = self.find_point(imgL)
        self.rl[0], self.rl[1], self.rr[0], self.rr[1] = self.find_point(imgR)

        disp =np.zeros((540,720))
        if move_side == 1: #right:
            disp[self.ll[0]][self.ll[1]]=self.ll[1]-self.rl[1]
            disp[self.lr[0]][self.lr[1]]=self.lr[1]-self.rr[1]
        elif move_side == 2: #left:
            disp[self.rl[0]][self.rl[1]]=self.ll[1]-self.rl[1]
            disp[self.rr[0]][self.rr[1]]=self.lr[1]-self.rr[1]
        else:
            print('wrong move_side in pair')
        disp = np.array(disp,dtype =np.float32)

        # print('generating 3d point cloud...',)
        h, w = imgL.shape[:2]
        f = 0.5*w                       # focal length
        Q = np.float32([[1, 0, 0, -0.5*w],
                        [0, -1, 0,  0.5*h], # turn points 180 deg around x-axis,
                        [0, 0, 0,     f], # so that y-axis looks up
                        [0, 0, 1,      0]])
        points = cv.reprojectImageTo3D(disp, Q)
        colors = points
        if disp.min()<0:
            if self.showstep:
                print('disp<0!')
            return data_map,[], self.neg_reward  # don't add any points
        else:
            mask = disp > disp.min() 
            out_points = points[mask]*self.camera_distacne
            out_colors = colors[mask]

            if self.throshold_near: # set in json
                # bugs
                if len(out_points) == 0 or (len(out_points) >= 1 and out_points[0][2] > self.throshold_near):
                    glass_tpye = [[2],[2]]
                    return data_map,[], self.neg_reward
                    # if out_points.shape[0]==2:
                    #     out_points[0,2] += np.random.normal(0,50)
                    #     out_points[1,2] += np.random.normal(0,50)
                else:
                    glass_tpye = [[1],[1]]

        camera_points = self.trans_camera2env(out_points, pitch, yaw)
        env_points = camera_points + camera_location
        
        if env_points.shape[0]%2 == 0:
            # self.write_ply(self.out_fn, env_points, glass_tpye)
            vert = np.hstack([env_points, glass_tpye]) # camera_points useles, could change to color
            if data_map.size==0:
                data_map = vert
            else:
                data_map = np.concatenate((data_map, vert))
            if self.showstep:
                print(env_points)
            return data_map,vert, -1
        else:
            print('odd number of points:',camera_points.shape, env_points.shape[0])
            return data_map,[], self.neg_reward

        # print('%s saved' % self.out_fn)
        # draw()

        # cv.imshow('left', imgL)
        # cv.imshow('disparity', disp)
        # cv.waitKey()

        # print('Done')
        


    def find_point(self, mask):
        points_list = np.argwhere(mask>0)
        max_x = points_list[:,0].max()
        min_x = points_list[:,0].min()
        max_y = points_list[:,1].max()
        min_y = points_list[:,1].min()

        return int((max_x+min_x)/2), int(min_y), int((max_x+min_x)/2), int(max_y), 

    def largestConnectComponent(self, bw_img):
        # 新版本中neighbors被移除，添加connectivity=1(4联通)  2(8联通)
        labeled_img, num = measure.label(bw_img, connectivity=1, background=0, return_num=True)
        max_label = 0
        max_num = 0
        for i in range(1, num+1):
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        mcr = (labeled_img == max_label)
        mcr = (mcr + 0)
        return mcr, max_num

    def write_ply(self, fn, verts, colors):
        verts = verts.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        verts = np.hstack([verts, colors])
        with open(fn, 'ab') as f:
            # f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

    def trans_camera2env(self, points, pitch=0, yaw=0):
        a =  pitch*np.pi/180.0 # pitch
        b =  yaw*np.pi/180.0 # yaw
        self.T = np.array([
            [-np.sin(b), np.sin(a)*np.cos(b), np.cos(a)*np.cos(b)],
            [np.cos(b), np.sin(a)*np.sin(b), np.cos(a)*np.sin(b)],
            [0, np.cos(a), np.sin(a)]
        ])
        R = np.dot(self.T, points.T).T
        return R

    def draw(self, data1):
        # 数据１
        # data1 = []
        # with open(self.out_fn, "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.strip().strip('[').strip(']').split(' ')
        #         line = list(map(float,line))
        #         data1.append(line)
        #     f.close()
        data1 = np.array(data1)
        # x1 = list(data1[:, 0])  
        # y1 = list(-data1[:, 1])  
        # z1 = list(data1[:, 2])  
        # color = data1[:,3:]/255.0

        data2 = []
        with open(self.trajectary, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().strip('[').strip(']').split(',')
                line = list(map(float,line[:-2]))
                data2.append(line)
            f.close()
        data2 = np.array(data2)
        x2 = list(data2[:, 2])  
        y2 = list(-data2[:, 3])  
        z2 = list(data2[:, 4])  
        

        # 绘制散点图
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot(x2,y2,z2,'-')
        # ax.scatter(x1, y1, z1, c='r', s=15, label='glass')
        for i in range(int(len(data1)/2)):
            i *= 2
            x = np.array([data1[i][0], data1[i+1][0]])
            y = -np.array([data1[i][1], data1[i+1][1]])
            z = np.array([data1[i][2], data1[i+1][2]])
            if np.linalg.norm(data1[i]-data1[i+1])>self.threshold_len_min and np.linalg.norm(data1[i]-data1[i+1])<self.threshold_len_max:
                ax.scatter(x, y, z, c='r', s=15)
                ax.plot(x,y,z)


        # 绘制图例
        ax.legend(loc='best')

        # 添加坐标轴(顺序是Z, Y, X)
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('-Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})

        # 展示
        # plt.show()
        plt.savefig(self.draw_path, dpi=750, bbox_inches = 'tight')
    
    def cleanout(self):
        with open(self.out_fn, "w+") as f:
            pass
        with open(self.trajectary, "w+") as f:
            pass
        f.close()
    
    def reward_map(self):
        data1 = []
        pixelscale3d = 7
        reward = 0
        with open(self.out_fn, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().strip('[').strip(']').split(' ')
                line = list(map(float,line))
                data1.append(line)
            f.close()
        pixelscale3d = 7
        data1 = np.array(data1)/pixelscale3d
        env_map = np.zeros((int((1890+930)/pixelscale3d), int((1260+1660)/pixelscale3d), int(300/pixelscale3d)))
        for i in range(int(len(data1)/2)):
            i *= 2
            x = np.array([data1[i][0], data1[i+1][0]]) + 1890/pixelscale3d
            y = -np.array([data1[i][1], data1[i+1][1]]) + 1260/pixelscale3d
            z = np.array([data1[i][2], data1[i+1][2]])
            points = []
            line_map = np.zeros((int((1890+930)/pixelscale3d), int((1260+1660)/pixelscale3d), int(300/pixelscale3d)))
            points = np.array(points)
            line_lengh = np.linalg.norm(data1[i]-data1[i+1])
            if line_lengh>self.threshold_len_min/pixelscale3d \
                and line_lengh<self.threshold_len_max/pixelscale3d:
                point_num = int(line_lengh*1.2)
                points = np.expand_dims(np.linspace(x[0],x[1],point_num), axis=0)
                points = np.concatenate([points, np.expand_dims(np.linspace(y[0],y[1],point_num), axis=0)], axis=0)
                points = np.concatenate([points, np.expand_dims(np.linspace(z[0],z[1],point_num), axis=0)], axis=0)
                points =  np.around(points.transpose())
            for j in range(points.shape[0]):
                px, py, pz = points[j].astype('int')
                if px>=0 and px< int((1890+930)/pixelscale3d) \
                    and py>=0 and py< int((1260+1660)/pixelscale3d) \
                    and pz>=0 and pz< int(300/pixelscale3d) \
                    and line_map[pz][py][pz] == 0:
                    line_map[pz][py][pz] = 1
            env_map += line_map
        env_map = env_map.sum(axis=2)
        for t in range(1,int(env_map.max())+1):
            if t<4:
                reward += np.where(env_map==t, 1, 0).sum()*(1)
            else:
                reward += np.where(env_map==t, 1, 0).sum()*(1)
        return reward
                

    
if __name__ == '__main__':
    # imgL = cv.imread("/home/user/project/glass/data/datatwo/result/ckpt/validation/rgb/img_first_color.png")
    # imgR = cv.imread("/home/user/project/glass/data/datatwo/result/ckpt/validation/rgb/img_second_color.png")
    bm = Binarymask(-0.01)
    # bm.pair(imgL, imgR, [1,1,1], pitch=0, yaw=0)
    while(1):
        bm.draw()
        # env_map = bm.reward_map()



