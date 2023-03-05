import numpy as np
import math
import cv2

class Reward():
    '''
    define different type reward function
    '''

    def __init__(self, setting):

        # self.dis_exp = setting['exp_distance']
        # self.dis_max = setting['max_distance']
        # self.dis_min = setting['min_distance']
        # self.angle_max = setting['max_direction']
        # self.angle_half = self.angle_max/2.0
        self.r_target = 0
        self.r_tracker = 0
        # self.dis2target = self.dis_exp
        self.angle2target = 0
        self.out_fn = setting["out_fn"]
        self.map_scale = setting["reset_area"]
        self.map_x = self.map_scale[1] - self.map_scale[0]
        self.map_y = self.map_scale[3] - self.map_scale[2]
        self.map_h = self.map_scale[5] - self.map_scale[4]
        self.map = np.zeros((self.map_x, self.map_y, self.map_h))
        self.threshold_len_min = setting["threshold_len_min"]
        self.threshold_len_max = setting["threshold_len_max"]
        self.pixelscale3d = setting["pixelscale3d"]
        self.threshold_pos = setting["threshold_pos"]
        self.reward_negative = setting["reward_negative"]
        self.label_map = []
        self.lines_map = np.zeros((int(self.map_x/self.pixelscale3d), int(self.map_y/self.pixelscale3d), int(self.map_h/self.pixelscale3d)))
        self.map_area = np.zeros_like(self.lines_map[:,:,0])
        self.map_area = self.map_area.T.copy()
        # self.glass_type_map = np.zeros_like(self.lines_map[:,:,0,1])

    def reward_map(self, data1):
        reward = 0
        # with open(self.out_fn, "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.strip().strip('[').strip(']').split(' ')
        #         line = list(map(float,line))
        #         data1.append(line)
        #     f.close()
        data1 = np.array(data1)/self.pixelscale3d
        env_map = np.zeros((int(self.map_x/self.pixelscale3d), int(self.map_y/self.pixelscale3d), int(self.map_h/self.pixelscale3d)))
        for i in range(int(len(data1)/2)):
            i *= 2
            x = np.array([data1[i][0], data1[i+1][0]]) - self.map_scale[0]/self.pixelscale3d
            y = -np.array([data1[i][1], data1[i+1][1]]) - self.map_scale[2]/self.pixelscale3d
            z = np.array([data1[i][2], data1[i+1][2]])
            points = []
            line_map = np.zeros((int(self.map_x/self.pixelscale3d), int(self.map_y/self.pixelscale3d), int(self.map_h/self.pixelscale3d)))
            points = np.array(points)
            line_lengh = np.linalg.norm(data1[i]-data1[i+1])
            if line_lengh>self.threshold_len_min/self.pixelscale3d \
                and line_lengh<self.threshold_len_max/self.pixelscale3d:
                point_num = int(line_lengh*1.2) # 1.2 can make sure every 3Dpixel has at least one point in it
                points = np.expand_dims(np.linspace(x[0],x[1],point_num), axis=0)
                points = np.concatenate([points, np.expand_dims(np.linspace(y[0],y[1],point_num), axis=0)], axis=0)
                points = np.concatenate([points, np.expand_dims(np.linspace(z[0],z[1],point_num), axis=0)], axis=0)
                points =  np.around(points.transpose())
            for j in range(points.shape[0]):
                px, py, pz = points[j].astype('int')
                if px>=0 and px< int(self.map_x/self.pixelscale3d) \
                    and py>=0 and py< int(self.map_y/self.pixelscale3d) \
                    and pz>=0 and pz< int(self.map_h/self.pixelscale3d) \
                    and line_map[px][py][pz] == 0:
                    line_map[px][py][pz] = 1
            env_map += line_map
        env_map = env_map.sum(axis=2)
        threshold = env_map.max()/self.threshold_pos
        for t in range(1,int(env_map.max())+1):
            if t<threshold:
                reward += np.where(env_map==t, 1, 0).sum()*(-0.1)
            else:
                reward += np.where(env_map==t, 1, 0).sum()*(1)
        reward = max(reward, -100)
        return reward

    '''
    get the map of detected glasses
    each line represents a glass
    save the distance of glass, useless now
    '''
    def reward_map_supervise_d(self, data1):
        reward = 0
        # with open(self.out_fn, "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.strip().strip('[').strip(']').split(' ')
        #         line = list(map(float,line))
        #         data1.append(line)
        #     f.close()
        glass_type = data1[0][3]
        data1 = np.array(data1)/self.pixelscale3d
        # env_map = np.zeros((int(self.map_x/self.pixelscale3d), int(self.map_y/self.pixelscale3d), int(self.map_h/self.pixelscale3d)))
        for i in range(int(len(data1)/2)):
            i *= 2
            x = np.array([data1[i][0], data1[i+1][0]]) - self.map_scale[0]/self.pixelscale3d
            y = np.array([data1[i][1], data1[i+1][1]]) - self.map_scale[2]/self.pixelscale3d
            z = np.array([data1[i][2], data1[i+1][2]])
            points = [] # points in this line
            line_map = np.zeros((int(self.map_x/self.pixelscale3d), int(self.map_y/self.pixelscale3d), int(self.map_h/self.pixelscale3d),2))
            points = np.array(points)
            line_lengh = np.linalg.norm(data1[i]-data1[i+1])
            if line_lengh>self.threshold_len_min/self.pixelscale3d \
                and line_lengh<self.threshold_len_max/self.pixelscale3d: # leave points of normal length
                point_num = int(line_lengh*1.2) # 1.2 can make sure every 3Dpixel has at least one point in it
                points = np.expand_dims(np.linspace(x[0],x[1],point_num), axis=0)
                points = np.concatenate([points, np.expand_dims(np.linspace(y[0],y[1],point_num), axis=0)], axis=0)
                points = np.concatenate([points, np.expand_dims(np.linspace(z[0],z[1],point_num), axis=0)], axis=0)
                points =  np.around(points.transpose())
            for j in range(points.shape[0]):
                px, py, pz = points[j].astype('int')
                if px>=0 and px< int(self.map_x/self.pixelscale3d) \
                    and py>=0 and py< int(self.map_y/self.pixelscale3d) \
                    and pz>=0 and pz< int(self.map_h/self.pixelscale3d) \
                    and line_map[px][py][pz][0] == 0:
                    line_map[px][py][pz][0] = 1
                    line_map[px][py][pz][1] = glass_type
            self.lines_map[:,:,:,0] += line_map[:,:,:,0]
            # for p in line_map[:,:,:,1] == 1:
            self.lines_map[:,:,:,1][line_map[:,:,:,1] == 1] = line_map[:,:,:,1][line_map[:,:,:,1] == 1]
            # for p in self.lines_map[:,:,:,1] == 0:
            self.lines_map[:,:,:,1][self.lines_map[:,:,:,1] == 0] = line_map[:,:,:,1][self.lines_map[:,:,:,1] == 0]
        env_map = self.lines_map[:,:,:,0].sum(axis=2)
        # self.glass_type_map = np.zeros_like(self.lines_map[:,:,0,1])
        point_type_1 = np.argwhere(self.lines_map[:,:,:,1]==1)[:,:2]
        for p in point_type_1:
            self.glass_type_map[p[0],p[1]] = 1
        point_type_2 = np.argwhere(self.lines_map[:,:,:,1]==2)[:,:2]
        for p in point_type_2:
            if self.glass_type_map[p[0],p[1]] != 1:
                self.glass_type_map[p[0],p[1]] = 2
        env_map = env_map * np.float32(self.glass_type_map==1) # only points within threshold count
        threshold = 1 #env_map.max()/self.threshold_pos
        label_p = self.label_map * (env_map>=threshold) # and  
        label_n = (self.label_map==0) * (env_map<threshold) * (env_map>0) # and
        reward = (label_p!=0).sum()/(self.label_map!=0).sum()- self.reward_negative*(label_n==True).sum()
        # reward = max(reward, -100)
        return reward

    '''
    get the map of detected glasses
    each line represents a glass
    '''
    def reward_map_supervise(self, data1):
        reward = 0
        # with open(self.out_fn, "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.strip().strip('[').strip(']').split(' ')
        #         line = list(map(float,line))
        #         data1.append(line)
        #     f.close()
        data1 = np.array(data1)/self.pixelscale3d
        # env_map = np.zeros((int(self.map_x/self.pixelscale3d), int(self.map_y/self.pixelscale3d), int(self.map_h/self.pixelscale3d)))
        for i in range(int(len(data1)/2)):
            i *= 2
            x = np.array([data1[i][0], data1[i+1][0]]) - self.map_scale[0]/self.pixelscale3d
            y = np.array([data1[i][1], data1[i+1][1]]) - self.map_scale[2]/self.pixelscale3d
            z = np.array([data1[i][2], data1[i+1][2]])
            points = [] # points in this line
            line_map = np.zeros((int(self.map_x/self.pixelscale3d), int(self.map_y/self.pixelscale3d), int(self.map_h/self.pixelscale3d)))
            points = np.array(points)
            line_lengh = np.linalg.norm(data1[i]-data1[i+1])
            if line_lengh>self.threshold_len_min/self.pixelscale3d \
                and line_lengh<self.threshold_len_max/self.pixelscale3d: # leave points of normal length
                point_num = int(line_lengh*1.2) # 1.2 can make sure every 3Dpixel has at least one point in it
                points = np.expand_dims(np.linspace(x[0],x[1],point_num), axis=0)
                points = np.concatenate([points, np.expand_dims(np.linspace(y[0],y[1],point_num), axis=0)], axis=0)
                points = np.concatenate([points, np.expand_dims(np.linspace(z[0],z[1],point_num), axis=0)], axis=0)
                points =  np.around(points.transpose())
            for j in range(points.shape[0]):
                px, py, pz = points[j].astype('int')
                if px>=0 and px< int(self.map_x/self.pixelscale3d) \
                    and py>=0 and py< int(self.map_y/self.pixelscale3d) \
                    and pz>=0 and pz< int(self.map_h/self.pixelscale3d) \
                    and line_map[px][py][pz] == 0:
                    line_map[px][py][pz] = 1
            # the cells in lines_map near line_map shuold decrease...
            self.lines_map += line_map
        env_map = self.lines_map.sum(axis=2)
        threshold = 3 #env_map.max()/self.threshold_pos
        label_p = self.label_map * (env_map>=threshold) # and  
        label_n = (self.label_map==0) * (env_map<threshold) * (env_map>0) # and
        reward = (label_p!=0).sum()/(self.label_map!=0).sum()# - 0.01*(label_n==True).sum()
        # reward = max(reward, -100)
        return reward


    '''
    get the map of label glesses
    '''
    def get_label_map(self, label_glass):
        data_label = np.empty((0,3))
        for glass in label_glass:
            yaw = glass[3]
            halfw = 50*glass[4]
            halfh = 75*glass[5]
            point1 = np.array([[glass[0]- halfw*math.sin(yaw / 180.0 * math.pi), glass[1]+ halfw*math.cos(yaw / 180.0 * math.pi), glass[2]]])
            point2 = np.array([[glass[0]+ halfw*math.sin(yaw / 180.0 * math.pi), glass[1]- halfw*math.cos(yaw / 180.0 * math.pi), glass[2]]])
            data_label = np.append(data_label,point1,axis=0)
            data_label = np.append(data_label,point2,axis=0)
        data_label = np.array(data_label)
        data_label = np.array(data_label)/self.pixelscale3d
        env_map = np.zeros((int(self.map_x/self.pixelscale3d), int(self.map_y/self.pixelscale3d), int(self.map_h/self.pixelscale3d)))
        for i in range(int(len(data_label)/2)):
            i *= 2
            x = np.array([data_label[i][0], data_label[i+1][0]]) - self.map_scale[0]/self.pixelscale3d
            y = np.array([data_label[i][1], data_label[i+1][1]]) - self.map_scale[2]/self.pixelscale3d
            z = np.array([data_label[i][2], data_label[i+1][2]])
            points = [] # points in this line
            line_map = np.zeros((int(self.map_x/self.pixelscale3d), int(self.map_y/self.pixelscale3d), int(self.map_h/self.pixelscale3d)))
            points = np.array(points)
            line_lengh = np.linalg.norm(data_label[i]-data_label[i+1])
            if line_lengh>self.threshold_len_min/self.pixelscale3d \
                and line_lengh<self.threshold_len_max/self.pixelscale3d: # leave points of normal length
                point_num = int(line_lengh*1.2) # 1.2 can make sure every 3Dpixel has at least one point in it
                points = np.expand_dims(np.linspace(x[0],x[1],point_num), axis=0)
                points = np.concatenate([points, np.expand_dims(np.linspace(y[0],y[1],point_num), axis=0)], axis=0)
                points = np.concatenate([points, np.expand_dims(np.linspace(z[0],z[1],point_num), axis=0)], axis=0)
                points =  np.around(points.transpose())
            for j in range(points.shape[0]):
                px, py, pz = points[j].astype('int')
                if px>=0 and px< int(self.map_x/self.pixelscale3d) \
                    and py>=0 and py< int(self.map_y/self.pixelscale3d) \
                    and pz>=0 and pz< int(self.map_h/self.pixelscale3d) \
                    and line_map[px][py][pz] == 0:
                    line_map[px][py][pz] = 1
            env_map += line_map
        env_map = env_map.sum(axis=2)
        return env_map

    '''
    reset the maps in reset()
    '''
    def clean_map(self):
        self.lines_map = np.zeros((int(self.map_x/self.pixelscale3d), int(self.map_y/self.pixelscale3d), int(self.map_h/self.pixelscale3d)))
        self.map_area = np.zeros_like(self.lines_map[:,:,0])
        self.map_area = self.map_area.T.copy()
        # self.glass_type_map = np.zeros_like(self.lines_map[:,:,0,1])


    '''
    get two channels state from map
    map_glass_camera: show the locationg of glasses and camera
    self.map_area: show the detected area using camera locationg and angle
    opencv coordinate is different from numpy!

    wait to add occludsion
    '''
    def map_c2(self, pose, show_map):
        show_map = False
        camera_angle = pose[4]/180*np.pi
        # print(pose[4]) 
        camera_pose = np.array(pose[:2])/self.pixelscale3d
        camera_pose[0] -= self.map_scale[0]/self.pixelscale3d
        camera_pose[1] -= self.map_scale[2]/self.pixelscale3d

        # glass and camera arrow
        map_glass_camera =  self.lines_map.sum(axis=2)
        map_glass_camera = map_glass_camera.T.copy() # lines_map(x-down y-right) to map_glass_camera(x-right,y-down)
        map_glass_camera = map_glass_camera.astype('uint8')*255
        arrow_begin = [int(camera_pose[0]),int(camera_pose[1])]
        # need change
        arrow_length = int(210/self.pixelscale3d)
        arrow_width = int(40/self.pixelscale3d)
        arrow_end = [int(camera_pose[0]+arrow_length*np.cos(camera_angle)), int(camera_pose[1]+arrow_length*np.sin(camera_angle))]
        cv2.arrowedLine(map_glass_camera, arrow_begin,arrow_end,255,arrow_width,8,0,0.5)
        # explored aera
        aera_now = np.zeros_like(map_glass_camera)
        cv2.ellipse(self.map_area,arrow_begin,(30,30), int(-45+pose[4]), 0,80, (255,255,255),-1) # 500 should big enough

        # cv2.circle(map_glass_camera,(300,100), 40, (255,255,255),1)

        # show the maps
        if show_map:
            two_map = np.hstack([map_glass_camera, self.map_area])
            cv2.imshow('map', two_map)
            cv2.waitKey(1)

        return np.stack([map_glass_camera, self.map_area]).transpose(1,2,0).astype(np.uint8)
