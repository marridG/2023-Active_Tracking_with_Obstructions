import os
import random
import shutil
from tkinter.messagebox import NO
from turtle import right
import gym
import numpy as np
import math
import cv2
from gym import spaces
from gym_unrealcv.envs.glass import reward
from gym_unrealcv.envs.navigation import reset_point
from gym_unrealcv.envs.navigation.visualization import show_info
from gym_unrealcv.envs.utils import env_unreal, misc
from gym_unrealcv.envs.navigation.interaction import Navigation
from gym_unrealcv.envs.glass.interaction import Glass
from example.glass.binarymask import Binarymask
from example.utils import io_util
from collections import deque
'''
It is a general env for searching target object.

State : raw color image and depth (640x480) 
Action:  (linear velocity ,angle velocity , trigger) 
Done : Collision or get target place or False trigger three times.
Task: Learn to avoid obstacle and search for a target object in a room, 
      you can select the target name according to the Recommend object list in setting files
      
'''


class UnrealCvGlassLocate(gym.Env):
    ##
    def __init__(self,
                 setting_file,
                 category=0,
                 reset_type='random',  # testpoint, waypoint,
                 augment_env=None,  # texture, target, light
                 action_type='Discrete',  # 'Discrete', 'Continuous'
                 observation_type='Rgbd',  # 'color', 'depth', 'rgbd'
                 reward_type='bbox',  # distance, bbox, bbox_distance,
                 docker=False,
                 resolution=(720, 540)
                 ):
        self.glasspose=0
        setting = self.load_env_setting(setting_file)
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets'][category]
        self.trigger_th = setting['trigger_th']
        self.height = setting['height']
        self.pitch = setting['pitch']
        self.discrete_actions = setting['discrete_actions']
        self.continous_actions = setting['continous_actions']

        self.glasses = setting['glasses']
        self.objects_env = setting['objects_list']
        self.background_list = setting['backgrounds']
        self.light_list = setting['lights']
        self.data_map = []
        self.neg_reward = setting['reward_others']
        self.max_steps = setting['max_steps']
        self.showstep = setting['show_step']
        
        self.glass_detect_dir = 'test' # set different dir for different envs
        self.glass_data_root = '/data3/songmingyu/glass/data/ueroom/'
        self.data_path = '/data3/songmingyu/glass/data/ueroom/validation'
        self.glass_detect_path = None

        # using env mask
        self.env_mask = True

        self.reward_map = 0

        self.docker = docker
        self.textures_list = misc.get_textures(setting['imgs_dir'], self.docker)
        self.reset_type = reset_type
        self.augment_env = augment_env

        # mask to 3D
        self.bm = Binarymask(self.neg_reward)
        self.bm.throshold_near = setting['throshold_near']
        self.bm.cleanout() 



        # start unreal env
        self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
        env_ip, env_port = self.unreal.start(docker, resolution)
        # env_ip = '192.168.50.126'
        # env_port =9000# 54716
        # self.use_docker = docker

        # connect UnrealCV
        self.unrealcv = Glass(cam_id=self.cam_id,
                                   port=env_port,
                                   ip=env_ip,
                                #    targets=self.target_list,
                                   env=self.unreal.path2env,
                                   resolution=resolution)
        self.unrealcv.pitch = self.pitch

        #  define action
        self.action_type = action_type
        assert self.action_type == 'Discrete' or self.action_type == 'Continuous'
        if self.action_type == 'Discrete':
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        elif self.action_type == 'Continuous':
            self.action_space = spaces.Box(low=np.array(self.continous_actions['low']),
                                           high=np.array(self.continous_actions['high']))

        # define reward type
        # distance, bbox, bbox_distance,
        self.reward_type = reward_type
        self.reward_function = reward.Reward(setting)

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type == 'Color' or self.observation_type == 'Depth' or self.observation_type == 'Rgbd'
        # state channel 4 rgb+mask
        # self.observation_space = spaces.Box(low=0, high=255, shape=[540,720,4], dtype=np.uint8)# self.unrealcv.define_observation(self.cam_id, self.observation_type, 'direct')
        # state channel 1 mask
        self.observation_space = spaces.Box(low=0, high=255, shape=[540,720,3], dtype=np.uint8)# self.unrealcv.define_observation(self.cam_id, self.observation_type, 'direct')
        # state channel 2 camera&glass  area
        self.observation_space = spaces.Box(low=0, high=255, shape=\
            [int(self.reward_function.map_y/self.reward_function.pixelscale3d),int(self.reward_function.map_x/self.reward_function.pixelscale3d),2], dtype=np.uint8)

    
        
        

        # set start position
        self.trigger_count = 0
        current_pose = self.unrealcv.get_pose(self.cam_id)
        current_pose[2] = self.height
        self.unrealcv.set_location(self.cam_id, current_pose[:3])

        self.count_steps = 0

        self.targets_pos = self.unrealcv.build_pose_dic(self.target_list)

        # for reset point generation and selection
        self.reset_module = reset_point.ResetPoint(setting, reset_type, current_pose)


    def step(self, action):
        info = dict(
            Collision=False,
            Done=False,
            Trigger=0.0,
            Reward=0.0,
            Action=action,
            Bbox=[],
            Pose=[],
            Trajectory=self.trajectory,
            Steps=self.count_steps,
            Target=[],
            Direction=None,
            Waypoints=self.reset_module.waypoints,
            Color=None,
            Depth=None,
        )
        # data_path = '/home/user/project/glass/data/datatwo/validation'

        action = np.squeeze(action)
        if self.showstep:
            print(action)
        velocity, angle, move_side = self.discrete_actions[action]
        move_side = np.random.randint(1,3)
        self.count_steps += 1
        info['Done'] = False
        

        # if move_side == 0:
        # take action
        info['Collision'] = self.unrealcv.move_2d(self.cam_id, angle, velocity)
        # reward = 0 if info['Collision']==False else self.neg_reward
        if info['Collision']==True:
            if self.showstep:
                print('collision!')
            info['Pose'] = self.unrealcv.get_pose(self.cam_id, 'hard')
            reward = self.neg_reward
            # get mask
            # mask1 = np.zeros([540,720,3])
            mask1 = self.unrealcv.get_observation(self.cam_id, 'Mask')
            mask1 = self.unrealcv.get_mask(mask1, self.glasses[0])
            mask1 = np.tile(mask1,(3,1,1)).transpose(1, 2, 0)  # (540,720) to (540,720,3)
        else:
            info['Pose'] = self.unrealcv.get_pose(self.cam_id, 'hard')
            # if move_side !=0: # need get mask
            camera_pose = info['Pose']
            # img_lit_first = self.unrealcv.get_observation(self.cam_id, 'Color')
            # img_root = os.path.join(self.data_path, 'images/', self.glass_detect_dir+'_'+str(self.count_steps)+'_')
            # cv2.imwrite(img_root + 'img_first.png', img_lit_first)
            if self.env_mask:
                img_mask_first = self.unrealcv.get_observation(self.cam_id, 'Mask')
                img_mask_first = self.unrealcv.get_mask(img_mask_first, self.glasses[0])
                # mask_root = os.path.join(self.data_path, 'masks/', self.glass_detect_dir+'_'+str(self.count_steps)+'_')
                # cv2.imwrite(mask_root + 'img_first.png', img_mask_first)
            movesuccess, camera_location = self.camera_move_side(self.cam_id, camera_pose, move_side)
            if movesuccess:
                # img_lit = self.unrealcv.get_observation(self.cam_id, 'Color')
                if move_side==1: #right
                    # cv2.imwrite(img_root +'img_right.jpg', img_lit)
                    if self.env_mask:
                        img_mask_right = self.unrealcv.get_observation(self.cam_id, 'Mask')
                        img_mask_right = self.unrealcv.get_mask(img_mask_right, self.glasses[0])
                        # mask_root = os.path.join(self.data_path, 'masks/', self.glass_detect_dir+'_'+str(self.count_steps)+'_')
                        # cv2.imwrite(mask_root + 'img_right.png', img_mask_right)
                elif move_side==2: # left
                    # cv2.imwrite(img_root +'img_left.jpg', img_lit)
                    if self.env_mask:
                        img_mask_left = self.unrealcv.get_observation(self.cam_id, 'Mask')
                        img_mask_left = self.unrealcv.get_mask(img_mask_left, self.glasses[0])
                        # mask_root = os.path.join(self.data_path, 'masks/', self.glass_detect_dir+'_'+str(self.count_steps)+'_')
                        # cv2.imwrite(mask_root + 'img_left.png', img_mask_left)
                else:
                    print('wrong move_side in step')
                # moveback
                self.unrealcv.moveto(self.cam_id, camera_pose[:3])
                # using EBL
                # mask1, mask2 = self.glass_detection(move_side, self.glass_detect_path)
                # using env mask
                if self.env_mask:
                    # mask1, mask2 = self.glass_detection(move_side, self.glass_detect_path)
                    if move_side==1: #right
                        mask1, mask2 = img_mask_first, img_mask_right
                    elif move_side==2: # left
                        mask1, mask2 = img_mask_left, img_mask_first
                    else:
                        print('wrong move_side in step')
                if mask1.max()==0 or mask2.max()==0:
                    if self.showstep:
                        print('black mask!')
                    reward = self.neg_reward
                else:
                    self.data_map,self.data_map_new, reward = self.bm.pair(self.data_map, mask1, mask2, move_side, camera_pose[:3], camera_pose[5], camera_pose[4]) # pitch yaw
                    if reward==-1:
                        reward_map_new = self.reward_function.reward_map_supervise(self.data_map_new)
                        # print('map_lenth:',len(self.data_map))
                        # print('sum_reward:',reward_map_new)
                        reward = reward_map_new - self.reward_map
                        print("{}: step:{} map_lenth:{}   sum_reward:{}   reward:{}".format(self.glass_detect_dir,self.count_steps,len(self.data_map), reward_map_new,reward))
                        self.reward_map = reward_map_new
                mask1 = np.tile(img_mask_first,(3,1,1)).transpose(1, 2, 0)  # (540,720) to (540,720,3)
                       
            else: # move fail
                info['Pose'] = self.unrealcv.get_pose(self.cam_id, 'hard')
                reward = self.neg_reward
                # mask1 = np.zeros([540,720,3])
                mask1 = self.unrealcv.get_observation(self.cam_id, 'Mask')
                mask1 = self.unrealcv.get_mask(mask1, self.glasses[0])
                mask1 = np.tile(mask1,(3,1,1)).transpose(1, 2, 0)  # (540,720) to (540,720,3)
        if self.glass_detect_dir == 'eval':#'worker-test':
            # if self.count_steps%100==0:
            if self.count_steps%(self.max_steps)==0:
               self.bm.draw(self.data_map)


        
        info['Reward'] = reward
        if self.count_steps >= self.max_steps:
            info['Done'] = True
        
    
        # update observation
        # state channel 4
        # state = self.unrealcv.get_observation(self.cam_id, self.observation_type)
        # state = np.concatenate([state,np.expand_dims(mask1[:,:,0],axis=2)],axis=2)
        # state channel 1

        # state = mask1
        # type_line = self.get_type_ob(info['Pose'])
        
        state = self.reward_function.map_c2(info['Pose'],self.glass_detect_dir == 'worker-test' or self.glass_detect_dir == 'nouse')
        # state = np.zeros([int(self.reward_function.map_y/self.reward_function.pixelscale3d),int(self.reward_function.map_x/self.reward_function.pixelscale3d),2])

        # info['Color'] = self.unrealcv.img_color
        # info['Depth'] = self.unrealcv.img_depth

        # save the trajectory
        # self.trajectory.append(info['Pose'][:6])
        self.trajectory.append(np.concatenate((info['Pose'][:3],np.array(info['Pose'][4:5])),axis=0))
        info['Trajectory'] = self.trajectory
        if info['Done'] and len(self.trajectory) > 5 and self.reset_type == 'waypoint':
            self.reset_module.update_waypoint(info['Trajectory'])

        return state, info['Reward'], info['Done'], info

    def reset(self):

        self.reward_map = 0
        self.reward_function.clean_map()

        # clean trajectory
        self.bm.cleanout()

        # set mask detection dir
        self.glass_detect_path = self.glass_data_root + self.glass_detect_dir + '/'

        # target appearance
        # if self.reset_type >= 2:
            # map_id = [2, 3, 6, 7, 9]
            # self.unrealcv.set_appearance(self.target_list[0], np.random.choice(map_id))
            # self.unrealcv.random_player_texture(self.target_list[0], self.textures_list, 3)

        # reset camera
        self.unrealcv.set_location(self.cam_id, [-510+random.randint(-200,200), -130+random.randint(-200,200), 200])
        self.unrealcv.set_rotation(self.cam_id, [0, +random.randint(-180,180), 0])

        # set mask color of glasses
        for g in range(len(self.glasses)):
            self.unrealcv.set_obj_color(self.glasses[g],[0,255,0])

        # move obstacle out of room
        for obst in range(len(self.objects_env)):
            self.unrealcv.set_obj_location(self.objects_env[obst],[-1200,1980,120])

        # move my_glass1,2,3,6
        # for g in (0,1,2,5):
        #     self.unrealcv.set_obj_location(self.glasses[g],[-1200,1980,120])
        # for g in (4,5):
        #     self.unrealcv.set_obj_location(self.glasses[g],[-1200,1980,120])
        self.unrealcv.set_obj_location(self.glasses[0],[-1600,300,120])
        self.unrealcv.set_obj_location(self.glasses[1],[700,300,120])
        self.unrealcv.set_obj_location(self.glasses[2],[-400,1400,120])
        self.unrealcv.set_obj_location(self.glasses[3],[-400,-1100,120])
        self.unrealcv.set_obj_rotation(self.glasses[2],[0,90,0])
        self.unrealcv.set_obj_rotation(self.glasses[3],[0,90,0])

        
        # random scale and save glass data
        label_glass = np.zeros((len(self.glasses),6)) # 6: location,location,location,yaw,scale[1],scale[2]
        for o, obj in enumerate(self.glasses): # change color to 0 0 255
            glass_scale = [1,1,1]
            glass_loc = [0,0,0]
            glass_scale[1] = np.random.uniform(1,2) # w
            glass_scale[2] = np.random.uniform(1,2) # h
            self.unrealcv.set_obj_scale(obj, glass_scale)
            # location = self.unrealcv.set_obj_location(obj, loc)
            loc = self.unrealcv.get_obj_location(obj)
            yaw = self.unrealcv.get_obj_rotation(obj)[1]
            label_glass[o] = [loc[0], loc[1], loc[2], yaw, glass_scale[1], glass_scale[2]]

        # label map
        self.reward_function.label_map = self.reward_function.get_label_map(label_glass)

        # light
        # if self.reset_type >= 3:
        self.unrealcv.random_lit(self.light_list)
        
        # texture
        # if self.reset_type >= 4:
        self.unrealcv.random_texture(self.background_list, self.textures_list)

        
        # state channel 4
        # state = self.unrealcv.get_observation(self.cam_id, self.observation_type)
        # state = np.concatenate([state,np.zeros([540,720,1])],axis=2)
        
        # state channel 1
        # state = np.expand_dims(np.zeros([540,720,1]),axis=0)
        # state = np.zeros([540,720,3])

        # state channel 2
        state = np.zeros([int(self.reward_function.map_y/self.reward_function.pixelscale3d),int(self.reward_function.map_x/self.reward_function.pixelscale3d),2])


        self.trajectory = []
        self.data_map = []
        self.data_map = np.array(self.data_map)
        # self.trajectory.append(current_pose)
        self.trigger_count = 0
        self.count_steps = 0
        # self.reward_function.dis2target_last, self.targetID_last = \
        #     self.select_target_by_distance(current_pose, self.targets_pos)
        return state

    def getlabel(self,current_pose):

        # double check the resetpoint, it is necessary for random reset type
        # collision = True
        # while collision:
        #     current_pose = self.reset_module.select_resetpoint()
        #     self.unrealcv.set_pose(self.cam_id, current_pose)
        #     collision = self.unrealcv.move_2d(self.cam_id, 0, 100)
        # self.unrealcv.set_pose(self.cam_id, current_pose)
        # current_pose = self.reset_module.select_resetpoint()
        # current_pose = self.glasspose
        #self.unrealcv.set_pose(self.cam_id, current_pose)
        self.unrealcv.set_location(self.cam_id, current_pose[:3])
        self.unrealcv.set_rotation(self.cam_id, current_pose[3:])


        state = self.unrealcv.get_observation(self.cam_id, self.observation_type)

        # self.trajectory = []
        # self.trajectory.append(current_pose)
        # self.trigger_count = 0
        # self.count_steps = 0
        # self.reward_function.dis2target_last, self.targetID_last = \
        #     self.select_target_by_distance(current_pose, self.targets_pos)
        return state

    def seed(self, seed=None):
        return seed

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        if mode == 'rgb_array':
            return self.unrealcv.img_color
        elif mode == 'Mask':
            return self.unrealcv.img_color

    def close(self):
        self.unreal.close()

    def _get_action_size(self):
        return len(self.action)

    def select_target_by_distance(self, current_pos, targets_pos):
        # find the nearest target, return distance and targetid
        target_id = list(self.targets_pos.keys())[0]
        distance_min = self.unrealcv.get_distance(targets_pos[target_id], current_pos, 2)
        for key, target_pos in targets_pos.items():
            distance = self.unrealcv.get_distance(target_pos, current_pos, 2)
            if distance < distance_min:
                target_id = key
                distance_min = distance
        return distance_min, target_id

    def get_direction(self, current_pose, target_pose):
        y_delt = target_pose[1] - current_pose[1]
        x_delt = target_pose[0] - current_pose[0]
        angle_now = np.arctan2(y_delt, x_delt)/np.pi*180-current_pose[4]
        if angle_now > 180:
            angle_now -= 360
        if angle_now < -180:
            angle_now += 360
        return angle_now

    def load_env_setting(self, filename):
        import gym_unrealcv
        gympath = os.path.dirname(gym_unrealcv.__file__)
        gympath = os.path.join(gympath, 'envs/setting', filename)
        f = open(gympath)
        filetype = os.path.splitext(filename)[1]
        if filetype == '.json':
            import json
            setting = json.load(f)
        else:
            print ('unknown type')

        return setting

    # get the expected location of camera: in front of the glass, set random distance(length) and camera height
    def camera_init(self, obj): 
        camera_length = np.random.uniform(200,300)
        camera_height = np.random.uniform(100,200)
        camera_down_flag = 0
        if camera_down_flag == 1: # camera look down to see glass all
            camera_down = np.arctan(camera_height/camera_length)*180/math.pi
        else:
            camera_down  = 0
        obj_location = self.unrealcv.get_obj_location(obj)
        obj_rotation = self.unrealcv.get_obj_rotation(obj) # pitch yaw roll

        obj_rotation[1] = obj_rotation[1] + np.random.uniform(-45,45)   # random angle to get img
        # camera_pose = [obj_location]
        yaw_exp = (obj_rotation[1]) % 360
        # pitch_exp = (obj_rotation[2] + pitch) % 360
        delt_x = camera_length * math.cos(yaw_exp / 180.0 * math.pi)
        delt_y = camera_length * math.sin(yaw_exp / 180.0 * math.pi)
        location_exp = [obj_location[0] + delt_x, obj_location[1]+delt_y, obj_location[2]+camera_height]
        
        rotation_exp = obj_rotation # pitch yaw roll
        rotation_exp.reverse() # roll yaw pitch
        # rotation_exp[2] = rotation_exp[2] + np.random.uniform(-10,20)-camera_down
        rotation_exp[2] = rotation_exp[2]
        rotation_exp[1] = rotation_exp[1] + np.random.uniform(-30,30)
        rotation_exp[1] = rotation_exp[1]-180 if rotation_exp[1]>180 else rotation_exp[1]+180
        
        location_exp.extend(rotation_exp)
        pose_exp = location_exp

        return obj_location, pose_exp

    def camera_move_side(self, cam_id, camera_pose, move_side):
        length = 100
        location_now =  camera_pose[:3]
        camera_rotation = camera_pose[3:]
        yaw_exp = (camera_rotation[1]) % 360
        if move_side == 1: # move right
            delt_x = -length * math.sin(yaw_exp / 180.0 * math.pi)
            delt_y = length * math.cos(yaw_exp / 180.0 * math.pi)
        elif move_side == 2: # move left
            delt_x = length * math.sin(yaw_exp / 180.0 * math.pi)
            delt_y = -length * math.cos(yaw_exp / 180.0 * math.pi)
        else:
            print('wrong move_side in camera_move_side')
        
        location_exp = [location_now[0] + delt_x, location_now[1]+delt_y, location_now[2]]

        self.unrealcv.moveto(cam_id, location_exp)
        location_now = self.unrealcv.get_location(cam_id)
        error = self.unrealcv.get_distance(location_now, location_exp, 2)
        if error < 10:
            return True, location_now
        else:
            if self.showstep:
                print('move side collision!')
            return False, location_now

    def movetop(self, obj_location, camera_pose):
        self.unrealcv.set_rotation(self.cam_id, camera_pose[3:])
        obj_location[2] = 1000 # set camera high enough to avoid Collision
        self.unrealcv.set_location(self.cam_id, obj_location)
        self.unrealcv.moveto(self.cam_id, [camera_pose[0],camera_pose[1],1000])
        self.unrealcv.moveto(self.cam_id, camera_pose[:3])
        location_now = self.unrealcv.get_location(self.cam_id)
        error = self.unrealcv.get_distance(location_now, camera_pose[:3], 3)
        if error > 10:
            if self.showstep:
                print('collision!')
            return False
        else:
            return True

    def get_mask_point(self, mask):
        mask = np.array(mask)[:, :, :3].mean(-1)
        points_list = np.argwhere(mask>0)
        
        max_x = points_list[:,0].max()
        min_x = points_list[:,0].min()
        max_y = points_list[:,1].max()
        min_y = points_list[:,1].min()
        return int((max_x+min_x)/2), int((max_y+min_y)/2)
    
    def get_mask_points(self, mask):
        mask = np.array(mask)[:, :, :3].mean(-1)
        points_list = np.argwhere(mask>0)
        random.shuffle(points_list)

        return points_list[:,0], points_list[:, 1]

    def camera_depth2obj(self, camera_pose, camera_depth):
        camera_location = camera_pose[0:3]
        camera_rotation = camera_pose[3:] 
        camera_rotation.reverse() # pitch yaw roll
        pitch_exp = ((camera_rotation[0]) % 360) / 180.0 * math.pi
        yaw_exp = (camera_rotation[1]) % 360 / 180.0 * math.pi

        delt_z = camera_depth * math.sin(pitch_exp)
        delt_x = camera_depth * math.cos(pitch_exp)* math.cos(yaw_exp)
        delt_y = camera_depth * math.cos(pitch_exp)* math.sin(yaw_exp)

        location_exp = [camera_location[0] + delt_x, camera_location[1]+delt_y, camera_location[2]+delt_z]
        return location_exp

    # def glass_detection(self, img_lit):
    #      # os.system('sh /home/user/project/glass/EBLNet/scripts/eval/eval_ueroom_R50_EBLNet.sh')
    #     subprocess.check_call('sh /home/user/project/glass/EBLNet/scripts/eval/eval_ueroom_R50_EBLNet.sh',shell=True)
    #     mask = cv2.imread("/home/user/project/glass/dataone/result/ckpt/validation/rgb/img_color.png")
    #     mask = cv2.resize(mask,(720,540))
    #     return mask

    def glass_detection(self, move_side, glass_path): #/home/user/project/glass/data/datatwo/
        # glass_path = '/home/user/project/glass/data/datatwo/'
        # if os.path.exists(glass_path + "result/ckpt/validation/rgb/"):
        #     shutil.rmtree(glass_path + "result/ckpt/validation/rgb/")
        # subprocess.check_call('/home/user/project/glass/EBLNet/scripts/eval/eval_ueroom_R50_EBLNet.sh',stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            if self.env_mask:
                first_path = os.path.join(self.data_path, 'masks/', self.glass_detect_dir+'_'+str(self.count_steps))+'_'+'img_first.png'
                left_path = os.path.join(self.data_path, 'masks/', self.glass_detect_dir+'_'+str(self.count_steps))+'_'+'img_left.png'
                right_path = os.path.join(self.data_path, 'masks/', self.glass_detect_dir+'_'+str(self.count_steps))+'_'+'img_right.png'
            else:
                # 待修改
                first_path = self.glass_data_root + 'result/ckpt/validation/rgb/'+self.glass_detect_dir+'_'+str(self.count_steps)+'_'+'img_first_color.png'
                left_path = self.glass_data_root + 'result/ckpt/validation/rgb/'+self.glass_detect_dir+'_'+str(self.count_steps)+'_'+'img_left_color.png'
                right_path = self.glass_data_root + 'result/ckpt/validation/rgb/'+self.glass_detect_dir+'_'+str(self.count_steps)+'_'+'img_right_color.png'
            if os.path.exists(left_path) or os.path.exists(right_path) and os.path.exists(first_path): # which mean detect finish
                if os.path.exists(left_path):
                    mask1 = cv2.imread(left_path)
                    mask2 = cv2.imread(first_path)
                    # shutil.rmtree("/home/user/project/glass/data/datatwo/result/ckpt/validation/rgb/")
                    if mask1 is not None and mask2 is not None:
                        os.remove(left_path)
                        os.remove(first_path)
                        break
                else:
                    mask1 = cv2.imread(first_path)
                    mask2 = cv2.imread(right_path)
                    # shutil.rmtree("/home/user/project/glass/data/datatwo/result/ckpt/validation/rgb/")
                    if mask1 is not None and mask2 is not None:
                        os.remove(first_path)
                        os.remove(right_path)
                        break
        mask1 = cv2.resize(mask1,(720,540))
        mask2 = cv2.resize(mask2,(720,540))
        if self.showstep:
            print('glass detection finish!')

        return mask1, mask2
    
    
    def save_points_depth(self, obj_location, camera_pose, mask):
        output_list = []
        points_x, points_y = self.get_mask_points(mask)
        #camera_depth = 1 / img_depth[points_x][points_y][0] 
        delt_ps, delt_ys = np.arctan(np.array((540/2-points_x, points_y-720/2))/360)*180/np.pi
        for i in range(delt_ps.shape[0]):
            if i%500 == 0:
                camera_change = camera_pose.copy()
                camera_change[5] += delt_ps[i]
                camera_change[4] += delt_ys[i]
                self.unrealcv.set_rotation(self.cam_id, camera_change[3:])
                img_depth = self.unrealcv.get_observation(self.cam_id, 'Depth')
                camera_depth_test = 1/img_depth[270][360][0]
                
                # test
                points_searched = self.camera_depth2obj(camera_change, camera_depth_test)
                print(f'the center point of {obj_location} is {points_searched}')
                output_list.append(f'the center point of {obj_location} is {points_searched}')
                with open('./example/glass/points.txt', "a") as f:
                    f.write(str(points_searched)+'\n')
                f.close()
        print(len(output_list))

    def get_type_ob(self, pose):
        type_line = np.zeros(int(360/20))
        camera_angle = pose[4]
        camera_pose = np.array(pose[:2])/self.reward_function.pixelscale3d
        camera_pose[0] -= self.reward_function.map_scale[0]/self.reward_function.pixelscale3d
        camera_pose[1] -= self.reward_function.map_scale[2]/self.reward_function.pixelscale3d
        if camera_angle<0:
            camera_angle +=360
        if len(self.reward_function.glass_type_map)!=0:
            points = np.argwhere(self.reward_function.glass_type_map!=0)
            for point in points:
                disp = point-camera_pose
                if np.linalg.norm(disp,keepdims=False)> self.bm.throshold_near/self.reward_function.pixelscale3d:
                    if disp[0] > 0:
                        angle = np.arctan(disp[1]/disp[0])
                        if angle<0:
                            angle += 2*np.pi
                    else:
                        angle = np.pi + np.arctan(disp[1]/disp[0])
                    angle = angle*180/np.pi - camera_angle
                    if angle<0:
                        angle +=360
                    if type_line[int(angle/20-0.001)] != 1:    
                        type_line[int(angle/20-0.001)] = self.reward_function.glass_type_map[point[0],point[1]]
        # change type_line: (18)->(720)->(540,720)->(540,720,3)->(0-255)
        type_line = np.repeat(np.array([type_line]),int(720/18), axis=1) 
        type_line = np.repeat(type_line,540,axis=0)
        type_line = np.tile(type_line,(3,1,1)).transpose(1, 2, 0)
        type_line = ((type_line)*(1/2)*255).astype('uint8')
        # print(type_line,camera_angle)
        return type_line

