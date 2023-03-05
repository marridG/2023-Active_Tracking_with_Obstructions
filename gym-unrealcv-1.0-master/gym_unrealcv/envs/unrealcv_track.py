import time
import gym
import numpy as np
from gym import spaces
from gym_unrealcv.envs.tracking import reward
from gym_unrealcv.envs.utils import env_unreal, misc
from gym_unrealcv.envs.tracking.interaction import Tracking

''' 
It is an env for active object tracking.

State : raw color image and depth
Action:  (linear velocity ,angle velocity) 
Done : the relative distance or angle to target is larger than the threshold.
Task: Learn to follow the target object(moving person) in the scene
'''


class UnrealCvTrack(gym.Env):
    def __init__(self,
                 setting_file,
                 category=0,
                 reset_type=0,
                 action_type='Discrete',  # 'discrete', 'continuous'
                 observation_type='Color',  # 'color', 'depth', 'rgbd'
                 reward_type='distance',  # distance
                 docker=False,
                 resolution=(720, 540)
                 ):
        self.docker = docker
        self.reset_type = reset_type
        self.roll = 0

        setting = misc.load_env_setting(setting_file)
        self.env_name = setting['env_name']
        self.cam_id = setting['cam_id']
        self.target_list = setting['targets']
        self.discrete_actions = setting['discrete_actions']
        self.continous_actions = setting['continous_actions']
        self.max_distance = setting['max_distance']
        self.min_distance = setting['min_distance']
        self.max_direction = setting['max_direction']
        self.height = setting['height']
        self.pitch = setting['pitch']
        self.reset_area = setting['reset_area']
        self.background_list = setting['backgrounds']
        self.light_list = setting['lights']
        self.target_num = setting['target_num']
        self.exp_distance = setting['exp_distance']
        self.safe_start = setting['safe_start']

        self.textures_list = misc.get_textures(setting['imgs_dir'], self.docker)

        # start unreal env
        self.unreal = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
        env_ip, env_port = self.unreal.start(docker, resolution)

        # connect UnrealCV
        self.unrealcv = Tracking(cam_id=self.cam_id, port=env_port, ip=env_ip,
                                 env=self.unreal.path2env, resolution=resolution)

        # define action
        self.action_type = action_type
        assert self.action_type == 'Discrete' or self.action_type == 'Continuous'
        if self.action_type == 'Discrete':
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        elif self.action_type == 'Continuous':
            self.action_space = spaces.Box(low=np.array(self.continous_actions['low']),
                                           high=np.array(self.continous_actions['high']))

        # define observation space,
        # color, depth, rgbd,...
        self.observation_type = observation_type
        assert self.observation_type == 'Color' or self.observation_type == 'Depth' or self.observation_type == 'Rgbd'
        self.observation_space = self.unrealcv.define_observation(self.cam_id, self.observation_type, 'fast')

        # define reward type
        self.reward_type = reward_type
        self.reward_function = reward.Reward(setting)

        self.rendering = False
        self.unrealcv.start_walking(self.target_list[0])
        self.count_steps = 0
        self.count_close = 0
        self.unrealcv.pitch = self.pitch

    def step(self, action):
        info = dict(
            Collision=False,
            Done=False,
            Trigger=0.0,
            Reward=0.0,
            Action=action,
            Pose=[],
            Trajectory=self.trajectory,
            Steps=self.count_steps,
            Direction=None,
            Distance=None,
            Color=None,
            Depth=None,
        )

        # lookbzz get action 
        # action = np.squeeze(action)
        # if self.action_type == 'Discrete':
        #     (velocity, angle) = self.discrete_actions[action]
        # else:
        #     (velocity, angle) = action


        
        self.count_steps += 1

        # get target location
        target_pos = self.unrealcv.get_obj_location(self.target_list[0])
        camera_pos = self.unrealcv.get_pose(self.cam_id)

        # lookbzz move camera
        velocity = 0
        angle = -np.arctan((target_pos[0]-camera_pos[0])/(target_pos[1]-camera_pos[1]))
        yaw = angle/np.pi*180
        info['Collision'] = self.unrealcv.set_rotation(self.cam_id, [0,90+yaw,-15])

        # lookbzz move target when x==-1300 or x==600, turn around
        if abs(target_pos[0]+1300)<=30 or abs(target_pos[0]-600)<=30:
            self.unrealcv.set_move(self.target_list[0], 180, 200)
            time.sleep(1)
        else:
            self.unrealcv.set_move(self.target_list[0], 0, 200)

        

        # get reward
        # info['Pose'] = self.unrealcv.get_pose(self.cam_id)
        # self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
        # info['Direction'] = misc.get_direction(info['Pose'], self.target_pos)
        # info['Distance'] = self.unrealcv.get_distance(self.target_pos, info['Pose'], 2)
        # if 'distance' in self.reward_type:
        #     info['Reward'] = self.reward_function.reward_distance(info['Distance'], info['Direction'])

        # update observation
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type, 'fast')

        info['Color'] = self.unrealcv.img_color
        info['Depth'] = self.unrealcv.img_depth

        # done condition
        # if info['Distance'] > self.max_distance or info['Distance'] < self.min_distance \
        #         or abs(info['Direction']) > self.max_direction:
        #     self.count_close += 1
        # else:
        #     self.count_close = 0

        # if self.count_close > 10:
        #     info['Done'] = True

        # save the trajectory
        self.trajectory.append(info['Pose'])
        info['Trajectory'] = self.trajectory

        self.C_reward += info['Reward']
        return state, np.float(info['Reward']), info['Done'], info

    def reset(self, ):
        self.C_reward = 0
        self.count_close = 0
        self.unrealcv.start_walking(self.target_list[0])  # stop moving
        np.random.seed()

        # movement
        # if self.reset_type >= 1:
            # self.unrealcv.random_character(self.target_list[0])

        # target appearance
        if self.reset_type >= 2:
            map_id = [2, 3, 6, 7, 9]
            self.unrealcv.set_appearance(self.target_list[0], np.random.choice(map_id))
            self.unrealcv.random_player_texture(self.target_list[0], self.textures_list, 3)

        # light
        if self.reset_type >= 3:
            self.unrealcv.random_lit(self.light_list)

        # texture
        # if self.reset_type >= 4:
        self.unrealcv.random_texture(self.background_list, self.textures_list)

        # replace tracker and target, tracker should aim at target
        # self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
        # res = self.unrealcv.get_startpoint(self.target_pos, self.exp_distance, self.reset_area, self.height)
        # count = 0
        # while not res:
        #     count += 1
        #     self.unrealcv.reset_target(self.target_list[0])
        #     time.sleep(0.1)
        #     self.target_pos = self.unrealcv.get_obj_location(self.target_list[0])
        #     res = self.unrealcv.get_startpoint(self.target_pos, self.exp_distance, self.reset_area)
        # cam_pos_exp, yaw = res
        # cam_pos_exp[-1] = self.height

        # lookbzz set camera loaction and rotation, location:on the wall, location:(0,90,-15)
        cam_pos_exp = [-370, -1260, 600]
        self.unrealcv.set_location(self.cam_id, cam_pos_exp)
        self.roll, yaw, self.pitch = 0, 90,-15
        self.unrealcv.set_rotation(self.cam_id, [self.roll, yaw, self.pitch])
        current_pose = self.unrealcv.get_pose(self.cam_id, 'soft')

        # set camera location

        # get observation
        time.sleep(0.5)
        state = self.unrealcv.get_observation(self.cam_id, self.observation_type, 'fast')

        # save trajectory
        self.trajectory = []
        self.trajectory.append(current_pose)
        self.count_steps = 0

        # set target move  -1265,-13,226
        # location
        self.unrealcv.set_obj_location(self.target_list[0], [-1265,0,226])
        # how to move
        while True:
            if self.unrealcv.start_walking(self.target_list[0]):  # stop moving
                break

        return state

    def close(self):
        self.unreal.close()

    def render(self, mode='rgb_array', close=False):
        if close==True:
            self.unreal.close()
        return self.unrealcv.img_color

    def seed(self, seed=None):
        pass
