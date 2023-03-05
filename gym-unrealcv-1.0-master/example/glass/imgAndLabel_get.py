import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import math
import numpy as np
# import matplotlib.pylab as plt
import os

# class RandomAgent(object):
#     """The world's simplest agent!"""
#     def __init__(self, action_space):
#         self.action_space = action_space

#     def act(self, observation, reward, done):
#         return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealGlass-GlassRoomtargetbp2-DiscreteColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    args = parser.parse_args()
    env = gym.make(args.env_id)

    # agent = RandomAgent(env.action_space)

    data_path = '/home/user/project/glass/data/glassdata000/'
    if os.path.exists(data_path)==0:
        os.mkdir(data_path)


    episode_count = 10
    reward = 0
    done = False

    for obj in env.glasses: # change color to 0 0 255
        env.unrealcv.set_obj_color(obj, [0,0,255])


    for i in range(episode_count):
        print('episode_count:{}'.format(i))
        env.seed(i+1)
        ob = env.reset()
        count_step = 0
        t0 = time.time()
        for obj in env.glasses: # change color to 0 0 255
            glass_scale = np.random.uniform(1,2.5,3)
            env.unrealcv.set_obj_scale(obj, glass_scale)
        # time.sleep(2)
        for obj in env.glasses:
            # if glass_num==2:
            #     obj = 'bp_glass' + '2_4'
            # else:
            #     obj = 'bp_glass' + str(glass_num)
            print(obj)
            camera_length = np.random.uniform(200,400)
            camera_height = np.random.uniform(100,200)
            camera_down_flag = 0
            if camera_down_flag == 1:
                camera_down = np.arctan(camera_height/camera_length)*180/math.pi
            else:
                camera_down  = 0
            # angle = np.random.uniform(0,61)*random.choice([1, -1])
            for angle in range(0,360,30):
                obj_location = env.unrealcv.get_obj_location(obj)
                obj_rotation = env.unrealcv.get_obj_rotation(obj)
                obj_rotation[1] = obj_rotation[1] + angle
                # camera_pose = [obj_location]
                yaw_exp = (obj_rotation[1]) % 360
                # pitch_exp = (obj_rotation[2] + pitch) % 360
                delt_x = camera_length * math.cos(yaw_exp / 180.0 * math.pi)
                delt_y = camera_length * math.sin(yaw_exp / 180.0 * math.pi)
                location_exp = [obj_location[0] + delt_x, obj_location[1]+delt_y, obj_location[2]+camera_height]
                
                rotation_exp = obj_rotation
                rotation_exp.reverse()
                rotation_exp[2] = rotation_exp[2] + np.random.uniform(-10,20)-camera_down
                rotation_exp[1] = rotation_exp[1] + np.random.uniform(-30,30)
                rotation_exp[1] = rotation_exp[1]-180 if rotation_exp[1]>180 else rotation_exp[1]+180

                location_exp.extend(rotation_exp)
                camera_pose = location_exp
                # ob = env.getlabel(camera_pose)
                # self.unrealcv.set_location(self.cam_id, current_pose[:3])
                env.unrealcv.set_rotation(env.cam_id, camera_pose[3:])
                obj_location[2] = 1000 # set camera high enough to avoid Collision
                env.unrealcv.set_location(env.cam_id, obj_location)
                env.unrealcv.moveto(env.cam_id, [camera_pose[0],camera_pose[1],1000])
                env.unrealcv.moveto(env.cam_id, camera_pose[:3])
                location_now = env.unrealcv.get_location(env.cam_id)
                error = env.unrealcv.get_distance(location_now, camera_pose[:3], 3)
                if error < 10:
                    args.render = True
                else:
                    print('collision!')
                    args.render = False
                
                state = env.unrealcv.get_observation(env.cam_id, env.observation_type)
                
                if args.render:
                    # img_lit = env.render(mode='rgb_array')
                    #  img = img[..., ::-1]  # bgr->rgb
                    img_lit = env.unrealcv.get_observation(env.cam_id, 'Color')
                    img_obmask = env.unrealcv.get_observation(env.cam_id, 'Mask')
                    mask = env.unrealcv.get_mask(img_obmask, obj)
                    # img = np.hstack([img_lit, mask])
                    # if obj == 'bp_glass5' and angle == 0:
                    #     cv2.imshow(obj, mask)
                    #     flag = 1
                    cv2.imwrite(os.path.join(data_path, 'images/','e'+str(i)+'-'+obj+'angle'+str(angle)+'.jpg'), img_lit)
                    cv2.imwrite(os.path.join(data_path, 'masks/','e'+str(i)+'-'+obj+'angle'+str(angle)+'_mask.png'), mask)
                    cv2.waitKey(1)


        # while True:
        #     ob = env.reset()
        #     # action = agent.act(ob, reward, done)
        #     # ob, reward, done, _ = env.step(action)
        #     # count_step += 1
        #     args.render = True
        #     if args.render:
        #         img = env.render(mode='rgb_array')
        #         #  img = img[..., ::-1]  # bgr->rgb
        #         cv2.imshow('show', img)
        #         cv2.waitKey(1)
        #     if done:
        #         fps = count_step / (time.time() - t0)
        #         print ('Fps:' + str(fps))

    # Close the env and write monitor result info to disk
    env.close()