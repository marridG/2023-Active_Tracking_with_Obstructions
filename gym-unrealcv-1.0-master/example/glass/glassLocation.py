import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
# import matplotlib.pylab as plt
import os
from binarymask import Binarymask


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealGlassLocate-GlassRoomtargetbp2-DiscreteColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    args = parser.parse_args()
    # bm = Binarymask()
    # bm.cleanout() 


    env = gym.make(args.env_id)

    agent = RandomAgent(env.action_space)

    data_path = '/home/user/project/glass/data/datatwo/validation'
    episode_count = 25
    reward = 0
    done = False
    

    # for obj in env.glasses: # change color to 0 0 255
    #     env.unrealcv.set_obj_color(obj, [0,0,255])

    for i in range(episode_count):
        env.seed(i)
        ob = env.reset()
        count_step = 0
        t0 = time.time()
        

        # base depth
        # for obj in env.glasses:
        #     obj_location, camera_pose = env.camera_init(obj)
        #     env.movetop(obj_location, camera_pose)
        #     img_lit = env.unrealcv.get_observation(env.cam_id, 'Color')
        #     cv2.imwrite(os.path.join(data_path, 'images/','img.jpg'), img_lit)
        #     img_depth = env.unrealcv.get_observation(env.cam_id, 'Depth') #(540,720,1)
        #     mask = env.glass_detection(img_lit)
        #     env.save_points_depth(obj_location, camera_pose, mask)

        # base stereo
        for obj in env.glasses:
            obj_location, camera_pose = env.camera_init(obj)
            movesuccess = env.movetop(obj_location, camera_pose)
            if movesuccess == False:
                continue
            img_lit = env.unrealcv.get_observation(env.cam_id, 'Color')
            cv2.imwrite(os.path.join(data_path, 'images/','img_first.jpg'), img_lit)
            movesuccess, camera_location = env.camera_move_side(env.cam_id, camera_pose)
            if movesuccess:
                img_lit = env.unrealcv.get_observation(env.cam_id, 'Color')
                cv2.imwrite(os.path.join(data_path, 'images/','img_second.jpg'), img_lit)
                mask1, mask2 = env.glass_detection()
                if mask1.max()==0 or mask2.max()==0:
                    continue
                bm.pair(mask1, mask2, camera_pose[:3], camera_pose[5], camera_pose[4]) # pitch yaw

                # cv2.imwrite(os.path.join(data_path, 'masks/','mask_first.jpg'), mask1)
                # cv2.imwrite(os.path.join(data_path, 'masks/','mask_second.jpg'), mask2)

    bm.draw()
    env.close()

            # img_obmask = env.unrealcv.get_observation(env.cam_id, 'Mask')
            # mask = env.unrealcv.get_mask(img_obmask, obj)

            # action = agent.act(ob, reward, done)
            # ob, reward, done, _ = env.step(action)
            # count_step += 1
            # if args.render:
            #     img = env.render(mode='rgb_array')
            #     #  img = img[..., ::-1]  # bgr->rgb
            #     cv2.imshow('show', img)
            #     cv2.waitKey(1000)
            # if done:
            #     fps = count_step / (time.time() - t0)
            #     print ('Fps:' + str(fps))
            #     break
    