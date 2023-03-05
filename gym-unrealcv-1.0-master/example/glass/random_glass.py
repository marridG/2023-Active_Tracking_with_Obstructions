import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
from example.utils import io_util
import subprocess
import os
import shutil

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealGlassLocate-GlassRoomglass-DiscreteColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    args = parser.parse_args()
    # bm = Binarymask()
    # bm.cleanout() 

    env = gym.make(args.env_id)
    env.env.env.glass_detect_dir = 'nouse'

    agent = RandomAgent(env.action_space)

    # if os.path.exists("/home/user/project/glass/data/datatwo/result/ckpt/validation/rgb/"):
    #         shutil.rmtree("/home/user/project/glass/data/datatwo/result/ckpt/validation/rgb/")
    # child_detection = subprocess.Popen('/home/user/project/glass/EBLNet/scripts/eval/eval_ueroom_R50_EBLNet.sh',stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    episode_count = 100
    reward = 0
    done = False
    TRA_DIR = './example/glass/out.csv'

    for i in range(episode_count):
        env.seed(i)
        ob = env.reset()
        count_step = 0
        t0 = time.time()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, info = env.step(action)
            # print('reward:', reward)
            io_util.save_trajectory(info, TRA_DIR, 0)
            count_step += 1
            if args.render:
                img = env.render(mode='rgb_array')
                #  img = img[..., ::-1]  # bgr->rgb
                cv2.imshow('show', img)
                cv2.waitKey(1000)
            if done:
                fps = count_step / (time.time() - t0)
                print ('Fps:' + str(fps))

                break

    # Close the env and write monitor result info to disk
    env.close()


