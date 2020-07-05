"""
@leofansq
"""
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

from env import ENV
from feature_encoder import FEATURE_ENCODER
from act import ACTOR

############################
####      仿真参数       ####
############################
T = 30
INTERVAL_A = 0.1
INTERVAL_ENV = 0.01

ACTION = [[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0], [0.0, -5.0]]


def main(EP, VIS, path, FAST):
    # 初始化仿真参数
    step_a = INTERVAL_A / INTERVAL_ENV
    # 初始化特征编码器 & 动作生成器
    encoder = FEATURE_ENCODER(ACTION)
    actor = ACTOR(encoder, ACTION, is_train=False)
    # 加载参数矩阵
    try:
        w = np.load(path)
        print ("Load {}".format(path))
        print ("-"*30)
    except:
        print ("Could not find {}".format(path))
        return 0
    # 实时可视化的初始化设置
    if VIS:
        plt.ion()
        plt.figure(figsize=(5, 5))
        plt.axis([0, 100, 0, 100])

    for ep in range(EP):
        sys.stdout.write("EP:{} ".format(ep+1))
        # 初始化环境
        # e = ENV(w=100, h=100, target=[85.0, 85.0], c_x=10.0, c_y=10.0, c_vx=0.0, c_vy=0.0)
        # e = ENV(w=100, h=100, c_vx=0.0, c_vy=0.0)
        e = ENV(w=100, h=100)

        # 可视化
        if VIS:
            plt.scatter(e.target[0], e.target[1], s=30, c='red')
        else:
            track_x = []
            track_y = []

        for t in range(int(T/INTERVAL_ENV)):

            if t%step_a == 0:
                a = actor.act([e.c.dx, e.c.dy, e.c.vx, e.c.vy], w)

            e.update(a)

            # 可视化
            if VIS and t%FAST==0:
                sys.stdout.write ("Ep:{}-{}  Vx:{:.2f}  Vy:{:.2f}  Action:{}        \r".format(ep, t+1, e.c.vx, e.c.vy, a))
                sys.stdout.flush()

                plt.scatter(e.c.x, e.c.y, s=10, c='blue', alpha=0.2) 
                plt.scatter(e.target[0], e.target[1], s=30, c='red')
                plt.pause(0.01)
            elif not VIS:
                track_x.append(e.c.x)
                track_y.append(e.c.y)
                str_out = "processing"
                if (t+1)%300==0:
                    sys.stdout.write(str_out[(t+1)//300-1])
                    sys.stdout.flush()
        
        print ("  Final_distance:{:.2f}                            ".format(-e.r))

        if VIS:
            plt.scatter(e.c.x, e.c.y, s=30, c='orange')
            plt.text(e.c.x, e.c.y-1, "EP{} Dist:{:.2f}".format(ep+1, -e.r))
            plt.pause(5)
        if not VIS:
            plt.scatter(track_x, track_y, s=5, c='blue', alpha=0.2) 
            plt.scatter(e.target[0], e.target[1], s=30, c='red')
            plt.scatter(track_x[-1], track_y[-1], s=30, c='orange')
            plt.text(e.c.x, e.c.y-1, "Dist:{:.2f}".format(-e.r))            
            plt.axis([0, 100, 0, 100])
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-Ep', default=5)
    parser.add_argument('-Vis', action='store_true')
    parser.add_argument('-Path', default="./p_w_test.npy")
    parser.add_argument('-Fast', default="30", help="n times fast for visualization")

    args = parser.parse_args()

    EP = int(args.Ep)
    VIS = args.Vis
    PATH = args.Path
    FAST = int(args.Fast)

    print ("-"*30)
    if VIS:
        print ("EPISODE:{}\nVIS:{}\nPATH:{}\n{} times fast\n".format(EP, VIS, PATH, FAST))
    else:
        print ("EPISODE:{}\nVIS:{}\nPATH:{}\n".format(EP, VIS, PATH))

    main(EP, VIS, PATH, FAST)

