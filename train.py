"""
@leofansq
"""
import numpy as np 
import pickle
import argparse

from env import ENV
from feature_encoder import FEATURE_ENCODER
from act import ACTOR

############################
####      仿真参数       ####
############################

# 单次仿真时间/s
T = 30
# 动作仿真间隔/s
INTERVAL_A = 0.1
# 环境仿真间隔/s
INTERVAL_ENV = 0.01

ACTION = [[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0], [0.0, -5.0]]


def p_loop(EPISODE, GAMMA, LAMBDA, ALPHA, path):
    """
    训练函数
    """
    # 初始化w
    try:
        w = np.load(path)
        print ("Load {}".format(path))
        print ("-"*30)
    except:
        w = np.zeros((12*12*12*12*4, 1))
        print ("Initialize Value")
        print ("-"*30)

    # 初始化Feature_Encoder & Actor
    encoder = FEATURE_ENCODER(ACTION)
    actor = ACTOR(encoder, ACTION, is_train=True)

    # 初始化训练参数
    step_a = INTERVAL_A / INTERVAL_ENV

    # 循环更新
    for ep in range(EPISODE):
        # 训练log记录
        w_hist = []
        r_hist = []

        # 初始化资格迹
        et = np.zeros_like(w)

        #随机初始化环境和状态
        e = ENV()

        # 初次动作生成a_t & 特征编码s_t & 状态更新
        a = actor.act([e.c.dx, e.c.dy, e.c.vx, e.c.vy], w)
        en = encoder.encode([e.c.dx, e.c.dy, e.c.vx, e.c.vy], a)
        e.update(a)

        for t in range(int(T/INTERVAL_ENV)):
            # 动作仿真
            if t%step_a == 0:
                # 更新动作 a_{t+1}
                a_new = actor.act([e.c.dx, e.c.dy, e.c.vx, e.c.vy], w)
                # 更新特征 s_{t+1}
                en_new = encoder.encode([e.c.dx, e.c.dy, e.c.vx, e.c.vy], a_new)
                # 计算delta
                delta = e.r + GAMMA * np.matmul(en_new.T, w) - np.matmul(en.T, w)
                # 更新资格迹
                et = GAMMA*LAMBDA*et + en
                # 更新参数矩阵w
                w += ALPHA*delta*et

                a = a_new
                en = en_new

                # Log记录
                w_hist.append(np.sum(np.abs(delta)))
                r_hist.append(e.r)

            # 状态仿真
            e.update(a)
        
        # Log输出
        w_hist = np.array(w_hist)
        r_hist = np.array(r_hist)
        print ("EP{}:  delta_w:{:.2f}  total_r:{:.2f}  final_dist:{:.2f}  Vx:{:.2f}  Vy:{:.2f}".format(ep+1, np.sum(w_hist), np.sum(r_hist), -e.r, e.c.vx, e.c.vy))

        # 每10个ep存储一次参数矩阵w
        if (ep+1)%10 == 0:
            np.save(path, w)
            print ("Saved in {}".format(path))
            print ("-"*30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-Ep', default=1000)
    parser.add_argument('-Gamma', default=0.9)
    parser.add_argument('-Lambda', default=0.9)
    parser.add_argument('-Alpha', default=0.1, help='lr')
    parser.add_argument('-save_path', default="./p_w.npy")

    args = parser.parse_args()

    ############################
    ####      训练参数       ####
    ############################
    EPISODE = int(args.Ep)
    GAMMA = float(args.Gamma)
    LAMBDA = float(args.Lambda)
    ALPHA = float(args.Alpha)  # 学习率

    SAVE_PATH = args.save_path

    print ("-"*30)
    print ("EPISODE:{}\nGAMMA:{}\nLAMBDA:{}\nALPHA:{}\nSAVE_PATH:{}".format(EPISODE, GAMMA, LAMBDA, ALPHA, SAVE_PATH))

    p_loop(EPISODE, GAMMA, LAMBDA, ALPHA, SAVE_PATH)
