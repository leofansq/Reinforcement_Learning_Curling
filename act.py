"""
@leofansq
"""
import numpy as np


class ACTOR():
    """
    动作生成器
    """
    def __init__(self, encoder, ACTION, is_train=False):
        """
        encoder: 特征编码器
        ACTION: 动作空间集合
        is_train: 是否处于训练阶段flag
        """
        self.encoder = encoder
        self.ACTION = ACTION
        self.is_train = is_train
    
    def act(self, s, w):
        """
        根据状态生成动作, 训练时采用epsilon贪心策略, 测试时采用贪心策略
        Parameter:
            s: 状态观测量列表 [dx, dy, vx, vy]
            w: 参数矩阵
        Return:
            执行的动作 e.g [5.0, 0.0]
        """
        # 若冰壶到达目标点, 则不再执行动作
        # s_abs = np.abs(s)
        # if np.max(s_abs) < 0.5: return [0.0, 0.0]

        # 训练时采用epsilon贪心策略
        if self.is_train and np.random.rand() <= 0.1:
            return self.ACTION[int(np.random.random()*4+0.5)-1]
        else:
            q = []
            for i in self.ACTION:
                q.append( np.matmul(self.encoder.encode(s, i).T, w) )
            q = np.array(q)
            # print (q)
            return self.ACTION[np.argmax(q)]


if __name__ == "__main__":
    from feature_encoder import FEATURE_ENCODER

    ACTION = [[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0], [0.0, -5.0]]
    encoder = FEATURE_ENCODER(ACTION)

    a = ACTOR(encoder, ACTION)
    w = np.ones((12*12*12*12*4, 1))
    s = [-65.5, -65.5, -26, -26]
    
    print (a.act(s, w))

            

    


