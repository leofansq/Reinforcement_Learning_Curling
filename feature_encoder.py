"""
@leofansq
"""
import numpy as np

class FEATURE_ENCODER():
    """
    特征编码器(径向基, 非均匀, 高斯)
    """
    def __init__(self, ACTION):
        # dx
        c_1 = [-65.5, -23, -11, -5, -2, -0.5, 0.5, 2, 5, 11, 23, 65.5]
        sigma_1 = [34.5, 8, 4, 2, 1, 0.5, 0.5, 1, 2, 4, 8, 34.5]
        # dy
        c_2 = [-65.5, -23, -11, -5, -2, -0.5, 0.5, 2, 5, 11, 23, 65.5]
        sigma_2 = [34.5, 8, 4, 2, 1, 0.5, 0.5, 1, 2, 4, 8, 34.5]
        # vx
        c_3 = [-26, -14, -8, -4.5, -2, -0.5, 0.5, 2, 4.5, 8, 14, 26]
        sigma_3 = [8, 4, 2, 1.5, 1, 0.5, 0.5, 1, 1.5, 2, 4, 8]
        # vy
        c_4 = [-26, -14, -8, -4.5, -2, -0.5, 0.5, 2, 4.5, 8, 14, 26]
        sigma_4 =  [8, 4, 2, 1.5, 1, 0.5, 0.5, 1, 1.5, 2, 4, 8]

        self.c = self._generate_comb(c_1, c_2, c_3, c_4)
        self.sigma = self._generate_comb(sigma_1, sigma_2, sigma_3, sigma_4)
        self.ACTION = ACTION
        

    def _generate_comb(self, a, b, c, d):
        """
        生成由四个不同维度的特征参数列表组合得到的状态空间特征参数列表, 用于后续特征的编码
        Parameters:
            a, b, c, d: 四个不同维度的特征参数列表. 
            本例中分别为: a--冰壶与目标x方向相对位置特征参数
                        b--冰壶与目标y方向相对位置特征参数
                        c--冰壶x方向速度
                        d--冰壶y方向速度
        Return:
            列表(4, len(a)*len(b)*len(c)*len(d))
        """
        la, lb, lc, ld = len(a), len(b), len(c), len(d)
        a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)

        a = np.tile(a, (1, lb*lc*ld)).T
        b = np.tile(np.reshape(np.tile(b, (la, 1)).T, (-1, 1)), (lc*ld, 1))
        c = np.tile(np.reshape(np.tile(c, (la*lb, 1)).T, (-1, 1)), (ld, 1))
        d = np.reshape(np.tile(d, (la*lb*lc, 1)).T, (-1, 1))

        return np.hstack((a,b,c,d))
        
    
    def encode(self, s, a):
        """
        特征编码(径向基, 高斯)
        Parameter:
            s: 状态观测值列表 [dx, dy, vx, vy]
            a: 动作
        Return:
            编码后的特征矩阵 (len(a)*len(b)*len(c)*len(d)*len(ACTION), 1)
        """
        # 根据输入状态计算状态空间特征(高斯)
        s = np.array(s)
        s = np.sum((s - self.c)**2 / self.sigma**2, axis=1)
        s = np.exp(-s/2)
        s = s / np.sum(s, axis=0)

        # 生成动作空间Mask
        act_idx = []
        for i in self.ACTION:
            act_idx.append(1.0 if i==a else 0.0)

        act_idx = np.reshape(np.tile(act_idx, (s.shape[0], 1)).T, (-1, 1))
        s = np.tile(s, (1, len(self.ACTION))).T

        return s*act_idx

        

if __name__ == "__main__":
    ACTION = [[5.0, 0.0], [-5.0, 0.0], [0.0, 5.0], [0.0, -5.0]]

    encoder = FEATURE_ENCODER(ACTION)

    s = [-65.5, -65.5, -26, -26]
    a = [5.0, 0.0]
    
    print(encoder.encode(s, a).shape)
