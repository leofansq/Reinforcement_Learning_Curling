"""
@leofansq
"""
import numpy as np

class CURLING():
    """
    冰壶状态
    """
    def __init__(self, w=100.0, h=100.0, target=[50.0, 50.0], r=1.0, m=1.0, x=None, y=None, vx=None, vy=None):
        """
        初始化冰壶状态
        """
        self.r = r
        self.m = m

        self.x = x if x is not None else np.random.random() * w 
        self.y = y if y is not None else np.random.random() * h

        self.vx = vx if vx is not None else np.random.random() * 20.0 - 10.0
        self.vy = vy if vy is not None else np.random.random() * 20.0 - 10.0

        self.fx = 0.005 * self.vx**2 * (-np.abs(self.vx)/self.vx) if self.vx else 0.0
        self.fy = 0.005 * self.vy**2 * (-np.abs(self.vy)/self.vy) if self.vy else 0.0

        self.ax = self.fx / self.m
        self.ay = self.fy / self.m

        self.dx = self.x - target[0]
        self.dy = self.y - target[1]

    def update(self, action=[0.0, 0.0], t=0.01, target=[50.0, 50.0]):
        """
        更新冰壶状态
        Parameters:
            action: 控制动作 [x轴方向施加的力, y轴方向施加的力]
            t: 仿真间隔
        """
        self.fx = 0.005 * self.vx**2 * (-np.abs(self.vx)/self.vx) + action[0] if self.vx else action[0]
        self.fy = 0.005 * self.vy**2 * (-np.abs(self.vy)/self.vy) + action[1] if self.vy else action[1]

        self.ax = self.fx / self.m
        self.ay = self.fy / self.m

        self.x += self.vx * t + 0.5 * self.ax * t**2
        self.y += self.vy * t + 0.5 * self.ay * t**2

        self.vx += self.ax * t
        self.vy += self.ay * t

        self.dx = self.x - target[0]
        self.dy = self.y - target[1]
    
    def copy(self, a):
        """
        复制一个已有的curling对象
        """
        self.r, self.m, self.x, self.y, self.vx, self.vy, self.fx, self.fy, self.ax, self.ay, self.dx, self.dy = \
        a.r, a.m, a.x, a.y, a.vx, a.vy, a.fx, a.fy, a.ax, a.ay, a.dx, a.dy

class ENV():
    """
    环境类
    """
    def __init__(self, w=100.0, h=100.0, alpha=0.9, target=None, c_r=1.0, c_m=1.0, c_x=None, c_y=None, c_vx=None, c_vy=None):
        # 初始化场地信息
        self.w = w
        self.h = h
        self.alpha = alpha    # 触边反弹系数
        # 初始化目标点
        if target is not None:
            self.target = target
        else:
            self.target = [np.random.random() * (self.w-2*c_r) + 1, np.random.random() * (self.h-2*c_r) + 1]
        # 初始化冰壶状态
        self.c = CURLING(self.w, self.h, self.target, c_r, c_m, c_x, c_y, c_vx, c_vy)
        # 初始化奖励
        self.r = self.reward()
        # 仿真间隔
        self.t = 0.01

    def update(self, action=[0.0, 0.0]):
        """
        状态更新
        """
        # 冰壶状态更新
        self.c.update(action, self.t, self.target)
        # 触边反弹判断 & 处理
        self.is_rebound()
        # 计算更新奖励
        self.r = self.reward()

    def is_rebound(self):
        """
        触边反弹情况判断&处理
        """
        # x方向触边反弹
        if (self.c.x < self.c.r) or (self.c.x > (self.w-self.c.r)):
            self.c.vx *= -self.alpha
            self.c.x = 2*self.c.r-self.c.x if self.c.x < self.c.r else 2 * (self.w-self.c.r) - self.c.x
        
        # y方向触边反弹
        if (self.c.y < self.c.r) or (self.c.y > (self.h-self.c.r)):
            self.c.vy *= -self.alpha         
            self.c.y = 2*self.c.r-self.c.y if self.c.y < self.c.r else 2 * (self.h-self.c.r) - self.c.y
    
    def reward(self):
        """
        计算奖励 r = -d
        距离使用欧氏距离
        """
        return -np.sqrt((self.target[0] - self.c.x)**2 + (self.target[1] - self.c.y)**2)
    
    def copy(self):
        """
        返回一个与当前状态相同的新环境env对象
        """
        cp = env()
        cp.w, cp.h, cp.alpha, cp.target, cp.r, cp.t = self.w, self.h, self.alpha, self.target, self.r, self.t
        cp.c.copy(self.c)

        return cp







    








        
