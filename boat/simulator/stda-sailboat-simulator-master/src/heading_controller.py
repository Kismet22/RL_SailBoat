from numpy import *
from scipy.linalg import solve_continuous_are

import yaml

#sim_params = open('sim_params_config.yaml')
#param_dict = yaml.load(sim_params)
with open('sim_params_config.yaml', 'r') as stream:
    param_dict = yaml.load(stream, Loader=yaml.FullLoader)


# COG:center of gravity
DISTANCE_COG_RUDDER = param_dict['boat']['distance_cog_rudder']  # 舵到重心的距离
# MOI:moment of inertia(惯性时间)
MOI_Z = param_dict['boat']['moi_z']  # z轴方向
RUDDER_BLADE_AREA = param_dict['boat']['rudder']['area']
WATER_DENSITY = param_dict['environment']['water_density']
YAW_TIMECONSTANT = param_dict['boat']['yaw_timeconstant']


# 前进方向调整

class heading_controller(object):
    def __init__(self, sample_time, factor=None, speed_adaption=.3, max_rudder_angle=15. / 180 * pi):
        # params:
        # 最大舵角:15
        self.sample_time = sample_time  # 采样时间
        self.speed_adaption = speed_adaption  # 采样速度
        if factor is None:
            self.factor = DISTANCE_COG_RUDDER * RUDDER_BLADE_AREA * pi * WATER_DENSITY / MOI_Z
        else:
            self.factor = factor
        self.max_rudder_angle = max_rudder_angle

        # controller params:
        # PID参数选择
        self.KP = .81  # .67 sampletime 0.5
        self.KI = .17  # 0.07
        self.KD = 1.25  # 1.04

        self.KP = .5
        self.KI = .1
        self.KD = .9

        # discrete controller sample_time 1s

        # self.KP = 2.92
        # self.KD = 2.77
        # self.KI = 1.08 / sample_time
        # discrete controller sample_time 2s
        # self.KP = .77
        # self.KD = .97
        # self.KI = .3 / sample_time

        ## discrete controller sample_time 5s
        # self.KP = .154
        # self.KD = .32
        # self.KI = .062 / sample_time

        # defensiv
        # self.KP = .33
        # self.KI = .055
        # self.KD = .7

        # initial values:
        self.summed_error = 0  # 误差的积分
        self.filtered_drift = 0  # 航向偏差的滤波值

    # 航向参数计算
    def calculate_controller_params(self, yaw_time_constant=None, store=True, Q=None, r=None):
        if yaw_time_constant is None:
            try:
                yaw_time_constant = YAW_TIMECONSTANT  # 时间常数
            except:
                raise ()

        # approximated system modell, linearized, continous calculated (seems to be fine)
        A = array([[0, 1, 0], [0, -1. / yaw_time_constant, 0], [-1, 0, 0]])  # 3×3(state)
        B = array([0, 1, 1])  # 输入，3×1，舵角

        # weigth matrices for riccati design
        if Q is None:
            Q = diag([1E-1, 1, 0.3])  # 3×3的对角阵
            r = ones((1, 1)) * 30  # 一个数字的矩阵(1×1)

        # Q = diag([1E-1, 1, 0.3])
        # r = ones((1,1))*1000

        # calculating feedback
        # 得到状态反馈的增益矩阵
        # Riccati方程
        P = solve_continuous_are(A, B[:, None], Q, r)
        K = sum(B[None, :] * P, axis=1) / r[0, 0]

        if store:
            # 通过K设置PID参数
            self.KP = K[0]
            self.KD = K[1]
            self.KI = -K[2]
        print(K)

        return list(K)

    def controll(self, desired_heading, heading, yaw_rate, speed, roll, drift_angle=0):

        # calculate error, difference and sum of error
        # 期望航向-当前航向
        # 单位是值(int)
        heading_error = desired_heading - heading

        # print 'heading_error', heading_error/pi*180
        # respect to periodicity of angle: maximum difference is 180 deg resp. pi
        while heading_error > pi:
            heading_error -= 2 * pi
        while heading_error < -pi:
            heading_error += 2 * pi

        # 积分误差
        self.summed_error += self.sample_time * (heading_error - drift_angle)

        # avoid high gains at low speed (singularity)
        if speed < self.speed_adaption:
            speed = self.speed_adaption

        factor2 = -1. / self.factor / speed ** 2 / cos(roll)

        # control equation
        # PID
        rudder_angle = factor2 * (self.KP * heading_error + self.KI * self.summed_error - self.KD * yaw_rate)

        # rudder restriction and anti-windup
        if abs(rudder_angle) > self.max_rudder_angle:
            rudder_angle = sign(rudder_angle) * self.max_rudder_angle
            self.summed_error = (rudder_angle / factor2 - (self.KP * heading_error - self.KD * yaw_rate)) / self.KI

        return rudder_angle
