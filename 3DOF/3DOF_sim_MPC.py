import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt
import casadi.tools as ca_tools

from simulation_3DOF import *
from sail_angle import *


# 动力学函数

def sign_ca(value):
    ''' Implements the sign function using CasADi.

    param value: The value to get the sign from

    return: The sign of value {-1, 0, 1}
    '''
    # 使用CasADi的if_else函数实现
    return ca.if_else(value < 0, -1, ca.if_else(value > 0, 1, 0))


def calculate_apparent_wind(yaw, vel_x, vel_y, true_wind):
    # 返回船体的视风
    ''' Calculate the apparent wind on the boat.

    param yaw:          The heading of the boat [radians]
    param vel_x:        The velocity along the x-axis [m/s]
    param vel_y:        The velocity along the y-axis [m/s]
    param true_wind:    The true wind directions

    return: The apparent wind on the boat
    '''
    # 忽略z轴方向的变化
    # yaw为行进的方向角

    # dx/dt
    # 此时以船前进方向作为坐标系
    # 将全局风速映射到前进坐标系上
    transformed_x = true_wind.x * ca.cos(yaw) + true_wind.y * ca.sin(yaw)

    # dy/dt
    transformed_y = true_wind.x * -ca.sin(yaw) + true_wind.y * ca.cos(yaw)

    # 视风 = transformed_wind - v_boat
    apparent_x = transformed_x - vel_x
    apparent_y = transformed_y - vel_y

    # 视风角度(x,y)
    apparent_angle = ca.atan2(-apparent_y, -apparent_x)

    # 合速度
    apparent_speed = ca.sqrt(apparent_x ** 2 + apparent_y ** 2)

    # 返回视风信息
    return {'x': apparent_x,
            'y': apparent_y,
            'angle': apparent_angle,
            'speed': apparent_speed}


def calculate_sail_force_ca(wind, sail_angle):
    # 计算帆力
    ''' Calculate the force that is applied to the sail.

    param wind:         The apparent wind on the boat
    param sail_angle:   The angle of the main sail [radians]

    return: The force applied on the sail by the wind
    '''
    # aoa : angle of attack
    aoa = wind['angle'] - sail_angle
    aoa = ca.if_else(aoa * sail_angle < 0, 0, aoa)

    # eff_aoa : effective angle of attack
    eff_aoa = ca.if_else(aoa < -ca.pi / 2, ca.pi + aoa, ca.if_else(aoa > ca.pi / 2, -ca.pi + aoa, aoa))

    pressure = (AIR_DENSITY / 2) * wind['speed'] ** 2  # roll = 0, cos(roll * cos(
    # sail_angle)) ** 2 = 1

    friction = ca.if_else(wind['speed'] != 0,
                          3.55 * ca.sqrt(AIR_VISCOSITY / (wind['speed'] * SAIL_LENGTH)),
                          0)

    separation = 1 - ca.exp(-((ca.fabs(eff_aoa)) / (ca.pi / 180 * 25)) ** 2)

    propulsion = (2 * ca.pi * eff_aoa * ca.sin(wind['angle']) - (
            friction + (4 * ca.pi * eff_aoa ** 2 * separation) / SAIL_STRETCHING) * ca.cos(wind['angle'])) \
                 * SAIL_AREA * pressure

    transverse_force = (-2 * ca.pi * eff_aoa * ca.cos(wind['angle'])
                        - (friction + (4 * ca.pi * eff_aoa ** 2 * separation) / SAIL_STRETCHING) * ca.sin(
                wind['angle'])) \
                       * SAIL_AREA * pressure

    separated_propulsion = sign_ca(aoa) * pressure * SAIL_AREA * ca.sin(aoa) ** 2 * ca.sin(sail_angle)
    separated_transverse_force = -sign_ca(aoa) * pressure * SAIL_AREA * ca.sin(aoa) ** 2 * ca.cos(sail_angle)

    # 返回帆力信息
    return {'x': (1 - separation) * propulsion + separation * separated_propulsion,
            'y': (1 - separation) * transverse_force + separation * separated_transverse_force}


def calculate_lateral_force_ca(vel_x, vel_y, speed):
    # 计算横向力
    ''' Calculate the lateral force.
    注意此时的速度是以船体前进方向为标准坐标系
    param vel_x:        The velocity along the x-axis   [m/s]
    param vel_y:        The velocity along the y-axis   [m/s]
    param speed:        The total speed of the boat     [m/s]

    return: The force applied to the lateral plane of the boat
    '''
    pressure = (WATER_DENSITY / 2) * speed ** 2

    friction = ca.if_else(speed != 0,
                          2.66 * ca.sqrt(WATER_VISCOSITY / (speed * KEEL_LENGTH)),
                          0)

    #     aoa :           angle of attack,攻角
    # eff_aoa : effective angle of attack，有效攻角
    # 攻角相对于x轴方向
    eff_aoa = aoa = ca.atan2(vel_y, vel_x)
    eff_aoa = ca.if_else(aoa < -ca.pi / 2, ca.pi + aoa, ca.if_else(aoa > ca.pi / 2, -ca.pi + aoa, aoa))

    # 模拟船的分离效应
    separation = 1 - ca.exp(-((ca.fabs(eff_aoa)) / (ca.pi / 180 * 25)) ** 2)

    # Identical calculation for x and y
    tmp = -(friction + (4 * ca.pi * eff_aoa ** 2 * separation) / KEEL_STRETCHING)

    separated_transverse_force = -sign_ca(aoa) * pressure * SAIL_AREA * ca.sin(aoa) ** 2

    # 返回横向力信息
    return {'x': (1 - separation) * (tmp * ca.cos(aoa) + 2 * ca.pi * eff_aoa * ca.sin(aoa)) * pressure * LATERAL_AREA,
            'y': (1 - separation) * (tmp * ca.sin(aoa) - 2 * ca.pi * eff_aoa * ca.cos(aoa)) * pressure * LATERAL_AREA +
                 separation * separated_transverse_force}, separation


def calculate_rudder_force(speed, rudder_angle):
    # 计算船舵力
    ''' Calculate the force that is applied to the rudder.

    param speed:        The total speed of the boat [m/s]
    param rudder_angle: The angle of the rudder     [radians]

    return: The force applied to the rudder of the boat
    '''
    pressure = (WATER_DENSITY / 2) * speed ** 2

    # 返回龙骨力信息
    return {'x': -(((4 * ca.pi) / RUDDER_STRETCHING) * rudder_angle ** 2) * pressure * RUDDER_BLADE_AREA,
            'y': 2 * ca.pi * pressure * RUDDER_BLADE_AREA * rudder_angle}


def calculate_wave_impedance_ca(vel_x, speed):
    # 波浪阻抗
    ''' Calculate the wave impedance.

    param vel_x: The velocity along the x-axis  [m/s]
    param speed: The total speed of the boat    [m/s]

    return: The force applied to the rudder of the boat
    '''
    return -ca.sign(vel_x) * speed ** 2 * (speed / HULL_SPEED) ** 2 * WAVE_IMPEDANCE_INVARIANT


def calculate_wave_influence(pos_x, pos_y, yaw, wave, time):
    # 波浪影响
    # 静水时恒为0
    ''' Calculate how the waves influence the boat.

    param pos_x:    The boats position on the x-axis        [m]
    param pos_y:    The boats position on the y-axis        [m]
    param yaw:      The heading of the boat                 [radians]
    param wave:     The direction and length of the waves
    param time:     The simulation time                     [s]

    return: The influence of the waves on the boat
    '''
    frequency = ca.sqrt((2 * ca.pi * GRAVITY) / wave.length)

    k_x = 2 * ca.pi / wave.length * ca.cos(wave.direction)
    k_y = 2 * ca.pi / wave.length * ca.sin(wave.direction)

    factor = -wave.amplitude * ca.cos(frequency * time - k_x * pos_x - k_y * pos_y)
    gradient_x = k_x * factor
    gradient_y = k_y * factor
    return WaveInfluence(
        height=wave.amplitude * ca.sin(frequency * time - k_x * pos_x - k_y * pos_y),
        gradient_x=gradient_x * ca.cos(yaw) + gradient_y * ca.sin(yaw),
        gradient_y=gradient_y * ca.cos(yaw) - gradient_x * ca.sin(yaw))


def calculate_hydrostatic_force(wave_influence):
    # 流体静力
    ''' Calculate the hydrostatic force.

    param wave_influence:   The influence of the waves on the boat

    return: The force applied on the boat by the waves
    '''
    force = HYDROSTATIC_INVARIANT_Z * (- wave_influence.height) + GRAVITY_FORCE

    return HydrostaticForce(
        x=force * wave_influence.gradient_x,
        y=force * wave_influence.gradient_y,
        z=force), \
           HYDROSTATIC_EFF_Y * ca.sin(ca.atan(wave_influence.gradient_x)), \
           HYDROSTATIC_EFF_X * -ca.sin(- ca.atan(wave_influence.gradient_y))


def calculate_damping(vel_x, vel_y, yaw_rate):
    # 阻尼
    ''' Calculate the damping.

    param vel_x:        The velocity along the x-axis           [m/s]
    param vel_y:        The velocity along the y-axis           [m/s]
    param yaw_rate:     The rate of change to the yaw angle     [radians/s]

    return: The amount of damping applied to the boat
    '''
    return Damping(
        x=DAMPING_INVARIANT_X * vel_x,
        y=DAMPING_INVARIANT_Y * vel_y,
        yaw=DAMPING_INVARIANT_YAW * yaw_rate)


def sail_angle_ca(wind_angle, wind_speed, sail_stretching, limit_wind_speed=6, stall_deg=14.):
    # Optimal Angle of Attack
    opt_aoa = ca.sin(wind_angle) / (ca.cos(wind_angle) + .4 * ca.cos(wind_angle) ** 2) * sail_stretching / 4

    # Maximum Angle of Attack Limitation
    opt_aoa = ca.if_else(ca.fabs(opt_aoa) > stall_deg / 180. * ca.pi,
                         ca.sign(wind_angle) * stall_deg / 180. * ca.pi,
                         opt_aoa)

    # Heading Controllability at High Wind Speeds
    opt_aoa *= ca.if_else(wind_speed > limit_wind_speed,
                          (limit_wind_speed / wind_speed) ** 2,
                          1)

    # Ensure the output is within the range of [-pi/2, pi/2]
    return ca.fabs(ca.if_else(ca.sign(wind_angle - opt_aoa) > 0,
                              ca.fmin(ca.fmax(wind_angle - opt_aoa, -ca.pi / 2), ca.pi / 2),
                              ca.fmax(ca.fmin(wind_angle - opt_aoa, ca.pi / 2), -ca.pi / 2)))


"""""""""
def boat_dynamics(state_init, control_init):
    true_wave = environment[WAVE]
    true_wind = environment[TRUE_WIND]
    st = state_init
    ct = control_init
    pos_x, pos_y, yaw, vel_x, vel_y, yaw_rate = st[0], st[1], st[2], st[3], st[4], st[5]
    rudder_angle, sail_init = ct[0], ct[1]
    # 实际船速
    # 只考虑二维平面中速度
    speed = ca.sqrt(vel_x ** 2 + vel_y ** 2)  # + vel_z**2)
    # 波浪影响
    # 由于环境的设置，WAVE(50, 0, 0),在静水中的time变化并不会影响实际的情况
    wave_influence = calculate_wave_influence(pos_x, pos_y, yaw, true_wave, 0)
    # 视风
    apparent_wind = calculate_apparent_wind(yaw, vel_x, vel_y, true_wind)

    # sail angle is determined from rope length and wind direction
    # 顺/逆风 × 帆角的绝对值
    true_sail_angle = sign_ca(apparent_wind['angle']) * ca.fabs(sail_init)

    # Force calculation
    # 阻尼
    damping = calculate_damping(vel_x, vel_y, yaw_rate)

    # 流体静力
    hydrostatic_force, x_hs, y_hs = calculate_hydrostatic_force(wave_influence)
    # 静水时流体静力恒为0

    # 波浪阻抗
    wave_impedance = calculate_wave_impedance_ca(vel_x, speed)
    # 龙骨力
    rudder_force = calculate_rudder_force(speed, rudder_angle)
    # 横向力
    lateral_force, lateral_separation = calculate_lateral_force_ca(vel_x, vel_y, speed)
    # lateral_separation = 0
    sail_force = calculate_sail_force_ca(apparent_wind, true_sail_angle)
    delta_pos_x = vel_x * ca.cos(yaw) - vel_y * ca.sin(yaw)
    delta_pos_y = vel_y * ca.cos(yaw) + vel_x * ca.sin(yaw)
    delta_yaw = yaw_rate
    delta_vel_x = yaw_rate * vel_y + (
            sail_force['x'] + lateral_force['x'] + rudder_force['x']
            + damping.x + wave_impedance + hydrostatic_force.x) / MASS
    delta_vel_y = -yaw_rate * vel_x + ((sail_force['y'] + lateral_force['y'] + rudder_force['y']) + hydrostatic_force.y
                                       + damping.y) / MASS
    delta_yaw_rate = (damping.yaw
                      - rudder_force['y'] * DISTANCE_COG_RUDDER
                      + sail_force['y'] * DISTANCE_COG_SAIL_PRESSURE_POINT
                      + sail_force['x'] * sin(true_sail_angle) * DISTANCE_MAST_SAIL_PRESSURE_POINT
                      + lateral_force['y'] * (DISTANCE_COG_KEEL_PRESSURE_POINT * (1 - lateral_separation)
                                              + DISTANCE_COG_KEEL_MIDDLE * lateral_separation)) / MOI_Z
    return ca.vertcat(delta_pos_x, delta_pos_y, delta_yaw, delta_vel_x, delta_vel_y, delta_yaw_rate)
"""


def boat_dynamics(state_init, control_init):
    true_wave = environment[WAVE]
    true_wind = environment[TRUE_WIND]
    st = state_init
    ct = control_init
    pos_x, pos_y, yaw, vel_x, vel_y, yaw_rate = st[0], st[1], st[2], st[3], st[4], st[5]
    rudder_angle = ct
    # 实际船速
    # 只考虑二维平面中速度
    speed = ca.sqrt(vel_x ** 2 + vel_y ** 2)  # + vel_z**2)
    # 波浪影响
    # 由于环境的设置，WAVE(50, 0, 0),在静水中的time变化并不会影响实际的情况
    wave_influence = calculate_wave_influence(pos_x, pos_y, yaw, true_wave, 0)
    # 视风
    apparent_wind = calculate_apparent_wind(yaw, vel_x, vel_y, true_wind)
    sail_init = sail_angle_ca(apparent_wind['angle'], apparent_wind['speed'], SAIL_STRETCHING)

    # sail angle is determined from rope length and wind direction
    # 顺/逆风 × 帆角的绝对值
    true_sail_angle = sign_ca(apparent_wind['angle']) * ca.fabs(sail_init)

    # Force calculation
    # 阻尼
    damping = calculate_damping(vel_x, vel_y, yaw_rate)

    # 流体静力
    hydrostatic_force, x_hs, y_hs = calculate_hydrostatic_force(wave_influence)
    # 静水时流体静力恒为0

    # 波浪阻抗
    wave_impedance = calculate_wave_impedance_ca(vel_x, speed)
    # 龙骨力
    rudder_force = calculate_rudder_force(speed, rudder_angle)
    # 横向力
    lateral_force, lateral_separation = calculate_lateral_force_ca(vel_x, vel_y, speed)
    # lateral_separation = 0
    sail_force = calculate_sail_force_ca(apparent_wind, true_sail_angle)
    delta_pos_x = vel_x * ca.cos(yaw) - vel_y * ca.sin(yaw)
    delta_pos_y = vel_y * ca.cos(yaw) + vel_x * ca.sin(yaw)
    delta_yaw = yaw_rate
    delta_vel_x = yaw_rate * vel_y + (
            sail_force['x'] + lateral_force['x'] + rudder_force['x']
            + damping.x + wave_impedance + hydrostatic_force.x) / MASS
    delta_vel_y = -yaw_rate * vel_x + ((sail_force['y'] + lateral_force['y'] + rudder_force['y']) + hydrostatic_force.y
                                       + damping.y) / MASS
    delta_yaw_rate = (damping.yaw
                      - rudder_force['y'] * DISTANCE_COG_RUDDER
                      + sail_force['y'] * DISTANCE_COG_SAIL_PRESSURE_POINT
                      + sail_force['x'] * sin(true_sail_angle) * DISTANCE_MAST_SAIL_PRESSURE_POINT
                      + lateral_force['y'] * (DISTANCE_COG_KEEL_PRESSURE_POINT * (1 - lateral_separation)
                                              + DISTANCE_COG_KEEL_MIDDLE * lateral_separation)) / MOI_Z
    return ca.vertcat(delta_pos_x, delta_pos_y, delta_yaw, delta_vel_x, delta_vel_y, delta_yaw_rate)


def shift_movement(T, t0, x0, u, f):
    # 运动到下一个状态
    f_value = f(x0, u[:, 0])
    st = x0 + T * f_value
    # 时间增加
    t = t0 + T
    # 准备下一个估计的最优控制，因为u[:, 0]已经采纳，我们就简单地把后面的结果提前
    u_end = ca.horzcat(u[:, 1:], u[:, -1])
    return t, st, u_end.T


#####################
# 环境变量声明
T = 0.1  # （模拟的）系统采样时间【秒】
N = 100  # 需要预测的步长【超参数】
t0 = 0  # 初始时间
tf = T * N  # 结束时间
rudder_max = 15 * pi / 180  # 最大舵角【物理约束】
sail_max = pi / 2
wave = Wave(50, 0, 0)
wind = TrueWind(0, 5, 5, ca.pi / 2)
environment[WAVE] = wave
environment[TRUE_WIND] = wind
#####################

# 根据数学模型建模

# 1 系统状态
x = ca.SX.sym('x')  # x轴状态
y = ca.SX.sym('y')  # y轴状态
yaw = ca.SX.sym('yaw')  # 艏向角
u = ca.SX.sym('u')  # x速度
v = ca.SX.sym('v')  # y速度
r = ca.SX.sym('r')  # 艏向角变化率

states = ca.vertcat(x, y, yaw, u, v, r)

n_states = states.size()[0]  # 获得系统状态的尺寸，向量以（n_states, 1）的格式呈现 【这点很重要】

# 2 控制输入
rudder = ca.SX.sym('rudder')
# sail = ca.SX.sym('sail')  # 最优帆角，视为一个状态
# controls = ca.vertcat(rudder, sail)
controls = ca.vertcat(rudder)
n_controls = controls.size()[0]  # 控制向量尺寸

# 3 运动学模型
# 定义右手函数
rhs = boat_dynamics(states, controls)

"""""""""
rhs = ca.vertcat(
    1,  # Change in x position
    2,  # Change in y position
)
"""

# 利用CasADi构建一个函数
f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
"""""""""""
等价于构建了一个
def f(states, controls):
    return rhs
"""

# 4 开始构建MPC
# 4.1 相关变量，格式(状态长度， 步长)
U = ca.SX.sym('U', n_controls, N)  # N步内的控制输出
X = ca.SX.sym('X', n_states, N + 1)  # N+1步的系统状态，通常长度比控制多1
P = ca.SX.sym('P', n_states + n_states)  # 构建问题的相关参数,在这里每次只需要给定当前/初始位置和目标终点位置

# 4.2 Single Shooting 约束条件
X[:, 0] = P[:6]  # 初始状态希望相等, X的第一列和P的前三个都是起点状态

# 4.2剩余N状态约束条件
for i in range(N):
    # 通过前述函数获得下个时刻系统状态变化。
    # 这里需要注意引用的index为[:, i]，因为X为(n_states, N+1)矩阵
    f_value = f(X[:, i], U[:, i])  # delta_X
    X[:, i + 1] = X[:, i] + f_value * T

# 4.3获得输入（控制输入，参数）和输出（系统状态）之间关系的函数ff
ff = ca.Function('ff', [U, P], [X], ['input_U', 'target_state'], ['horizon_states'])

# NLP问题
# 惩罚矩阵
# 分别代表了对6个状态变量的惩罚系数
Q = np.zeros((6, 6))  # 创建一个全零的 6x6 矩阵
Q[:2, :2] = np.array([[2.0, 0.0],  # 对状态矩阵的前两个进行惩罚
                      [0.0, 2.0]])  # 对状态矩阵的前两个进行惩罚

# 构建惩罚矩阵 R
# R = np.array([[0.05, 0.0], [0.0, 0.05]])  # 惩罚控制输入
R = np.array([0.05])

# 优化目标
obj = 0  # 初始化优化目标值
for i in range(N):
    # 在N步内对获得优化目标表达式
    # .T表示矩阵转置,计算惩罚函数 对应误差的平方与系数相乘再相加
    # ca.mtimes,矩阵乘法操作
    # obj = obj + ca.mtimes([(X[:, i] - P[6:]).T, Q, X[:, i] - P[6:]]) + ca.mtimes([U[:, i].T, R, U[:, i]])
    obj = ca.mtimes([(X[:2, -1] - P[6:8]).T, Q[:2, :2], X[:2, -1] - P[6:8]]) + ca.mtimes([U[:, -1].T, R, U[:, -1]])

# 约束条件定义
g = []  # 用list来存储优化目标的向量
for i in range(N + 1):
    # 这里的约束条件只有小车的坐标（x,y）必须在-2至2之间
    # 由于xy没有特异性，所以在这个例子中顺序不重要（但是在更多实例中，这个很重要）
    # g中表示需要约束的内容
    # g.append(X[0, i])  # 第一行第n列
    # g.append(X[1, i])  # 第二行第n列
    pass

# 定义NLP问题，'f'为目标函数，'x'为需寻找的优化结果（优化目标变量），'p'为系统参数，'g'为约束条件
# 需要注意的是，用SX表达必须将所有表示成标量或者是一维矢量的形式
# .reshape(U, -1, 1):-1 矩阵总数, 1 一列
nlp_prob = {'f': obj, 'x': ca.reshape(U, -1, 1), 'p': P, 'g': ca.vertcat(*g)}
# ipot设置:
# ipopt.max_iter: 最大迭代次数
# ipopt.print_level: 输出信息的详细级别，0 表示关闭输出
# print_time: 控制是否输出求解时间
# ipopt.acceptable_tol: 接受的目标函数值的容忍度
# ipopt.acceptable_obj_change_tol: 接受的目标函数变化的容忍度
opts_setting = {'ipopt.max_iter': 100, 'ipopt.print_level': 0, 'print_time': 0, 'ipopt.acceptable_tol': 1e-1,
                'ipopt.acceptable_obj_change_tol': 1e-1}
# 最终目标，获得求解器:
# solver' 是求解器的名称
# ipopt' 指定了所使用的求解器为 IPOPT
# nlp_prob 是定义好的非线性优化问题
# opts_setting 是求解器的设置参数，告诉求解器如何进行求解
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

# 5 开始仿真
#   定义约束条件，实际上CasADi需要在每次求解前更改约束条件。不过我们这里这些条件都是一成不变的
#   因此我们定义在while循环外，以提高效率
#   状态约束
#   控制约束
lbx = []  # 最低约束条件
ubx = []  # 最高约束条件
for _ in range(N):
    #   U是以(n_controls, N)存储的，但是在定义问题的时候被改变成(n_controlsxN,1)的向量
    #   实际上，第一组控制rudder和sail的index为U_0为U_1，第二组为U_2和U_3
    #   因此，在这里约束必须在一个循环里连续定义。
    lbx.append(-rudder_max)
    ubx.append(rudder_max)
    # lbx.append(-sail_max)
    # ubx.append(sail_max)

# 仿真条件和相关变量
t0 = 0.0  # 仿真时间
x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)  # 初始始状态
xs = np.array([20.0, 10.0, np.nan, np.nan, np.nan, np.nan]).reshape(-1, 1)  # 末状态
u0 = np.array([0.0] * N).reshape(-1, 2)  # 系统初始控制状态，为了统一本例中所有numpy有关,N行,2列,每个值都是0
# 变量都会定义成（N,状态）的形式方便索引和print
x_c = []  # 存储系统的状态
u_c = []  # 存储控制全部计算后的控制指令
t_c = []  # 保存时间
xx = []  # 存储每一步位置
sim_time = 30  # 仿真时长
index_t = []  # 存储时间戳，以便计算每一步求解的时间

# 6 开始仿真
mpciter = 0  # 迭代计数器
start_time = time.time()  # 获取开始仿真时间
# 终止条件为目标的欧式距离小于4或者仿真超时
while np.linalg.norm(x0[:2] - xs[:2]) > 4 and mpciter - sim_time / T < 0.0:
    print("'''''''''''''''''''''''''")
    print("mpciter", mpciter)
    # 初始化优化参数
    # c_p中存储的是当前的位置信息和目标点的位置信息
    c_p = np.concatenate((x0, xs))
    # 初始化优化目标变量
    init_control = ca.reshape(u0, -1, 1)
    # 计算结果并且
    t_ = time.time()
    res = solver(x0=init_control, p=c_p, lbx=lbx, ubx=ubx)
    index_t.append(time.time() - t_)
    # 获得最优控制结果u
    u_sol = ca.reshape(res['x'], n_controls, N)  # 记住将其恢复U的形状定义
    # 每一列代表了系统在每个时间步上的最优控制输入
    ###
    ff_value = ff(u_sol, c_p)  # 利用之前定义ff函数获得根据优化后的结果
    # 之后N+1步后的状态（n_states, N+1）
    # 存储结果
    x_c.append(ff_value)
    u_c.append(u_sol[:, 0])
    t_c.append(t0)
    # 根据数学模型和MPC计算的结果移动并且准备好下一个循环的初始化目标
    t0, x0, u0 = shift_movement(T, t0, x0, u_sol, f)
    # 存储位置
    x0 = ca.reshape(x0, -1, 1)
    xx.append(x0.full())
    # 打印状态值
    print("Current State:", x0.full())
    # 计数器+1
    mpciter = mpciter + 1

# 绘制轨迹
plt.figure(figsize=(8, 6))
plt.plot(xs[0], xs[1], 'ro', markersize=10)  # 绘制目标点
plt.plot(xx[0][0], xx[0][1], 'bo', markersize=10)  # 绘制初始点
for i in range(len(xx) - 1):
    plt.plot([xx[i][0], xx[i + 1][0]], [xx[i][1], xx[i + 1][1]], 'k--')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Car Trajectory')
plt.grid(True)
plt.axis('equal')
plt.show()

"""""""""
# SIMULATION_TEST
# casadi系统仿真测试
####################################################################################
# Initial state
initial_state = [0, 0, 0, 0, 0.0, 0]
# Simulate
num_steps = 500
trajectory = np.zeros((num_steps + 1, 2))
current_state = initial_state
# current_control = [ca.pi * 15 / 180, 0]
current_control = ca.pi * 15 / 180
print("初始状态", current_state)
for i in range(num_steps + 1):
    print(f'第{i}步')
    # Create an integrator to simulate the system
    integrator = ca.integrator('integrator', 'rk', {'x': states, 'p': controls, 'ode': rhs},
                               {'grid': [i * T, i * T + T]})

    trajectory[i, 0] = current_state[0]  # Store x
    trajectory[i, 1] = current_state[1]  # Store y
    #apparent_wind = calculate_apparent_wind(current_state[2], current_state[3], current_state[4],
    #                                        environment[TRUE_WIND])
    #best_sail = sail_angle(apparent_wind['angle'], apparent_wind['speed'], SAIL_STRETCHING)
    #print("best sail angle:", best_sail)
    #current_control = [current_control[0], best_sail]
    # 调用 integrator 并获得结果
    integrator_result = integrator(x0=current_state, p=current_control)
    current_state = integrator_result['xf'].full().flatten()
    print("current_state", current_state)

# Plot the trajectory of x and y
plt.figure()
plt.plot(trajectory[:, 0], trajectory[:, 1], '-o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory of x and y')
plt.grid(True)
plt.show()
####################################################################################
"""
