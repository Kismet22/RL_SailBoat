#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' This module simulates a sailboat with 3 degrees of freedom. '''

###################################################################################################
# Standard import
from sys import argv
from math import pi, sin, cos, atan, atan2, sqrt, copysign, degrees, radians, modf
from collections import namedtuple
import casadi as ca
# from time import sleep, clock
from time import sleep, process_time

###################################################################################################
# Library import
import numpy as np
import yaml
from scipy.integrate import ode

###################################################################################################
# Load simulation parameters
# sim_params = open('sim_params_config.yaml')
# param_dict = yaml.load(sim_params)
with open('../3DOF/sim_params_config.yaml', 'r') as stream:
    param_dict = yaml.load(stream, Loader=yaml.FullLoader)

# 船体信息
# Boat
BOAT_LENGTH = param_dict['boat']['length']  # 船长，未使用
MASS = param_dict['boat']['mass']  # 船的质量
HULL_SPEED = param_dict['boat']['hull_speed']  # 船体指在水里的最大速度（与船体长度相关），用来计算水的阻抗
LATERAL_AREA = param_dict['boat']['lateral_area']  # 船侧面积，算波浪的时候用的
WATERLINE_AREA = param_dict['boat']['waterline_area']  # 船入水后水线包围的面积，算静浮力用的
HEIGHT_BOUYANCY = param_dict['boat']['height_bouyancy']  # 浮心在重心以上多高

# MOI : moment of inertia
GEOMETRICAL_MOI_X = param_dict['boat']['geometrical_moi_x']  # 绕x轴的几何转动惯量
GEOMETRICAL_MOI_Y = param_dict['boat']['geometrical_moi_y']  # 绕y轴的几何转动惯量
MOI_X = param_dict['boat']['moi_x']  # 船绕轴的真正转动惯量
MOI_Y = param_dict['boat']['moi_y']
MOI_Z = param_dict['boat']['moi_z']

# COG : center of gravity
DISTANCE_COG_RUDDER = param_dict['boat']['distance_cog_rudder']  # 舵受力点离船重心水平多远(sqrt(x^2+y^2))
DISTANCE_COG_SAIL_PRESSURE_POINT = param_dict['boat']['distance_cog_sail_pressure_point']  # 帆受力点离船重心水平多远(sqrt(x^2+y^2))
DISTANCE_COG_KEEL_PRESSURE_POINT = param_dict['boat']['distance_cog_keel_pressure_point']  # 龙骨受力点离船重心水平多远(sqrt(
# x^2+y^2))
DISTANCE_MAST_SAIL_PRESSURE_POINT = param_dict['boat']['distance_mast_sail_pressure_point']

# Sail
SAIL_LENGTH = param_dict['boat']['sail']['length']  # [m]  用来算帆的摩擦阻力
SAIL_HEIGHT = param_dict['boat']['sail']['height']  # [m]  未使用
SAIL_AREA = param_dict['boat']['sail']['area']  # [m^2]  计算帆受力用
SAIL_STRETCHING = param_dict['boat']['sail']['stretching']  # 计算派生阻力时用到
SAIL_PRESSURE_POINT_HEIGHT = param_dict['boat']['sail']['pressure_point_height']  # [m]
# 帆受力点高度（z，相对重心），将帆上的受力集中在一个点上，方便计算

# Keel
KEEL_LENGTH = param_dict['boat']['keel']['length']  # 龙骨长
KEEL_HEIGHT = param_dict['boat']['keel']['height']  # 龙骨高
KEEL_STRETCHING = param_dict['boat']['keel']['stretching']  # 计算水的派生阻力时候用到了

# Rudder
RUDDER_BLADE_AREA = param_dict['boat']['rudder']['area']  # 面积
RUDDER_STRETCHING = param_dict['boat']['rudder']['stretching']  # 展弦比

# Damping
# 阻尼
ALONG_DAMPING = param_dict['boat']['along_damping']  # 纵向阻尼,前后水平晃动
TRANSVERSE_DAMPING = param_dict['boat']['transverse_damping']  # 横向阻尼
DAMPING_Z = param_dict['boat']['damping_z']  # 竖直阻尼
# 由三个角度产生的阻尼
YAW_TIMECONSTANT = param_dict['boat']['yaw_timeconstant']
PITCH_DAMPING = param_dict['boat']['pitch_damping']
ROLL_DAMPING = param_dict['boat']['roll_damping']

# Physical constants
WATER_DENSITY = param_dict['environment']['water_density']
WATER_VISCOSITY = param_dict['environment']['water_viscosity']
AIR_VISCOSITY = param_dict['environment']['air_viscosity']
AIR_DENSITY = param_dict['environment']['air_density']
GRAVITY = param_dict['environment']['gravity']

###################################################################################################
# Invariants
# Rudder force
# RUDDER_FORCE_INVARIANT_X = -(WATER_DENSITY / 2) * RUDDER_BLADE_AREA
# RUDDER_FORCE_INVARIANT_Y = 2 * pi * (WATER_DENSITY / 2) * RUDDER_BLADE_AREA
# Wave impedance
# 波浪阻抗。不变
WAVE_IMPEDANCE_INVARIANT = (WATER_DENSITY / 2) * LATERAL_AREA  # (水密度/2) * 船侧面积

# Hydrostatic force
# 静水压力
HYDROSTATIC_EFF_X = HEIGHT_BOUYANCY + (WATER_DENSITY / MASS) * GEOMETRICAL_MOI_X
HYDROSTATIC_EFF_Y = HEIGHT_BOUYANCY + (WATER_DENSITY / MASS) * GEOMETRICAL_MOI_Y
# 浮心相对重心高度 + (水密度/ 质量) * 转动惯量

HYDROSTATIC_INVARIANT_Z = - WATER_DENSITY * WATERLINE_AREA * GRAVITY
# z轴静水压力，不变

GRAVITY_FORCE = MASS * GRAVITY
# 重力

# Damping
# 阻尼 invariant(不改变的)
DAMPING_INVARIANT_X = -MASS / ALONG_DAMPING  # x方向阻尼
DAMPING_INVARIANT_Y = -MASS / TRANSVERSE_DAMPING  # y方向阻尼
DAMPING_INVARIANT_Z = -.5 * DAMPING_Z * sqrt(WATER_DENSITY * WATERLINE_AREA * GRAVITY * MASS)  # z方向阻尼
DAMPING_INVARIANT_YAW = -(MOI_Z / YAW_TIMECONSTANT)  # 前进方向夹角
DAMPING_INVARIANT_PITCH = -2 * PITCH_DAMPING * sqrt(MOI_Y * MASS * GRAVITY * HYDROSTATIC_EFF_Y)
DAMPING_INVARIANT_ROLL = -2 * ROLL_DAMPING * sqrt(MOI_X * MASS * GRAVITY * HYDROSTATIC_EFF_X)

###################################################################################################
# Structured data
# Environment
TrueWind = namedtuple('TrueWind', 'x, y, strength, direction')  # 真实风向
ApparentWind = namedtuple('ApparentWind', 'x, y, angle, speed')  # 视风
Wave = namedtuple('Wave', 'length, direction, amplitude')  # 波浪
WaveVector = namedtuple('WaveVector', 'x, y')
WaveInfluence = namedtuple('WaveInfluence', 'height, gradient_x, gradient_y')
# Forces
RudderForce = namedtuple('RudderForce', 'x, y')
LateralForce = namedtuple('LateralForce', 'x, y')  # 横向力
SailForce = namedtuple('SailForce', 'x, y')
HydrostaticForce = namedtuple('HydrostaticForce', 'x, y, z')  # 静水压力
Damping = namedtuple('Damping', 'x, y, yaw')

###################################################################################################
# State description
# 全局位置x,y,z
# 旋转角,倾斜角,前进方向夹角(YAW)
# 速度该变量vx,vy,vz(以船体为坐标系)
# 角度p,q,r
POS_X, POS_Y, \
YAW, \
VEL_X, VEL_Y, \
YAW_RATE = range(6)


def initial_state():
    # 初态设置
    ''' Returns the initial state for a simulation.
        Consists of position, velocity, rotation and angular velocity. '''
    return np.array([0,
                     0,
                     param_dict['simulator']['initial']['yaw'],
                     param_dict['simulator']['initial']['vel_x'],
                     param_dict['simulator']['initial']['vel_y'],
                     param_dict['simulator']['initial']['yaw_rate']])


###################################################################################################
# Environment index description
# 创建环境信息枚举
SAIL_ANGLE, RUDDER_ANGLE, TRUE_WIND, WAVE = range(4)


###################################################################################################
# Force calculations
# pylint: enable = C0326
def sign(value):
    ''' Implements the sign function.

    param value: The value to get the sign from

    return: The sign of value {-1, 0, 1}
    '''
    # 返回-1，0, +1(根据value的符号)
    return copysign(1, value) if value != 0 else 0


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
    transformed_x = true_wind.x * cos(yaw) + true_wind.y * sin(yaw)
    # dy/dt
    transformed_y = true_wind.x * -sin(yaw) + true_wind.y * cos(yaw)

    # 视风 = transformed_wind - v_boat
    apparent_x = transformed_x - vel_x
    apparent_y = transformed_y - vel_y
    # 视风角度(x,y)
    apparent_angle = atan2(-apparent_y, -apparent_x)
    # 合速度
    apparent_speed = sqrt(apparent_x ** 2 + apparent_y ** 2)

    # 返回视风信息
    return ApparentWind(x=apparent_x,
                        y=apparent_y,
                        angle=apparent_angle,
                        speed=apparent_speed)


def calculate_sail_force(wind, sail_angle):
    # 计算帆力
    ''' Calculate the force that is applied to the sail.

    param roll:         The roll angle of the boat [radians]
    param wind:         The apparent wind on the boat
    param sail_angle:   The angle of the main sail [radians]

    return: The force applied on the sail by the wind
    '''
    # aoa : angle of attack
    aoa = wind.angle - sail_angle
    if aoa * sail_angle < 0:
        aoa = 0
    # eff_aoa : effective angle of attack
    eff_aoa = aoa
    if aoa < -pi / 2:
        eff_aoa = pi + aoa
    elif aoa > pi / 2:
        eff_aoa = -pi + aoa

    pressure = (AIR_DENSITY / 2) * wind.speed ** 2  # roll = 0, cos(roll * cos(
    # sail_angle)) ** 2 = 1

    friction = 3.55 * sqrt(AIR_VISCOSITY / (wind.speed * SAIL_LENGTH)) \
        if wind.speed != 0 \
        else 0

    separation = 1 - np.exp(-((abs(eff_aoa)) / (pi / 180 * 25)) ** 2)

    propulsion = (2 * pi * eff_aoa * sin(wind.angle) - (
            friction + (4 * pi * eff_aoa ** 2 * separation) / SAIL_STRETCHING) * cos(wind.angle)) \
                 * SAIL_AREA * pressure

    transverse_force = (-2 * pi * eff_aoa * cos(wind.angle)
                        - (friction + (4 * pi * eff_aoa ** 2 * separation) / SAIL_STRETCHING) * sin(wind.angle)) \
                       * SAIL_AREA * pressure

    separated_propulsion = sign(aoa) * pressure * SAIL_AREA * sin(aoa) ** 2 * sin(sail_angle)
    separated_transverse_force = -sign(aoa) * pressure * SAIL_AREA * sin(aoa) ** 2 * cos(sail_angle)

    # 返回帆力信息
    return SailForce(
        x=(1 - separation) * propulsion + separation * separated_propulsion,
        y=(1 - separation) * transverse_force + separation * separated_transverse_force)


def calculate_lateral_force(vel_x, vel_y, speed):
    # 计算横向力
    ''' Calculate the lateral force.
    注意此时的速度是以船体前进方向为标准坐标系
    param vel_x:        The velocity along the x-axis   [m/s]
    param vel_y:        The velocity along the y-axis   [m/s]
    param roll:         The roll angle of the boat      [radians]
    param speed:        The total speed of the boat     [m/s]

    return: The force applied to the lateral plane of the boat
    '''
    pressure = (WATER_DENSITY / 2) * speed ** 2

    friction = 2.66 * sqrt(WATER_VISCOSITY / (speed * KEEL_LENGTH)) \
        if speed != 0 \
        else 0

    #     aoa :           angle of attack,攻角
    # eff_aoa : effective angle of attack，有效攻角
    # 攻角相对于x轴方向
    eff_aoa = aoa = atan2(vel_y, vel_x)
    if aoa < -pi / 2:
        eff_aoa = pi + aoa
    elif aoa > pi / 2:
        eff_aoa = -pi + aoa

    # 模拟船的分离效应
    separation = 1 - np.exp(-((abs(eff_aoa)) / (pi / 180 * 25)) ** 2)

    # Identical calculation for x and y
    tmp = -(friction + (4 * pi * eff_aoa ** 2 * separation) / KEEL_STRETCHING)

    separated_transverse_force = -sign(aoa) * pressure * SAIL_AREA * sin(aoa) ** 2

    # 返回横向力信息
    return LateralForce(
        x=(1 - separation) * (tmp * cos(aoa) + 2 * pi * eff_aoa * sin(aoa)) * pressure * LATERAL_AREA,
        y=(1 - separation) * (tmp * sin(aoa) - 2 * pi * eff_aoa * cos(
            aoa)) * pressure * LATERAL_AREA + separation * separated_transverse_force) \
        , separation


def calculate_rudder_force(speed, rudder_angle):
    # 计算船舵力
    ''' Calculate the force that is applied to the rudder.

    param speed:        The total speed of the boat [m/s]
    param rudder_angle: The angle of the rudder     [radians]

    return: The force applied to the rudder of the boat
    '''
    pressure = (WATER_DENSITY / 2) * speed ** 2

    # 返回龙骨力信息
    return RudderForce(
        x=-(((4 * pi) / RUDDER_STRETCHING) * rudder_angle ** 2) * pressure * RUDDER_BLADE_AREA,
        y=2 * pi * pressure * RUDDER_BLADE_AREA * rudder_angle)


def calculate_wave_impedance(vel_x, speed):
    # 波浪阻抗
    ''' Calculate the wave impedance.

    param vel_x: The velocity along the x-axis  [m/s]
    param speed: The total speed of the boat    [m/s]

    return: The force applied to the rudder of the boat
    '''
    return -sign(vel_x) * speed ** 2 * (speed / HULL_SPEED) ** 2 * WAVE_IMPEDANCE_INVARIANT


def calculate_wave_impedance_ca(vel_x, speed):
    # 波浪阻抗
    ''' Calculate the wave impedance.

    param vel_x: The velocity along the x-axis  [m/s]
    param speed: The total speed of the boat    [m/s]

    return: The force applied to the rudder of the boat
    '''
    return -ca.if_else(vel_x > 0, 1, ca.if_else(vel_x < 0, -1, 0)) * speed ** 2 * (
            speed / HULL_SPEED) ** 2 * WAVE_IMPEDANCE_INVARIANT


def calculate_wave_influence(pos_x, pos_y, yaw, wave, time):
    # 波浪影响
    # 静水时恒为0?
    # print("现在计算波浪影响")
    ''' Calculate how the waves influence the boat.

    param pos_x:    The boats position on the x-axis        [m]
    param pos_y:    The boats position on the y-axis        [m]
    param yaw:      The heading of the boat                 [radians]
    param wave:     The direction and length of the waves
    param time:     The simulation time                     [s]

    return: The influence of the waves on the boat
    '''
    frequency = sqrt((2 * pi * GRAVITY) / wave.length)

    k = WaveVector(x=2 * pi / wave.length * cos(wave.direction),
                   y=2 * pi / wave.length * sin(wave.direction))

    factor = -wave.amplitude * cos(frequency * time - k.x * pos_x - k.y * pos_y)
    gradient_x = k.x * factor
    gradient_y = k.y * factor
    # print("高度", wave.amplitude * sin(frequency * time - k.x * pos_x - k.y * pos_y))
    # print("X", gradient_x * cos(yaw) + gradient_y * sin(yaw))
    return WaveInfluence(
        height=wave.amplitude * sin(frequency * time - k.x * pos_x - k.y * pos_y),
        gradient_x=gradient_x * cos(yaw) + gradient_y * sin(yaw),
        gradient_y=gradient_y * cos(yaw) - gradient_x * sin(yaw))


def calculate_hydrostatic_force(wave_influence):
    # 流体静力
    ''' Calculate the hydrostatic force.

    param pos_z:            The boats position on the z-axis        [m]
    param roll:             The roll angle of the boat              [radians]
    param pitch:            The pitch angle of the boat             [radians]
    param wave_influence:   The influence of the waves on the boat

    return: The force applied on the boat by the waves
    '''
    force = HYDROSTATIC_INVARIANT_Z * (- wave_influence.height) + GRAVITY_FORCE

    return HydrostaticForce(
        x=force * wave_influence.gradient_x,
        y=force * wave_influence.gradient_y,
        z=force), \
           HYDROSTATIC_EFF_Y * sin(atan(wave_influence.gradient_x)), \
           HYDROSTATIC_EFF_X * -sin(- atan(wave_influence.gradient_y)),


def calculate_damping(vel_x, vel_y, yaw_rate):
    # 阻尼
    ''' Calculate the damping.

    param vel_x:        The velocity along the x-axis           [m/s]
    param vel_y:        The velocity along the y-axis           [m/s]
    param vel_z:        The velocity along the z-axis           [m/s]
    param roll_rate:    The rate of change to the roll angle    [radians/s]
    param pitch_rate:   The rate of change to the pitch angle   [radians/s]
    param yaw_rate:     The rate of change to the yaw angle     [radians/s]

    return: The amount of damping applied to the boat
    '''
    return Damping(
        x=DAMPING_INVARIANT_X * vel_x,
        y=DAMPING_INVARIANT_Y * vel_y,
        yaw=DAMPING_INVARIANT_YAW * yaw_rate)


''' Solving
    以下部分是积分器的定义，进行积分运算的内容
    actor_dynamics = True
    此时的状态有: 
    # 全局位置x,y,z
    # 旋转角,倾斜角,前进方向夹角(YAW)
    # 速度该变量vx,vy,vz
    # 角度p,q,r(角动量的一阶信息)
    # 舵状态，帆状态
'''
# 舵的状态、帆状态
RUDDER_STATE, SAIL_STATE = 6, 7
actor_dynamics = False

DISTANCE_COG_KEEL_MIDDLE = DISTANCE_COG_KEEL_PRESSURE_POINT - .7


def solve(time, boat):
    """
    Solve the ode for the given state, time and environment.
    """

    """
    初始化
    """
    # State unpacking
    # pylint: disable = C0326

    pos_x, pos_y = boat[POS_X: POS_Y + 1]
    yaw = boat[YAW]
    vel_x, vel_y = boat[VEL_X: VEL_Y + 1]
    yaw_rate = boat[YAW_RATE]
    # actor_dynamics = True
    # 考虑船体力学信息的影响
    if actor_dynamics:
        # boat中保存的是本次舵角和帆角的信息
        rudder_angle, sail_angle = boat[RUDDER_STATE: SAIL_STATE + 1]
        # environment unpacking
        # environment中存储的是期望达到的舵角和帆角的信息(输入的舵角和帆角信息)
        sail_angle_reference = environment[SAIL_ANGLE]  # environment中存储sail_angle位置的元素
        rudder_angle_reference = environment[RUDDER_ANGLE]  # environment中存储rudder_angle位置的元素

    else:
        sail_angle = environment[SAIL_ANGLE]
        rudder_angle = environment[RUDDER_ANGLE]

    wave = environment[WAVE]
    true_wind = environment[TRUE_WIND]

    """
    For force calulations needed values
    运动学信息、力学信息
    """
    # 实际船速
    # 只考虑二维平面中速度
    speed = sqrt(vel_x ** 2 + vel_y ** 2)  # + vel_z**2)
    # 波浪影响
    # 由于环境的设置，WAVE(50, 0, 0),在静水中的time变化并不会影响实际的情况
    wave_influence = calculate_wave_influence(pos_x, pos_y, yaw, wave, time)
    # 视风
    apparent_wind = calculate_apparent_wind(yaw, vel_x, vel_y, true_wind)

    # sail angle is determined from rope length and wind direction
    # 顺/逆风 × 帆角的绝对值
    true_sail_angle = np.sign(apparent_wind.angle) * abs(sail_angle)

    # Force calculation
    # 阻尼
    damping = calculate_damping(vel_x, vel_y, yaw_rate)
    # 流体静力
    hydrostatic_force, x_hs, y_hs = calculate_hydrostatic_force(wave_influence)
    # 静水时流体静力恒为0

    # 波浪阻抗
    wave_impedance = calculate_wave_impedance(vel_x, speed)
    # 龙骨力
    rudder_force = calculate_rudder_force(speed, rudder_angle)
    # 横向力
    lateral_force, lateral_separation = calculate_lateral_force(vel_x, vel_y, speed)
    # lateral_separation = 0
    sail_force = calculate_sail_force(apparent_wind, true_sail_angle)

    """""""""
    print("sail_force", sail_force)
    print("rudder_force", rudder_force)
    print("lateral_force", lateral_force)
    print("wave_impedance", wave_impedance)
    print("lateral_separation", lateral_separation)
    """

    # Calculate changes
    # 位置一阶导数
    delta_pos_x = vel_x * cos(yaw) - vel_y * sin(yaw)
    delta_pos_y = vel_y * cos(yaw) + vel_x * sin(yaw)

    # 角度一阶导数
    delta_yaw = yaw_rate  # roll = 0, delta_yaw = yaw_rate

    # 位置二阶导数
    delta_vel_x = delta_yaw * vel_y + (
            sail_force.x + lateral_force.x + rudder_force.x + damping.x + wave_impedance + hydrostatic_force.x) / MASS

    delta_vel_y = -delta_yaw * vel_x + \
                  ((sail_force.y + lateral_force.y + rudder_force.y) +
                   hydrostatic_force.y +
                   damping.y) / MASS

    # MASS * GRAVITY + damping.z) / MASS

    # 角度二阶导数
    delta_yaw_rate = (damping.yaw
                      # + hydrostatic_force.z * hydrostatic_force.x * sin(roll)
                      - rudder_force.y * DISTANCE_COG_RUDDER
                      + sail_force.y * DISTANCE_COG_SAIL_PRESSURE_POINT
                      + sail_force.x * sin(true_sail_angle) * DISTANCE_MAST_SAIL_PRESSURE_POINT
                      + lateral_force.y * (DISTANCE_COG_KEEL_PRESSURE_POINT * (1 - lateral_separation)
                                           + DISTANCE_COG_KEEL_MIDDLE * lateral_separation)) / MOI_Z

    if actor_dynamics:
        print("b_rudder", rudder_angle)
        print("e_rudder", rudder_angle_reference)
        # actor dynamics
        # delta_rudder是船实际的舵角和本次action的期望舵角的差值
        delta_rudder = - 2 * (rudder_angle - rudder_angle_reference)
        max_rudder_speed = pi / 30
        # if delta_rudder > max_rudder_speed:
        # print delta_rudder, max_rudder_speed
        delta_rudder = np.clip(delta_rudder, -max_rudder_speed, max_rudder_speed)
        print("delta_rudder", delta_rudder)
        print("plus", rudder_angle + delta_rudder)

        delta_sail = - .1 * (sail_angle - sail_angle_reference)
        max_sail_speed = pi / 10
        delta_sail = np.clip(delta_sail, -max_sail_speed, max_sail_speed)

    # delta中，保存了船体速度和角度的一阶信息和二阶信息
    # 以全局坐标为参考系
    delta = np.array(
        [delta_pos_x, delta_pos_y, delta_yaw,
         delta_vel_x, delta_vel_y, delta_yaw_rate], dtype=object)
    if actor_dynamics:
        delta = np.concatenate((delta, [
            delta_rudder, delta_sail]))

    # delta保存所有的高阶信息
    print("delta", delta)
    # print("_________________________________________________")
    return delta


###################################################################################################
# Simulation parameters
STEPSIZE = param_dict['simulator']['stepper']['stepsize']
CLOCKRATE = param_dict['simulator']['stepper']['clockrate']

###################################################################################################
# Simulation state
INITIAL_WIND_STRENGTH = param_dict['simulator']['initial']['wind_strength']
INITIAL_WIND_DIRECTION = param_dict['simulator']['initial']['wind_direction']
INITIAL_WAVE_DIRECTION = param_dict['simulator']['initial']['wave_direction']
INITIAL_WAVE_LENGTH = param_dict['simulator']['initial']['wave_length']
INITIAL_WAVE_AMPLITUDE = param_dict['simulator']['initial']['wave_amplitude']  # 波浪振幅
INITIAL_SAIL_ANGLE = param_dict['simulator']['initial']['sail_angle']
INITIAL_RUDDER_ANGLE = param_dict['simulator']['initial']['rudder_angle']

# 初始化状态
state = initial_state()

# 环境给定
environment = [INITIAL_SAIL_ANGLE,
               INITIAL_RUDDER_ANGLE,
               TrueWind(INITIAL_WIND_STRENGTH * cos(radians(INITIAL_WIND_DIRECTION)),
                        INITIAL_WIND_STRENGTH * sin(radians(INITIAL_WIND_DIRECTION)),
                        INITIAL_WIND_STRENGTH,
                        INITIAL_WIND_DIRECTION),
               Wave(direction=INITIAL_WAVE_DIRECTION,
                    length=INITIAL_WAVE_LENGTH,
                    amplitude=INITIAL_WAVE_AMPLITUDE)]

###################################################################################################
