import random
from simulation_3DOF import *


# 生成初始状态的风
def fixed_wind():
    wind = TrueWind(0, 5, 5, pi / 2)
    return wind


def random_wind(previous_wind):
    max_speed_change = 0.1  # 最大速度变化
    max_angle_change = 0.02 * pi / 2

    # 在一定范围内随机生成速度和角度的增量
    speed_change = random.uniform(-max_speed_change, max_speed_change)
    angle_change = random.uniform(-max_angle_change, max_angle_change)

    # 应用增量到前一个时间步的风力环境值上
    new_speed = previous_wind.strength + speed_change
    new_angle = previous_wind.direction + angle_change

    # 返回新的风力环境
    return TrueWind(new_speed * cos(new_angle), new_speed * sin(new_angle), new_speed, new_angle)


def random_wind_seed(previous_wind, seed):
    random.seed(seed)  # 设置随机种子
    max_speed_change = 0.1  # 最大速度变化
    max_angle_change = 0.02 * pi / 2

    # 在一定范围内随机生成速度和角度的增量
    speed_change = random.uniform(-max_speed_change, max_speed_change)
    angle_change = random.uniform(-max_angle_change, max_angle_change)

    # 应用增量到前一个时间步的风力环境值上
    new_speed = previous_wind.strength + speed_change
    new_angle = previous_wind.direction + angle_change

    # 返回新的风力环境
    return TrueWind(new_speed * cos(new_angle), new_speed * sin(new_angle), new_speed, new_angle)


def read_wind_data(file_path):
    wind_data = []
    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            a, b, c, d = map(float, values)
            wind_data.append(TrueWind(a, b, c, d))

    return wind_data


