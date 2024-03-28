import random
import matplotlib.pyplot as plt
import numpy as np
from simulation_3DOF import *


def random_wind(previous_wind):
    max_speed_change = 0.1  # 最大速度变化
    max_angle_change = 0.05 * pi / 2  # 最大角度变化

    # 在一定范围内随机生成速度和角度的增量
    speed_change = random.uniform(-max_speed_change, max_speed_change)
    angle_change = random.uniform(-max_angle_change, max_angle_change)

    # 应用增量到前一个时间步的风力环境值上
    new_speed = previous_wind.strength + speed_change
    new_angle = previous_wind.direction + angle_change

    # 返回新的风力环境
    return TrueWind(new_speed * cos(new_angle), new_speed * sin(new_angle), new_speed, new_angle)


def generate_smooth_wind_environment(num_steps):
    wind_environment = []
    previous_wind = random_wind(TrueWind(0, 5, 5, pi / 2))  # 初始风力环境
    for _ in range(num_steps):
        new_wind = random_wind(previous_wind)
        wind_environment.append(new_wind)
        previous_wind = new_wind
    return wind_environment


num_steps = 150
smooth_wind_environment = generate_smooth_wind_environment(num_steps)

# 解压缩风速和角度数据
speed_data = [wind.strength for wind in smooth_wind_environment]
angle_data_radians = [wind.direction for wind in smooth_wind_environment]
angle_data_degrees = np.degrees(angle_data_radians)  # 将角度转换为度

# 创建时间步
time_steps = range(num_steps)

# 绘制风速图像
plt.figure(figsize=(10, 5))
plt.plot(time_steps, speed_data, color='blue')
plt.xlabel('Time Step')
plt.ylabel('Wind Speed (m/s)')
plt.title('Wind Speed over Time')
plt.grid(True)
plt.show()

# 绘制风向图像（以度为单位）
plt.figure(figsize=(10, 5))
plt.plot(time_steps, angle_data_degrees, color='green')
plt.xlabel('Time Step')
plt.ylabel('Wind Angle (degrees)')  # 设置y轴标签为角度
plt.title('Wind Angle over Time')
plt.grid(True)
plt.show()
