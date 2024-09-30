import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams

# 设置字体以支持中文（例如：SimHei 字体）
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False  # 避免负号显示问题

#data_dir = './HJB_PPO_logs/Boat_env/20240913/success/HJB_PPOBoat_env_log_9.csv'  # 1 6 9
data_dir = './data/comparison.csv'
data_frame = pd.read_csv(data_dir)


def plot_avg_return(input_dir):
    """
    读取CSV文件并绘制 'avg_return_in_period' 列数据的折线图

    参数：
        data_dir (str): CSV文件的路径
    """
    try:
        # 读取CSV文件
        data = pd.read_csv(input_dir)

        # 检查是否存在 'avg_return_in_period' 列
        if 'avg_return_in_period' in data.columns:
            # 绘制 'avg_return_in_period' 列数据的折线图
            plt.figure(figsize=(10, 6))
            plt.plot(data['avg_return_in_period'], label='HJB-PPO')
            plt.title('Average Reward')
            plt.xlabel('period')
            plt.ylabel('reward')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("CSV文件中未找到 'avg_return_in_period' 列")
    except FileNotFoundError:
        print(f"文件 {data_dir} 未找到，请检查路径。")
    except Exception as e:
        print(f"发生错误：{e}")


#plot_avg_return(data_dir)


# 提取数据
col1 = data_frame['HJB_PPO']
col2 = data_frame['PPO']

# 创建 x 轴的采样位置
x1 = range(0, len(col1) * 3000, 3000)  # Column 1 的 x 轴
x2 = range(0, len(col2) * 3200, 3200)  # Column 2 的 x 轴
x_combined = list(x1) + list(x2)

# 计算3000和3200的公倍数
lcm = 2 * np.lcm(3000, 3200)  # 取两倍的最小公倍数
multiples = np.arange(0, 1e6, lcm)  # 生成公倍数

# 绘制图形
plt.figure(figsize=(10, 5))
plt.plot(x1, col1, label='HJB_PPO')
plt.plot(x2, col2, label='PPO')

# 设置 x 轴的刻度为公倍数
plt.xticks(multiples)

plt.title('Average Reward')
plt.xlabel('Sampling Steps')
plt.ylabel('Values')
plt.legend()
plt.grid()
plt.show()
