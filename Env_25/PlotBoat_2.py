import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


def plot_images_with_auto_layout(image_paths, x_label="Image", save_dir=None):
    num_images = len(image_paths)

    cols = math.ceil(math.sqrt(num_images))  # 列数
    rows = math.ceil(num_images / cols)  # 行数

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))  # 4x4
    axes = axes.flatten() if num_images > 1 else [axes]  # 将axes展平成一维，如果只有一张图保持二维

    # 遍历每张图像并绘制
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)  # 读取图像
        axes[i].imshow(img)  # 在子图中显示图像
        axes[i].set_title(f'({chr(97 + i)})')
        axes[i].axis('off')

    # 对于剩余的空白子图，关闭显示
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()  # 调整布局
    if save_dir:
        plt.savefig(save_dir)  # 保存拼接后的图像
    else:
        plt.show()  # 显示图像

"""""""""
path_1 = './data/rw0_end2010_act.png'
path_2 = './data/rw1_end4020_act.png'
path_3 = './data/rw3_end5030_act.png'
path_4 = './data/rw4_end6025_act.png'
path_5 = './data/rw2_end6030_act.png'
path_6 = './data/rw2_end7035_act.png'


plot_images_with_auto_layout([path_1, path_2, path_3, path_4, path_5, path_6])
"""
path_1 = './data/01.png'
path_2 = './data/02.png'
plot_images_with_auto_layout([path_1, path_2])

