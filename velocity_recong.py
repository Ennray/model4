import numpy as np
import matplotlib.pyplot as plt
import math


def one_enemy_speed_calculate(enemy_base_distance, base_speed, enemy_move, scale):
    # enemy_base_distance 敌我的距离变化，enemy_move敌机在视场内的移动距离[x,z]，base_speed我机速度[x,y,z],scale是实际距离与图片中距离的比例
    t = 2  # 位移时间
    enemy_speed_x = (enemy_move[-1]-enemy_move[1]) / t
    enemy_speed_opposite = (enemy_base_distance[-1] - enemy_base_distance[1]) / t - base_speed[1]
    enemy_speed = [enemy_speed_x[0], enemy_speed_opposite, enemy_speed_x[1]]
    return enemy_speed


def enemies_speed_calculate(enemy_speed):
    sums = [sum(x) for x in zip(*enemy_speed)]
    count = len(enemy_speed)
    enemies_speed = [s / count for s in sums]
    return enemies_speed


# enemies_base_distance通过雷达得到的敌机与我方距离
def enemy_gap(enmey_posi, enemies_base_distance, scale):
    # scale是实际距离与图片中距离的比例
    # 上下间距z_gap、前后间距y_gap、左右间距x_gap
    # enmey_posi在视场的位置[[x1,z1],[x2,z2]]
    x_coords = sorted([point[0] for point in enmey_posi])
    z_coords = sorted([point[1] for point in enmey_posi])
    y_coords = sorted(enemies_base_distance)

    x_diffs = []
    y_diffs = []
    z_diffs = []
    for i in range(len(x_coords) - 1):
        if abs(x_coords[i] - x_coords[i + 1]) != 0:
            x_diffs.append(abs(x_coords[i] - x_coords[i + 1]))

    for i in range(len(y_coords) - 1):
        if abs(y_coords[i] - y_coords[i + 1]) != 0:
            y_diffs.append(abs(y_coords[i] - y_coords[i + 1]))

    for i in range(len(z_coords) - 1):
        if abs(z_coords[i] - z_coords[i + 1]) != 0:
            z_diffs.append(abs(z_coords[i] - z_coords[i + 1]))

    x_gap = sum(x_diffs) / len(x_diffs) * scale
    z_gap = sum(z_diffs) / len(z_diffs) * scale
    y_gap = sum(y_diffs) / len(y_diffs)

    return [x_gap, y_gap, z_gap]


if __name__ == "__main__":
    enemy_center = ["28:11:34.95W", "10:30:35.24N", "55.0"]
