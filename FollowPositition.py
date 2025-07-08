import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, Point
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from geopy.point import Point as geopyPoint
import math
import GeodeticConverter
import position_setting
import sympy as sp
import velocity_recong


#距离时间计算，当无人机和敌机迎面飞来时，计算无人机需要的加速、匀速、减速时间和行程
def calculate_distances_opposite(total_dist, v_start, v_max, v_end, acc, dec, v_enemy):
    # 定义符号变量
    t1, t2, t3, t4, v_tmp = sp.symbols('t1 t2 t3 t4 v_tmp')
    total_distance = (0.5 / acc * v_tmp ** 2 - 0.5 / acc * v_start ** 2 +
                      0.5 / dec * v_tmp ** 2 - 0.5 / dec * v_end ** 2 +
                      v_enemy * ((v_tmp - v_start) / acc + (v_tmp - v_end) / dec) - total_dist)
    v_tmp_solution = sp.solve(total_distance, v_tmp)
    if abs(v_tmp_solution[0]) < v_max:
        # 判断能否达到最大速度
        v_max = abs(v_tmp_solution[0])

    # 加速阶段
    t1 = (v_max - v_start) / acc
    d1 = v_start * t1 + 0.5 * acc * t1 ** 2

    # 减速阶段
    t3 = (v_max - v_end) / dec
    d3 = v_max * t3 - 0.5 * dec * t3 ** 2

    # 匀速阶段距离
    d2 = v_max * t2

    # 总距离方程
    total_distance = d1 + d2 + d3 + v_enemy * (t1 + t2 + t3) - total_dist

    # 解方程求t2
    t2_solution = sp.solve(total_distance, t2)

    return t1 + t2_solution[0] + t3

    # 计算A飞行距离/飞行时间
    # if t2_solution:
    #     t2_val = t2_solution[0]
    #     if t2_val >= 0:
    #         d2_val = Vmax * t2_val
    #         total_distance_A = d1 + d2_val + d3
    #         return total_distance_A
    #     else:
    #         return "No valid solution for t2"
    # else:
    #     return "No solution found"


#无人机和敌机同向飞行情况下，加速、匀速、减速三段运动完成给定的追击距离total_dist
def calculate_distance_same(total_dist, v_start, v_max, v_end, acc, dec, v_enemy):
    # 定义符号变量
    t1, t2, t3, t4, v_tmp = sp.symbols('t1 t2 t3 t4 v_tmp')

    total_distance = (0.5 / acc * v_tmp ** 2 - 0.5 / acc * v_start ** 2
                      + 0.5 / dec * v_tmp ** 2 - 0.5 / dec * v_end ** 2
                      - v_enemy * (v_tmp - v_start) / acc + v_enemy * (v_tmp - v_end) / dec)
    v_tmp_solution = sp.solve(total_distance, v_tmp)
    if abs(v_tmp_solution[0]) < v_max:
        # 判断能否达到最大速度
        v_max = abs(v_tmp_solution[0])

    # 加速阶段
    t1 = (v_max - v_start) / dec
    d1 = v_start * t1 + 0.5 * acc * t1 ** 2

    # 减速阶段
    t3 = (v_max - v_end) / dec
    d3 = v_max * t3 - 0.5 * dec * t3 ** 2

    # 匀速阶段距离
    d2 = v_max * t2

    # 总距离方程
    total_distance = (d1 + d2 + d3) - v_enemy * (t1 + t2 + t3) - total_dist

    # 解方程求t2
    t2_solution = sp.solve(total_distance, t2)

    return t1 + t2_solution[0] + t3

    # 计算总距离/飞行时间
    # if t2_solution:
    #     t2_val = t2_solution[0]
    #     if t2_val >= 0:
    #         d2_val = Vmax * t2_val
    #         total_distance_A = d1 + d2_val + d3
    #         return total_distance_A
    #     else:
    #         return "No valid solution for t2"
    # else:
    #     return "No solution found"


#根据敌机的速度和位置，计算无人机最终需要多移动
def calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow, distance_enemy_base):
    uavs_speed_now = uavs_speed[0]
    uavs_speed_acc = uavs_speed[1]
    uavs_speed_dec = uavs_speed[2]
    uavs_speed_max = uavs_speed[3]
    uavs_speed_turn = uavs_speed[4]

    if enemy_speed * turn_time > distance_enemy_base:
        # 先原点转弯，需加速追赶距离，最后减速过顶敌群
        distance = enemy_speed * turn_time - distance_enemy_base - distance_follow  # 追击距离
        move_t = calculate_distance_same(distance, uavs_speed_turn, uavs_speed_max,
                                         enemy_speed, uavs_speed_acc, uavs_speed_dec, enemy_speed)
        uavs_speed_now = enemy_speed
    else:
        # 先迎向加速，再减速转弯，抵近敌群
        distance_turnbegin = distance_enemy_base - enemy_speed * turn_time + distance_follow  # 相向行驶距离
        move_t = calculate_distances_opposite(distance_turnbegin, uavs_speed_now, uavs_speed_max,
                                              uavs_speed_turn, uavs_speed_acc, uavs_speed_dec, enemy_speed)
        uavs_speed_now = uavs_speed_turn

    # 变动位移 ：
    move_dist = (move_t + turn_time) * enemy_speed
    return move_dist


def find_edge_points(data):
    # 初始化边界值
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    min_z = float('inf')
    max_z = float('-inf')

    # 遍历数据
    for point in data:
        x, y, z = point[0]
        # 更新边界值
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        min_z = min(min_z, z)
        max_z = max(max_z, z)

    # 确定边缘点
    edge_points = []
    for point in data:
        x, y, z = point[0]
        if (x == min_x or x == max_x) or (y == min_y or y == max_y):
            edge_points.append(point)

    return min_x, max_x, min_y, max_y, min_z, max_z


def second_turning_position(enemy_geo, enemy_center, second_points, base, enemy_speed, uavs_speed, turn_time, distance_follow,
                 detection_size, height, y_gap):
    # enemy_geo敌机位置[敌机1，敌机2..]，second_points第二波次位置，enemy_center敌机中心位置，base第一波次中心位置，enemy_center - base表示敌机飞行方向，enemy_speed敌机速度
    # uavs_speed侦察机速度[,,]，turn_time侦察机转弯时间，distance_follow伴随侦察水平间距，detection_size伴随侦察探测区域[x,z]，height伴随侦察高度差，y_gap敌机前后间距
    A_lat, A_lon, A_alt = GeodeticConverter.decimal_dms_to_degrees(base)
    B_lat, B_lon, B_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    # 创建坐标系转换器
    converter = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)

    # 第二波次无人机坐标转换,并按y值排序
    seconds = []
    i = 0
    for second_point in second_points:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(second_point)
        second_point_local = converter.geodetic_to_local(lat, lon, alt)
        seconds.append([second_point_local, i, 0])  # 0表示未转弯
        i += 1
    seconds_sorted = sorted(seconds, key=lambda item: item[0][1], reverse=False)  # 按照y值 从小到大排序
    print(i)

    # 敌机坐标转换  注：有可能没有敌机坐标呢
    enemys = []
    enemy_center_local = converter.geodetic_to_local(B_lat, B_lon, B_alt)
    i = 0
    for enemy in enemy_geo:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(enemy)  # 将字符串转为经纬度数
        enemy = converter.geodetic_to_local(lat, lon, alt)
        enemys.append([enemy, i, 0])
        i += 1

    # 首先设置无人机在边界点的后上方(前后距离：根据探测效果确定，高度：根据期望值确定。当前前后间距是distance_follow,高度为height)
    min_x, max_x, min_y, max_y, min_z, max_z = find_edge_points(enemys)

    # 确认敌机边缘，保证边缘在视场内。
    # for i in range(edge_points_len):
    #     distance_enemy_base = edge_points[i][0][1] - edge_points[i][0][1]
    #     move_dist = calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow, distance_enemy_base)
    #     seconds_sorted[i][0] = [edge_points[i][0][0], edge_points[i][0][1] - move_dist + distance_follow, edge_points[i][0][2] + height]

    # 从第二波次最尾开始转
    # left = 0
    # right = 1
    # up = 0
    # down = 1
    # for i in range(0, len(seconds_sorted) / 4):
    #     while enemys[x_sorted_enemy[left][1]][1][2] == 1:
    #         left += 1
    #     distance_enemy_base = seconds_sorted[i][0][1] - enemys[x_sorted_enemy[left][1]][0][1]
    #     move_dist = calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow, distance_enemy_base)
    #     seconds_sorted[i][0] = [enemys[x_sorted_enemy[left][1]][0][0], move_dist + seconds_sorted[i][0][1],
    #                             enemys[x_sorted_enemy[left][1]][0][2]]
    #
    #     while enemys[x_sorted_enemy[-right][1]][2] == 1:
    #         right += 1
    #     distance_enemy_base = seconds_sorted[i + 1][0][1] - enemys[x_sorted_enemy[-right][1]][0][1]
    #     move_dist = calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow, distance_enemy_base)
    #     seconds_sorted[i + 1][0] = [enemys[x_sorted_enemy[-right][1]][0][0], move_dist + seconds_sorted[i + 1][0][1],
    #                                 enemys[x_sorted_enemy[-right][1]][0][2]]
    #
    #     while enemys[x_sorted_enemy[up][1]][2] == 1:
    #         up += 1
    #     distance_enemy_base = seconds_sorted[i + 2][0][1] - enemys[x_sorted_enemy[up][1]][0][1]
    #     move_dist = calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow, distance_enemy_base)
    #     seconds_sorted[i + 2][0] = [enemys[x_sorted_enemy[up][1]][0][0], move_dist + seconds_sorted[i + 2][0][1],
    #                                 enemys[x_sorted_enemy[up][1]][0][2]]
    #
    #     while enemys[x_sorted_enemy[down][1]][2] == 1:
    #         down += 1
    #     distance_enemy_base = seconds_sorted[i + 3][0][1] - enemys[x_sorted_enemy[-down][1]][0][1]
    #     move_dist = calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow, distance_enemy_base)
    #     seconds_sorted[i + 3][0] = [enemys[x_sorted_enemy[-down][1]][0][0], move_dist + seconds_sorted[i + 3][0][1],
    #                                 enemys[x_sorted_enemy[-down][1]][0][2]]
    #     if i == 0:
    #         move_t = move_dist / enemy_speed
    # 最大移动时间
    enemy_center_local[1] - seconds_sorted[0][1]
    move_t = turn_time + calculate_distances_opposite(enemy_center_local[1] - seconds_sorted[0][1], uavs_speed[0],
                                                      uavs_speed[3], uavs_speed[4], uavs_speed[1], uavs_speed[2],
                                                      enemy_speed)
    max_p = enemy_center_local[1] - (enemy_speed + uavs_speed[0]) * move_t  # 第二波次转弯完成，敌机飞行到达的位置

    i = 0
    n = 0
    quantity = (max_x - min_x) / detection_size[0] * 2 + (max_y - min_y) / detection_size[1] * 2 - 4
    tmp = quantity
    while i in range(len(seconds_sorted)) and seconds_sorted[i][0][1] < max_p - 200:
        # 不随第1波次转弯的无人机（避免迎头侦察时敌机丢失），200是设置的迎头探测最小距离
        if i > quantity:
            n = n + 1  # 第n圈
            quantity += tmp - 8 * n

        casei = i % 4
        num = math.floor(i/4)
        seconds_sorted[i][2] = 1  # 表示无人机已转弯
        if casei == 0:  # 上边缘
            move_dist = calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow,
                                            max_y - n * y_gap - seconds_sorted[i][0][1])  # 敌机位移变动
            seconds_sorted[i][0] = [min_x + (n + num) * detection_size[0], max_y - n * y_gap - move_dist + distance_follow,
                                    max_z - n * detection_size[1] + height]
        elif casei == 1:  # 右边缘
            move_dist = calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow,
                                            max_y - (n+num) * y_gap - seconds_sorted[i][0][1])  # 敌机位移变动
            seconds_sorted[i][0] = [max_x - n * detection_size[0],max_y - (n + num) * y_gap - move_dist + distance_follow,
                                    max_z - (n + num) * detection_size[1] + height]
        elif casei == 2:  # 下边缘
            move_dist = calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow,
                                            min_y + n * y_gap - seconds_sorted[i][0][1])  # 敌机位移变动
            seconds_sorted[i][0] = [max_x - (n + num) * detection_size[0], min_y + n * y_gap - move_dist + distance_follow,
                                    min_z + n * detection_size[1] + height]
        elif casei == 1:  # 左边缘
            move_dist = calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow,
                                            min_y + (n+num) * y_gap - seconds_sorted[i][0][1])  # 敌机位移变动
            seconds_sorted[i][0] = [min_x + n * detection_size[0], min_y + (n + num) * y_gap - move_dist + distance_follow,
                                    min_z + (n + num) * detection_size[1] + height]
        # if i < 2 * (n_x + m_y):
        #     if m == 0 and n < n_x:
        #         distance_enemy_base = max_y + m * y_gap - seconds_sorted[i][0][1]
        #         move_dist = calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow,
        #                                         distance_enemy_base)  # 敌机位移变动
        #         seconds_sorted[i][0] = [min_x + n * detection_size[0],
        #                                 max_y + m * y_gap - move_dist + distance_follow,
        #                                 min_z + height + m * detection_size[1]]
        #     elif m == -1:
        #         distance_enemy_base = max_y + m * y_gap - seconds_sorted[i][0][1]
        #         move_dist = calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow,
        #                                         distance_enemy_base)  # 敌机位移变动
        #         seconds_sorted[i][0] = [min_x + n * detection_size[0],
        #                                 max_y + m * y_gap - move_dist + distance_follow,
        #                                 min_z + height + m * detection_size[1]]

        # # 根据敌机的边缘（x,z的最值）按照x由小到达，z由小到大排列无人机
        # distance_enemy_base = max_y + m * y_gap * ratio - seconds_sorted[i][0][1]  # z_sorted_enemy[0][0][1]最上边敌机的y值
        # move_dist = calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow, distance_enemy_base) # 敌机位移变动
        # if move_dist/enemy_speed < move_t:
        #     seconds_sorted[i][0] = [min_x + n * detection_size[0]*ratio, max_y + m * y_gap - move_dist + distance_follow,
        #                             min_z + height + m * detection_size[1]*ratio]
        # n = n + 1
        # i = i + 1
    print(i)

    uavstr = []
    for uav in seconds_sorted:
        result = converter.local_to_geodetic_dms(uav[0])
        uavstr.append(result)
    return uavstr


def adjust_speed(speed, appropriate_relative_speed):
    # 追击航速调整，speed敌机速度[敌机A，敌机B...],appropriate_relative_speed:根据光电探测图像质量确定的相对速度[最小值，最大值]
    enemy_cluster_speed = velocity_recong.enemies_speed_calculate(speed)
    # 计算A的速度矢量的模
    v_mod = math.sqrt(enemy_cluster_speed[0] ** 2 + enemy_cluster_speed[1] ** 2 + enemy_cluster_speed[2] ** 2)

    vmin_base_mod = v_mod + appropriate_relative_speed[0]
    vmin_base = (v_mod / vmin_base_mod) * enemy_cluster_speed

    vmax_base_mod = v_mod + appropriate_relative_speed[0]
    vmax_base = (v_mod / vmax_base_mod) * enemy_cluster_speed
    appropriate_speed_range = [vmin_base, vmax_base]
    return appropriate_speed_range


def relative_posi(first_uavs, second_uavs):
    relative_position=[]
    for first_uav in first_uavs:
        for second_uav in second_uavs:
            relative_position.append(first_uav - second_uav)
    return relative_position


def first_turning_position(turn_points, second_uavs, ecenter, base_point, e_geo, espeed, detect_size, y_gap,
                           height, uav_speed, follow_distance):
    # e_geo敌机位置[敌机1，敌机2..]，ecenter敌机中心位置，second_uavs第二波次位置，base_point第一波次中心位置，ecenter - base_point表示敌机飞行方向,espeed敌机速度
    # uav_speed侦察机速度[,,]，turn_time侦察机转弯时间，follow_distance伴随侦察水平间距，detect_size伴随侦察探测区域[x,z]，height伴随侦察高度差，y_gap敌机前后间距
    A_lat, A_lon, A_alt = GeodeticConverter.decimal_dms_to_degrees(base_point)
    B_lat, B_lon, B_alt = GeodeticConverter.decimal_dms_to_degrees(ecenter)
    # 创建坐标系转换器
    converter = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)

    # 无人机坐标转换
    turn_uavs = []
    i = 0
    for turn_uav in turn_points:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(turn_uav)  # 将字符串转为经纬度数
        turn_uav = converter.geodetic_to_local(lat, lon, alt)
        turn_uavs.append([turn_uav, i])
        i += 1

    # 敌机坐标转换
    enemys = []
    i = 0
    for enemy in e_geo:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(enemy)  # 将字符串转为经纬度数
        enemy = converter.geodetic_to_local(lat, lon, alt)
        enemys.append([enemy, i, 0])
        i += 1
    min_x, max_x, min_y, max_y, min_z, max_z = find_edge_points(enemys)

    i = 0
    n = 0
    second_uavs_len = len(second_uavs)
    quantity = (max_x - min_x) / detect_size[0] * 2 + (max_y - min_y) / detect_size[1] * 2 - 4
    tmp = quantity
    while i in range(len(turn_uavs)):
        # 无人机转弯，由外圈向内圈占位。
        while second_uavs_len + i > quantity:
            n = n + 1  # 由边缘向中心点第n圈
            quantity += tmp - 8 * n
        casei = (second_uavs_len + i) % 4
        num = math.floor((second_uavs_len + i) / 4)
        if casei == 0:  # 上边缘
            move_dist = calculate_move_dist(espeed, uav_speed, turn_time, follow_distance,
                                            max_y - n * y_gap - turn_uavs[i][0][1])  # 敌机位移变动
            turn_uavs[i][0] = [min_x + (n + num) * detect_size[0], max_y - n * y_gap - move_dist + follow_distance,
                               max_z - n * detect_size[1] + height]
        elif casei == 1:  # 右边缘
            move_dist = calculate_move_dist(espeed, uav_speed, turn_time, follow_distance,
                                            max_y - (n+num) * y_gap - turn_uavs[i][0][1])  # 敌机位移变动
            turn_uavs[i][0] = [max_x - n * detect_size[0], max_y - (n + num) * y_gap - move_dist + follow_distance,
                               max_z - (n + num) * detect_size[1] + height]
        elif casei == 2:  # 下边缘
            move_dist = calculate_move_dist(espeed, uav_speed, turn_time, follow_distance,
                                            min_y + n * y_gap - turn_uavs[i][0][1])  # 敌机位移变动
            turn_uavs[i][0] = [max_x - (n + num) * detect_size[0], min_y + n * y_gap - move_dist + follow_distance,
                               min_z + n * detect_size[1] + height]
        elif casei == 1:  # 左边缘
            move_dist = calculate_move_dist(espeed, uav_speed, turn_time, follow_distance,
                                            min_y + (n+num) * y_gap - turn_uavs[i][0][1])  # 敌机位移变动
            turn_uavs[i][0] = [min_x + n * detect_size[0], min_y + (n + num) * y_gap - move_dist + follow_distance,
                               min_z + (n + num) * detect_size[1] + height]

    uavstr = []
    for uav in turn_uavs:
        result = converter.local_to_geodetic_dms(uav[0])
        uavstr.append(result)

    return uavstr


if __name__ == "__main__":
    enemy_geo = [
        [
            "28:11:37.48W",
            "10:31:48.11N",
            "1029.40"
        ],
        [
            "28:11:37.43W",
            "10:31:48.12N",
            "129.40"
        ],
        [
            "28:11:37.38W",
            "10:31:48.13N",
            "-770.60"
        ],
        [
            "28:11:37.33W",
            "10:31:48.14N",
            "-1670.59"
        ],
        [
            "28:11:36.84W",
            "10:31:28.59N",
            "1523.51"
        ],
        [
            "28:11:36.79W",
            "10:31:28.60N",
            "623.51"
        ],
        [
            "28:11:36.75W",
            "10:31:28.61N",
            "-276.48"
        ],
        [
            "28:11:36.70W",
            "10:31:28.61N",
            "-1176.48"
        ],
        [
            "28:11:36.65W",
            "10:31:28.62N",
            "-2076.48"
        ],
        [
            "28:11:36.19W",
            "10:31:9.07N",
            "1795.17"
        ],
        [
            "28:11:36.15W",
            "10:31:9.08N",
            "895.17"
        ],
        [
            "28:11:36.10W",
            "10:31:9.09N",
            "-4.83"
        ],
        [
            "28:11:36.05W",
            "10:31:9.09N",
            "-904.83"
        ],
        [
            "28:11:36.00W",
            "10:31:9.10N",
            "-1804.82"
        ],
        [
            "28:11:35.95W",
            "10:31:9.11N",
            "-2704.82"
        ],
        [
            "28:11:35.54W",
            "10:30:49.56N",
            "1910.13"
        ],
        [
            "28:11:35.49W",
            "10:30:49.56N",
            "1010.14"
        ],
        [
            "28:11:35.44W",
            "10:30:49.57N",
            "110.14"
        ],
        [
            "28:11:35.39W",
            "10:30:49.57N",
            "-789.86"
        ],
        [
            "28:11:35.34W",
            "10:30:49.58N",
            "-1689.86"
        ],
        [
            "28:11:35.29W",
            "10:30:49.58N",
            "-2589.86"
        ],
        [
            "28:11:34.88W",
            "10:30:30.05N",
            "1917.68"
        ],
        [
            "28:11:34.83W",
            "10:30:30.05N",
            "1017.68"
        ],
        [
            "28:11:34.78W",
            "10:30:30.05N",
            "117.68"
        ],
        [
            "28:11:34.73W",
            "10:30:30.05N",
            "-782.32"
        ],
        [
            "28:11:34.68W",
            "10:30:30.05N",
            "-1682.32"
        ],
        [
            "28:11:34.63W",
            "10:30:30.05N",
            "-2582.32"
        ],
        [
            "28:11:34.21W",
            "10:30:10.54N",
            "1886.71"
        ],
        [
            "28:11:34.16W",
            "10:30:10.54N",
            "986.71"
        ],
        [
            "28:11:34.11W",
            "10:30:10.54N",
            "86.71"
        ],
        [
            "28:11:34.06W",
            "10:30:10.53N",
            "-813.28"
        ],
        [
            "28:11:34.01W",
            "10:30:10.53N",
            "-1713.28"
        ],
        [
            "28:11:33.96W",
            "10:30:10.53N",
            "-2613.28"
        ],
        [
            "28:11:33.54W",
            "10:29:51.03N",
            "1711.82"
        ],
        [
            "28:11:33.49W",
            "10:29:51.02N",
            "811.82"
        ],
        [
            "28:11:33.44W",
            "10:29:51.02N",
            "-88.18"
        ],
        [
            "28:11:33.39W",
            "10:29:51.01N",
            "-988.17"
        ],
        [
            "28:11:33.34W",
            "10:29:51.01N",
            "-1888.17"
        ],
        [
            "28:11:32.85W",
            "10:29:31.51N",
            "1327.45"
        ],
        [
            "28:11:32.80W",
            "10:29:31.51N",
            "427.45"
        ],
        [
            "28:11:32.76W",
            "10:29:31.50N",
            "-472.55"
        ],
        [
            "28:11:32.71W",
            "10:29:31.49N",
            "-1372.54"
        ],
        [
            "28:11:32.14W",
            "10:29:11.99N",
            "444.39"
        ],
        [
            "28:11:32.09W",
            "10:29:11.98N",
            "-455.60"
        ]
    ]
    enemy_center = ["28:11:34.95W", "10:30:35.24N", "55.0"]
    second_points = [
        [
            "28:30:49.77W",
            "10:11:26.53N",
            "119.59"
        ],
        [
            "28:31:0.68W",
            "10:11:14.36N",
            "121.54"
        ],
        [
            "28:31:11.59W",
            "10:11:2.19N",
            "123.53"
        ],
        [
            "28:31:4.51W",
            "10:11:39.49N",
            "88.62"
        ],
        [
            "28:31:15.42W",
            "10:11:27.32N",
            "90.57"
        ],
        [
            "28:31:26.33W",
            "10:11:15.16N",
            "92.56"
        ],
        [
            "28:30:35.03W",
            "10:11:13.56N",
            "112.05"
        ],
        [
            "28:30:45.94W",
            "10:11:1.39N",
            "114.00"
        ],
        [
            "28:30:56.85W",
            "10:10:49.22N",
            "115.99"
        ],
        [
            "28:30:49.69W",
            "10:11:26.61N",
            "1019.58"
        ],
        [
            "28:31:0.60W",
            "10:11:14.45N",
            "1021.53"
        ],
        [
            "28:31:11.51W",
            "10:11:2.28N",
            "1023.52"
        ],
        [
            "28:31:4.43W",
            "10:11:39.57N",
            "988.62"
        ],
        [
            "28:31:15.34W",
            "10:11:27.41N",
            "990.57"
        ],
        [
            "28:31:26.25W",
            "10:11:15.24N",
            "992.55"
        ],
        [
            "28:30:49.85W",
            "10:11:26.44N",
            "-780.40"
        ],
        [
            "28:31:0.76W",
            "10:11:14.27N",
            "-778.45"
        ]
    ]
    base = ["28:30:34.95W", "10:11:35.24N", "55.0"]
    enemy_speed = 240
    uavs_speed = [180, 80, 80, 300, 200]
    turn_time = 15
    distance_follow = 50
    detection_size = [200, 300]
    print(second_turning_position(enemy_geo, enemy_center, second_points, base, enemy_speed, uavs_speed,
                                  turn_time, distance_follow, detection_size,))

