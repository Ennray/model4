"""
无人机协同侦察任务中的跟随定位模块

该模块实现了多波次无人机对敌机群的协同侦察任务，包括：
1. 无人机运动规划（加速、匀速、减速）
2. 敌我双方位置实时更新和追踪
3. 无人机转弯时机预测和分组决策
4. 3D轨迹可视化

"""

import numpy as np
import matplotlib.pyplot as plt
import math
import GeodeticConverter
import sympy as sp
import velocity_recong



#计算无人机与敌机迎面飞行相遇的总时间和我方飞行距离
def calculate_distances_opposite(total_dist, v_start, v_max, v_end, acc, dec, v_enemy):
    #总距离， 起始速度， 最大速度， 结束速度， 加速度， 减速度， 敌机速度

    # 定义符号变量
    t1, t2, t3, v_tmp = sp.symbols('t1 t2 t3 v_tmp')
    
    # 求解实际最大速度
    total_distance = (0.5 / acc * v_tmp ** 2 - 0.5 / acc * v_start ** 2 +
                      0.5 / dec * v_tmp ** 2 - 0.5 / dec * v_end ** 2 +
                      v_enemy * ((v_tmp - v_start) / acc + (v_tmp - v_end) / dec) - total_dist)
    v_tmp_solution = sp.solve(total_distance, v_tmp)
    
    if abs(v_tmp_solution[0]) < v_max:
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
    t2_value = t2_solution[0]

    #总时间
    time = t1 + t2_value + t3
    total_time = float(time.evalf())


    #无人机飞行距离
    d2 = v_max * t2_value
    uav_distance_fx = d1 + d2 + d3
    uav_distance = float(uav_distance_fx.evalf())

    # 计算竖直方向爬升距离
    height_distances_max = calculate_max_vertical_distance(total_time, v_max=80, acceleration=40)

    return total_time,uav_distance,height_distances_max

def calculate_max_vertical_distance(total_time, v_max=80, acceleration=40):
    """
    计算竖直方向可以爬升的最大距离
    运动过程：匀加速 -> 匀速 -> 匀减速为零
    
    参数:
        total_time: 总飞行时间 (s)
        v_max: 竖直方向最大速度 (m/s)
        acceleration: 加速度 (m/s²)
    
    返回:
        height_distance: 最大爬升距离 (m)
    """
    # 加速到最大速度需要的时间
    t_acc = v_max / acceleration
    
    # 从最大速度减速到零需要的时间
    t_dec = v_max / acceleration  # 减速度等于加速度
    
    # 匀速飞行时间
    t_uniform = total_time - t_acc - t_dec
    
    # 如果总时间不够进行完整的三段式运动
    if t_uniform < 0:
        # 只能进行加速和减速，无法达到最大速度
        # 设加速时间为 t1，减速时间为 t2 = total_time - t1
        # 最大速度为 v_peak = acceleration * t1
        # 由于对称，t1 = t2 = total_time / 2
        t1 = total_time / 2
        v_peak = acceleration * t1
        
        # 加速段距离
        s1 = 0.5 * acceleration * t1**2
        # 减速段距离
        s2 = v_peak * t1 - 0.5 * acceleration * t1**2
        
        height_distance = s1 + s2
    else:
        # 完整的三段式运动
        # 加速段距离
        s1 = 0.5 * acceleration * t_acc**2
        
        # 匀速段距离
        s2 = v_max * t_uniform
        
        # 减速段距离
        s3 = v_max * t_dec - 0.5 * acceleration * t_dec**2
        
        height_distance = s1 + s2 + s3
    
    return height_distance

def calculate_time_to_reach_height(target_height, v_max=80, acceleration=40):
    """
    计算达到指定高度所需的最小时间
    运动过程：匀加速 -> 匀速 -> 匀减速为零
    
    参数:
        target_height: 目标爬升高度 (m)
        v_max: 竖直方向最大速度 (m/s)
        acceleration: 加速度 (m/s²)
    
    返回:
        min_time: 达到目标高度的最小时间 (s)
    """
    # 加速到最大速度需要的时间
    t_acc = v_max / acceleration
    
    # 从最大速度减速到零需要的时间
    t_dec = v_max / acceleration
    
    # 加速段距离
    s_acc = 0.5 * acceleration * t_acc**2
    
    # 减速段距离
    s_dec = v_max * t_dec - 0.5 * acceleration * t_dec**2
    
    # 仅靠加速和减速能达到的最大高度
    max_height_without_uniform = s_acc + s_dec
    
    if target_height <= max_height_without_uniform:
        # 不需要匀速段，只需要对称的加速和减速
        # 解方程：target_height = 2 * (0.5 * acceleration * t^2)
        # 其中 t 是加速时间（也等于减速时间）
        t_half = math.sqrt(target_height / acceleration)
        min_time = 2 * t_half
    else:
        # 需要匀速段
        uniform_distance = target_height - max_height_without_uniform
        t_uniform = uniform_distance / v_max
        min_time = t_acc + t_uniform + t_dec
    
    return min_time


def calculate_min_safe_vertical_distance(enemy_positions_geo, uav_positions_geo, converter, safety_margin=100):
    """
    计算竖直方向爬升的最小安全高度
    要求我方高度最小的无人机也要超过敌方高度最大的无人机
    
    参数:
        enemy_positions_geo: 敌机位置（经纬度）
        uav_positions_geo: 我方无人机位置（经纬度）
        converter: 坐标转换器
        safety_margin: 安全余量 (m)
    
    返回:
        min_climb_height: 最小爬升高度 (m)
        enemy_max_altitude: 敌机最大高度 (m)
        uav_min_altitude: 我方最小高度 (m)
    """
    # 获取敌机的最大高度
    enemy_max_altitude = float('-inf')
    for geo in enemy_positions_geo:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        enemy_max_altitude = max(enemy_max_altitude, alt)
    
    # 获取我方无人机的最小高度
    uav_min_altitude = float('inf')
    for geo in uav_positions_geo:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        uav_min_altitude = min(uav_min_altitude, alt)
    
    # 计算最小爬升高度 = 敌机最大高度 - 我方最小高度 + 安全余量
    min_climb_height = enemy_max_altitude - uav_min_altitude + safety_margin
    
    # 如果计算结果为负数，说明我方已经在足够高度，只需要安全余量
    if min_climb_height < safety_margin:
        min_climb_height = safety_margin
    
    return min_climb_height, enemy_max_altitude, uav_min_altitude

def calculate_distance_same(total_dist, v_start, v_max, v_end, acc, dec, v_enemy):
    """
    计算无人机与敌机同向飞行时的运动时间
    
    参数:
        total_dist: 追击距离 (m)
        v_start: 起始速度 (m/s)
        v_max: 最大速度 (m/s)
        v_end: 结束速度 (m/s)
        acc: 加速度 (m/s²)
        dec: 减速度 (m/s²)
        v_enemy: 敌机速度 (m/s)
    
    返回:
        总飞行时间 (s)
    """
    # 定义符号变量
    t1, t2, t3, v_tmp = sp.symbols('t1 t2 t3 v_tmp')

    total_distance = (0.5 / acc * v_tmp ** 2 - 0.5 / acc * v_start ** 2
                      + 0.5 / dec * v_tmp ** 2 - 0.5 / dec * v_end ** 2
                      - v_enemy * (v_tmp - v_start) / acc + v_enemy * (v_tmp - v_end) / dec)
    v_tmp_solution = sp.solve(total_distance, v_tmp)
    
    if abs(v_tmp_solution[0]) < v_max:
        # 判断能否达到最大速度
        v_max = abs(v_tmp_solution[0])

    # 加速阶段
    t1 = (v_max - v_start) / acc  # 修正：应该是 acc 而不是 dec
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


#根据敌机的速度和位置，计算无人机最终需要多移动多少距离来完成追击、转弯到达目标点
def calculate_move_dist(enemy_speed, uavs_speed, turn_time, distance_follow, distance_enemy_base):
    uavs_speed_now = uavs_speed[0]
    uavs_speed_acc = uavs_speed[1]
    uavs_speed_dec = uavs_speed[2]
    uavs_speed_max = uavs_speed[3]
    uavs_speed_turn = uavs_speed[4]

    if enemy_speed * turn_time > distance_enemy_base:
        # 先原点转弯，需加速追赶距离，最后减速过顶敌群（迎面情况我方飞过敌方）
        distance = enemy_speed * turn_time - distance_enemy_base - distance_follow  # 追击距离=敌机跑过距离-无人机初始到基线距离-伴飞水平距离
        move_t = calculate_distance_same(distance, uavs_speed_turn, uavs_speed_max,
                                         enemy_speed, uavs_speed_acc, uavs_speed_dec, enemy_speed)
        uavs_speed_now = enemy_speed
    else:
        # 先迎向加速，再减速转弯，抵近敌群（迎面情况我方还未接近敌方）
        distance_turnbegin = distance_enemy_base - enemy_speed * turn_time + distance_follow  # 相向行驶距离
        move_t = calculate_distances_opposite(distance_turnbegin, uavs_speed_now, uavs_speed_max,
                                              uavs_speed_turn, uavs_speed_acc, uavs_speed_dec, enemy_speed)
        uavs_speed_now = uavs_speed_turn

    # 变动位移
    move_dist = (move_t + turn_time) * enemy_speed
    return move_dist


def find_edge_points(data):
    """
    在三维点集中找出边界点的最大值和最小值
    
    参数:
        data: 包含三维坐标点的数据列表，格式为 [[point, index, status], ...]
    
    返回:
        tuple: (min_x, max_x, min_y, max_y, min_z, max_z)
    """
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

    return min_x, max_x, min_y, max_y, min_z, max_z

#对我方无人机按照靠近敌机的顺序进行排序
def sorted_y_points(second_points, converter):
    seconds = []
    i = 0
    for point in second_points:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(point)
        point_local = converter.geodetic_to_local(lat, lon, alt)
        seconds.append([point_local, i])
        i += 1
    seconds_sorted = sorted(seconds, key=lambda item: item[0][1], reverse=True)#按y轴排序


    return seconds_sorted











#计算第二波次无人机在跟随敌机运行后，如何根据敌机位置和队形，确定每个无人机转弯后的位置
def second_turning_position(enemy_geo, enemy_center, second_points, base, enemy_speed, uavs_speed, turn_time, distance_follow,
                 detection_size, height, y_gap):
    # enemy_geo敌机位置[敌机1，敌机2..]，second_points第二波次位置，enemy_center敌机中心位置，base第一波次中心位置，enemy_center - base表示敌机飞行方向，enemy_speed敌机速度
    # uavs_speed侦察机速度[,,]，turn_time侦察机转弯时间，distance_follow伴随侦察水平间距，detection_size伴随侦察探测区域[x,z]，height伴随侦察高度差，y_gap敌机前后间距
    #把经纬度转换为局部坐标系
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
    seconds_sorted = sorted(seconds, key=lambda item: item[0][1], reverse=False)  # 按照y值从小到大排序
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
    #

    #无人机与敌群中心在y方向上的相对距离
    enemy_center_local[1] - seconds_sorted[0][1]
    #转弯时间+对头飞行所需的时间（保证无人机赶到前面）
    move_t = turn_time + calculate_distances_opposite(enemy_center_local[1] - seconds_sorted[0][1], uavs_speed[0],
                                                      uavs_speed[3], uavs_speed[4], uavs_speed[1], uavs_speed[2],
                                                      enemy_speed)
    #敌机在无人机完成转弯、飞行后的位置
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

        casei = i % 4 #确定无人机排在哪条边缘上（上下左右）
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
        elif casei == 3:  # 左边缘
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
    #转回经纬度
    for uav in seconds_sorted:
        result = converter.local_to_geodetic_dms(uav[0])
        uavstr.append(result)
    return uavstr

#根据敌机速度，计算无人机适当跟飞速度范围
def adjust_speed(speed, appropriate_relative_speed):
    # 追击航速调整，speed敌机速度[敌机A，敌机B...],appropriate_relative_speed:根据光电探测图像质量确定的相对速度[最小值，最大值]
    enemy_cluster_speed = velocity_recong.enemies_speed_calculate(speed)
    # 计算A的速度矢量的模
    v_mod = math.sqrt(enemy_cluster_speed[0] ** 2 + enemy_cluster_speed[1] ** 2 + enemy_cluster_speed[2] ** 2)

    #计算无人机最小跟随速度
    vmin_base_mod = v_mod + appropriate_relative_speed[0]
    vmin_base = (v_mod / vmin_base_mod) * enemy_cluster_speed

    #计算无人机最大跟随速度
    vmax_base_mod = v_mod + appropriate_relative_speed[1]
    vmax_base = (v_mod / vmax_base_mod) * enemy_cluster_speed
    appropriate_speed_range = [vmin_base, vmax_base]
    return appropriate_speed_range


#计算两组无人机之间的相对位置差
def relative_posi(first_uavs, second_uavs):
    relative_position=[]
    for first_uav in first_uavs:
        for second_uav in second_uavs:
            relative_position.append(first_uav - second_uav)
    return relative_position


#计算第一波次无人机完成转弯动作后，如何根据敌机群当前分布以及第二波无人机数量，在外围环绕敌机群形成新的占位布局（封锁圈或侦查圈）
def first_turning_position(turn_points, second_uavs, ecenter, base_point, e_geo, espeed, detect_size, y_gap,
                           height, uav_speed, follow_distance):
    # e_geo敌机位置[敌机1，敌机2..]，ecenter敌机中心位置，second_uavs第二波次位置，base_point第一波次中心位置，ecenter敌机群中心位置 - base_point表示敌机飞行方向,espeed敌机速度
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
    #计算敌机群边界（包围圈）
    min_x, max_x, min_y, max_y, min_z, max_z = find_edge_points(enemys)

    i = 0
    n = 0
    second_uavs_len = len(second_uavs)#第二波无人机数量
    quantity = (max_x - min_x) / detect_size[0] * 2 + (max_y - min_y) / detect_size[1] * 2 - 4 #边缘一圈最多可容纳的无人机数
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
        elif casei == 3:  # 左边缘
            move_dist = calculate_move_dist(espeed, uav_speed, turn_time, follow_distance,
                                            min_y + (n+num) * y_gap - turn_uavs[i][0][1])  # 敌机位移变动
            turn_uavs[i][0] = [min_x + n * detect_size[0], min_y + (n + num) * y_gap - move_dist + follow_distance,
                               min_z + (n + num) * detect_size[1] + height]

    uavstr = []
    for uav in turn_uavs:
        result = converter.local_to_geodetic_dms(uav[0])
        uavstr.append(result)

    return uavstr



#1.1.1.2

#计算方向确定敌我前进方向
def calculate_direction_vector(enemy_center_geo, our_center_geo, converter):
    # 转换敌机中心
    lat_e, lon_e, alt_e = GeodeticConverter.decimal_dms_to_degrees(enemy_center_geo)
    enemy_local = converter.geodetic_to_local(lat_e, lon_e, alt_e)

    # 转换我方中心
    lat_o, lon_o, alt_o = GeodeticConverter.decimal_dms_to_degrees(our_center_geo)
    our_local = converter.geodetic_to_local(lat_o, lon_o, alt_o)

    # 差值向量（指向敌机）
    delta = enemy_local - our_local

    # 单位化
    norm = np.linalg.norm(delta)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0])  # 避免除零
    direction_unit = delta / norm

    return direction_unit

#根据相遇时间获得敌我两方的位置
def update_positions_geo(enemy_positions_geo, uav_positions_geo, enemy_center_geo, our_center_geo,
                        enemy_speed, converter, dt, uav_distances, height_distances):
    #敌方经纬度，我方经纬度，敌方中心，我方中心，敌方速度，对象，我方预测飞行距离

    #计算敌我两方的方向向量
    direction_unit = calculate_direction_vector(enemy_center_geo, our_center_geo, converter)
    
    updated_enemy_positions = []
    updated_uav_positions = []

    # 敌群经纬度转坐标系
    for geo in enemy_positions_geo:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        pos_local = converter.geodetic_to_local(lat, lon, alt)
        
        # 敌机沿相反方向飞行
        new_pos_local = pos_local - direction_unit * enemy_speed * dt
        new_geo = converter.local_to_geodetic_dms(new_pos_local)
        updated_enemy_positions.append(new_geo)

    #更新敌方中心坐标
    enemy_center_lat, enemy_center_lon, enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center_geo)
    pos_enemy_center = converter.geodetic_to_local(enemy_center_lat, enemy_center_lon, enemy_center_alt)

    new_pos_enemy_center = pos_enemy_center - direction_unit * enemy_speed * dt
    new_geo_enemy_center = converter.local_to_geodetic_dms(new_pos_enemy_center)

    # 我方经纬度转坐标系
    for idx, geo in enumerate(uav_positions_geo):
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        pos_local = converter.geodetic_to_local(lat, lon, alt)
        
        # 我方无人机沿正方向飞行
        pos_local[1] = pos_local[1] + uav_distances
        # 竖直方向爬升
        pos_local[0] = pos_local[0] - height_distances -200
        new_pos_local = pos_local
        new_geo = converter.local_to_geodetic_dms(new_pos_local)
        updated_uav_positions.append(new_geo)

    # 更新我方中心坐标
    uav_center_lat, uav_center_lon, uav_center_alt = GeodeticConverter.decimal_dms_to_degrees(our_center_geo)
    pos_uav_center = converter.geodetic_to_local(uav_center_lat, uav_center_lon, uav_center_alt)

    pos_uav_center[1] = pos_uav_center[1] + uav_distances
    pos_uav_center[0] = pos_uav_center[0] - height_distances -200
    new_pos_uav_center = pos_uav_center
    new_geo_uav_center = converter.local_to_geodetic_dms(new_pos_uav_center)


    return updated_enemy_positions, updated_uav_positions, new_geo_enemy_center, new_geo_uav_center

#对第二波次进行分类，按照与base的阈值分为随第1波次转弯以及不随第1波次转弯，而是自己按最优航迹开始转弯，记录类别
def classify_uav_turning(uav_positions_geo, base_geo, converter, distance_threshold):
    # 我方无人机经纬度、base、对象、阈值
    # 将base转换为局部坐标
    base_lat, base_lon, base_alt = GeodeticConverter.decimal_dms_to_degrees(base_geo)
    base_local = converter.geodetic_to_local(base_lat, base_lon, base_alt)

    decisions = []
    for geo in uav_positions_geo:
        # 将无人机位置转换为局部坐标
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        pos_local = converter.geodetic_to_local(lat, lon, alt)
        print(f"第{geo}个无人机的坐标:{pos_local}")
        
        # 计算到base的距离
        dist = np.linalg.norm(np.array(pos_local) - np.array(base_local))
        
        # 判定转弯策略
        if dist <= distance_threshold:
            decisions.append(1)  # 跟随第一波转弯
        else:
            decisions.append(0)  # 独立转弯

    return decisions

#判断第1波无人机何时即将与敌群交错，以便得到转弯时机
def predict_cross_time(base_geo, enemy_center_geo,
                       base_speed, enemy_speed,
                       converter, threshold_dist=500, dt=1.0, max_time=300):
    #base， 敌群中心坐标， 第1波无人机速度， 敌群速度， 对象， 距离阈值， 单位时间， 最大模拟时间（防止死循环）

    # 转换初始坐标为局部
    b_lat, b_lon, b_alt = GeodeticConverter.decimal_dms_to_degrees(base_geo)
    base_local = np.array(converter.geodetic_to_local(b_lat, b_lon, b_alt))

    e_lat, e_lon, e_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center_geo)
    enemy_local = np.array(converter.geodetic_to_local(e_lat, e_lon, e_alt))

    # 计算方向向量
    direction = enemy_local - base_local
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction_unit = direction / norm
    else:
        direction_unit = np.zeros(3)

    time_elapsed = 0
    while time_elapsed < max_time:
        # 每1秒更新第1波无人机的位置
        base_local = base_local + direction_unit * base_speed * dt
        # 每1秒更新敌群中心的位置
        enemy_local = enemy_local - direction_unit * enemy_speed * dt

        # 计算当前距离
        dist = np.linalg.norm(enemy_local - base_local)

        if dist <= threshold_dist:
            return time_elapsed + dt, base_local, enemy_local

        time_elapsed += dt

    # 如果超过最大时间还没满足条件，返回 None
    return None, base_local, enemy_local


#在预测转弯时间点获取无人机实时位置（起点）并计算转弯终点（占位位置）
def get_uav_turning_points(enemy_positions_geo, enemy_center_geo,
                           uav_positions_geo, base_geo,
                           enemy_speed, uav_speed_scalar,
                           uavs_speed_params,
                           converter,
                           predicted_time,
                           turn_time, distance_follow, detection_size, height, y_gap,
                           dt=1.0):
    #敌群经纬度、敌群中心、我方经纬度、base位置、敌群速度、我方速度、我方速度参数列表、对象、转弯时间、无人机跟随最短距离、探测大小、高度、落差、时间片

    # 生成无人机速度列表（每架无人机一个值180）
    uav_speeds_list = [uav_speed_scalar for _ in uav_positions_geo]

    # 更新次数（目前默认1秒更新一次）
    steps = int(predicted_time / dt)

    #调用敌我双方经纬度实时更新函数，获得敌我双方即将拐弯时的经纬度
    for _ in range(steps):
            enemy_positions_geo, uav_positions_geo = update_positions_geo(
            enemy_positions_geo,
            uav_positions_geo,
            enemy_center_geo,
            base_geo,
            enemy_speed,
            uav_speeds_list,
            converter,
            dt
        )

    final_positions_geo = second_turning_position(
        enemy_positions_geo,
        enemy_center_geo,
        uav_positions_geo,
        base_geo,
        enemy_speed,
        uavs_speed_params,
        turn_time,
        distance_follow,
        detection_size,
        height,
        y_gap
    )

    return uav_positions_geo, final_positions_geo

"""
def plot_moving_3D(enemy_positions_geo, uav_positions_geo, enemy_center_geo, our_center_geo,
                   enemy_speed, uav_speed_scalar, converter, total_time=60, dt=1.0):

    steps = int(total_time / dt)
    uav_speeds = [uav_speed_scalar for _ in uav_positions_geo]

    # 只取前10架敌机和前10架我方无人机进行可视化
    enemy_positions_geo = enemy_positions_geo[:10]
    uav_positions_geo = uav_positions_geo[:10]
    uav_speeds = uav_speeds[:10]

    # 轨迹记录
    enemy_tracks = [[] for _ in range(len(enemy_positions_geo))]
    uav_tracks = [[] for _ in range(len(uav_positions_geo))]

    for t in range(steps):
        print(f"Step {t+1}, time {t+1}s")

        # 更新位置
        enemy_positions_geo, uav_positions_geo, new_enemy_center, new_uav_center = update_positions_geo(
            enemy_positions_geo,
            uav_positions_geo,
            enemy_center_geo,
            our_center_geo,
            enemy_speed,
            converter,
            dt
        )

        print(f"敌群:{enemy_positions_geo}, 我方:{uav_positions_geo}")

        # 敌机局部坐标记录
        for i, geo in enumerate(enemy_positions_geo):
            lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
            local = converter.geodetic_to_local(lat, lon, alt)
            enemy_tracks[i].append(local)

        # 我方无人机局部坐标记录
        for i, geo in enumerate(uav_positions_geo):
            lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
            local = converter.geodetic_to_local(lat, lon, alt)
            uav_tracks[i].append(local)

    # 绘制3D轨迹图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制敌机轨迹
    for i, track in enumerate(enemy_tracks):
        xs = [p[0] for p in track]
        ys = [p[1] for p in track]
        zs = [p[2] for p in track]
        ax.plot(xs, ys, zs, label=f'Enemy {i}', linestyle='--', marker='x', alpha=0.7)

    # 绘制我方无人机轨迹
    for i, track in enumerate(uav_tracks):
        xs = [p[0] for p in track]
        ys = [p[1] for p in track]
        zs = [p[2] for p in track]
        ax.plot(xs, ys, zs, label=f'UAV {i}', linestyle='-', marker='o', alpha=0.7)

    ax.set_xlabel('Local X (m)')
    ax.set_ylabel('Local Y (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('3D Trajectories of Enemy and UAV (Step by Step)')
    ax.legend()
    plt.tight_layout()
    plt.show()
"""

def plot_positions_with_centers(enemy_positions_geo_init, uav_positions_geo_init,
                                enemy_positions_geo_new, uav_positions_geo_new,
                                enemy_center_geo_init, our_center_geo_init,
                                enemy_center_geo_new, our_center_geo_new,
                                converter):
    """
    绘制初始 & 新位置的3D分布以及中心位置
    """
    import numpy as np

    # ----------------------------------
    # 初始敌机点
    enemy_init_lats, enemy_init_lons, enemy_init_alts = [], [], []
    for geo in enemy_positions_geo_init:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        enemy_init_lats.append(lat)
        enemy_init_lons.append(lon)
        enemy_init_alts.append(alt)

    # 初始我方点
    uav_init_lats, uav_init_lons, uav_init_alts = [], [], []
    for geo in uav_positions_geo_init:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        uav_init_lats.append(lat)
        uav_init_lons.append(lon)
        uav_init_alts.append(alt)

    # ----------------------------------
    # 新敌机点
    enemy_new_lats, enemy_new_lons, enemy_new_alts = [], [], []
    for geo in enemy_positions_geo_new:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        enemy_new_lats.append(lat)
        enemy_new_lons.append(lon)
        enemy_new_alts.append(alt)

    # 新我方点
    uav_new_lats, uav_new_lons, uav_new_alts = [], [], []
    for geo in uav_positions_geo_new:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        uav_new_lats.append(lat)
        uav_new_lons.append(lon)
        uav_new_alts.append(alt)

    # ----------------------------------
    # 初始中心（已知）
    enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center_geo_init)
    our_center_init_lat, our_center_init_lon, our_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(our_center_geo_init)

    # 新中心（已知）
    enemy_center_new_lat, enemy_center_new_lon, enemy_center_new_alt = GeodeticConverter.decimal_dms_to_degrees(
        enemy_center_geo_new)
    uav_center_new_lat, uav_center_new_lon, uav_center_new_alt = GeodeticConverter.decimal_dms_to_degrees(
        our_center_geo_new)


    # ----------------------------------
    # 绘图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 初始敌机
    ax.scatter(enemy_init_lons, enemy_init_lats, enemy_init_alts, c='red', marker='x', label='Enemy Init', s=50)

    # 初始我方
    ax.scatter(uav_init_lons, uav_init_lats, uav_init_alts, c='blue', marker='o', label='UAV Init', s=50)

    # 新敌机
    ax.scatter(enemy_new_lons, enemy_new_lats, enemy_new_alts, c='orange', marker='x', label='Enemy New', s=50)

    # 新我方
    ax.scatter(uav_new_lons, uav_new_lats, uav_new_alts, c='cyan', marker='o', label='UAV New', s=50)

    # 初始中心
    ax.scatter(enemy_center_init_lon, enemy_center_init_lat, enemy_center_init_alt, c='purple', marker='^', s=120, label='Enemy Center Init')
    ax.scatter(our_center_init_lon, our_center_init_lat, our_center_init_alt, c='green', marker='^', s=120, label='Our Center Init')

    # 新中心
    ax.scatter(enemy_center_new_lon, enemy_center_new_lat, enemy_center_new_alt, c='magenta', marker='*', s=160, label='Enemy Center New')
    ax.scatter(uav_center_new_lon, uav_center_new_lat, uav_center_new_alt, c='lime', marker='*', s=160, label='Our Center New')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('3D Positions: Initial & New with Centers')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # 打印坐标信息
    print("=== 初始中心坐标 ===")
    print(f"Enemy Center Init: lat={enemy_center_init_lat:.6f}, lon={enemy_center_init_lon:.6f}, alt={enemy_center_init_alt:.1f}")
    print(f"Our Center Init:   lat={our_center_init_lat:.6f}, lon={our_center_init_lon:.6f}, alt={our_center_init_alt:.1f}")

    print("\n=== 新中心坐标（计算） ===")
    print(f"Enemy Center New: lat={enemy_center_new_lat:.6f}, lon={enemy_center_new_lon:.6f}, alt={enemy_center_new_alt:.1f}")
    print(f"Our Center New:   lat={uav_center_new_lat:.6f}, lon={uav_center_new_lon:.6f}, alt={uav_center_new_alt:.1f}")


# ================================
# 主函数和测试代码
# ================================

if __name__ == "__main__":
    """
    主函数：演示多波次无人机协同侦察任务
    """
    # 敌机数据（经纬度格式）
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

    #第二波次无人机初始点
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
    #第一波次无人机基点
    base = ["28:30:34.95W", "10:11:35.24N", "55.0"]
    # 敌机群中心位置
    enemy_center = ["28:11:34.95W", "10:30:35.24N", "55.0"]
    enemy_speed = 240
    uavs_speed = [180, 80, 80, 300, 200] #当前、加速、减速、最大、转弯速度
    turn_time = 15
    distance_follow = 50
    detection_size = [200, 300]
    height = 500
    y_gap = 200
    dt = 1
    base_distance_threshold = 1000 #base距离阈值
    turn_distance_threshold = 500 #迎面距离阈值
    #print(second_turning_position(enemy_geo, enemy_center, second_points, base, enemy_speed, uavs_speed,
                                 # turn_time, distance_follow, detection_size, height, y_gap))

    # 创建局部坐标转换器
    A_lat, A_lon, A_alt = GeodeticConverter.decimal_dms_to_degrees(base)
    B_lat, B_lon, B_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    converter = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)

    uav_speed = [180 for _ in second_points]

    #对第2波次无人机进行分类，一部分延迟转弯，一部分独立转弯
    decision_array = classify_uav_turning(second_points, base, converter, base_distance_threshold)


    second_iuav_points = []
    second_duav_points = []
    # 打印结果
    for idx, val in enumerate(decision_array):
        if val == 1:
            print(f"无人机 {idx}: 距离近，跟随第一波转弯 (1)")
            second_duav_points.append(second_points[idx])
        else:
            print(f"无人机 {idx}: 距离远，独立转弯 (0)")
            second_iuav_points.append(second_points[idx])

    # 得到按y轴排序的我方无人机队列
    sort_second_points = sorted_y_points(second_points,converter)
    print(f"纵队最前方无人机:{sort_second_points}")
    #获得敌方中心转坐标系的点位
    enemy_center_local = converter.geodetic_to_local(B_lat, B_lon, B_alt)

    # #获得我方第一波次坐标系点位
    # uav_first_local = converter.geodetic_to_local(A_lat, A_lon, A_alt)
    # print(f"我方第一波次无人机位置:{uav_first_local[1]}")

    #获得预测时间和我方预期飞行距离
    predict_time, predict_distance, height_distances_max = calculate_distances_opposite(enemy_center_local[1], uavs_speed[0],
                                                                  uavs_speed[3], uavs_speed[4], uavs_speed[1], uavs_speed[2], enemy_speed) #因为原点是0，只需要输入敌方距离即可判定转弯时机

    print(f"预期飞行时间:{predict_time} 预期飞行距离:{predict_distance}")

    # 计算最小安全爬升高度
    print(f"\n=== 最小安全爬升高度计算 ===")
    min_climb_height, enemy_max_alt, uav_min_alt = calculate_min_safe_vertical_distance(
        enemy_geo, second_iuav_points, converter, safety_margin=100
    )
    print(f"敌机最大高度: {enemy_max_alt:.2f} 米")
    print(f"我方无人机最小高度: {uav_min_alt:.2f} 米") 
    print(f"要求最小爬升高度: {min_climb_height:.2f} 米")

    height_distances = min(height_distances_max, min_climb_height)

    print(f"爬升高度为: {height_distances:.2f} 米")

    #获得相遇时敌方的坐标，我方的坐标，敌方的中心，我方的中心
    new_enemy_points, new_uav_points, new_enemy_center, new_uav_center = update_positions_geo(enemy_geo, second_iuav_points, enemy_center, base,
                                                                                              enemy_speed, converter, predict_time, predict_distance, height_distances)

    print("敌机相遇位置:")
    for e in new_enemy_points:
        print(e)

    print("\n无人机转弯位置:")
    for u in new_uav_points:
        print(u)

    # 只调用一次，画初始位置
    plot_positions_with_centers(enemy_geo, second_points, new_enemy_points, new_uav_points, enemy_center, base, new_enemy_center, new_uav_center, converter)






    # plot_moving_3D(
    #     enemy_geo,
    #     second_points,
    #     enemy_center,
    #     base,
    #     enemy_speed,
    #     uavs_speed[0],
    #     converter,
    #     toal_time,
    #     dt
    # )
    # uav_start_points_local, uav_end_points_local = get_uav_turning_points(
    #     enemy_geo,
    #     enemy_center,
    #     second_points,
    #     base,
    #     enemy_speed,
    #     uavs_speed[0],  # 无人机飞行速度标量
    #     uavs_speed,  # 无人机速度参数列表
    #     converter,
    #     predicted_time,
    #     turn_time,
    #     distance_follow,
    #     detection_size,
    #     height,
    #     y_gap,
    #     dt
    # )



    # # 打印起点和终点
    # for idx in range(len(uav_start_points_local)):
    #     print(f"无人机 {idx}: 起点 local = {uav_start_points_local[idx]}, 终点 local = {uav_end_points_local[idx]}")
    # dt = 5  # 每 5 秒更新
    # total_time = 120  # 总共模拟时间（秒）
    # steps = total_time // dt
    # run_and_plot_tracking_3D(
    #     enemy_geo,
    #     second_points,
    #     enemy_center,
    #     base,
    #     enemy_speed,
    #     uavs_speed[0],
    #     dt,
    #     steps,
    #     converter
    # )

    # enemy_positions_geo_new, uav_positions_geo_new,enemy_center,base = update_positions_geo(enemy_geo, second_points, enemy_center, base, enemy_speed, uav_speed, converter, 117)
    #
    # print("敌机新位置:")
    # for e in enemy_positions_geo_new:
    #     print(e)
    #
    # print("\n无人机新位置:")
    # for u in uav_positions_geo_new:
    #     print(u)


