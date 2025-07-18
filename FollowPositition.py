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

from numpy.ma.core import remainder
from requests.packages import target

import GeodeticConverter
import sympy as sp
import velocity_recong
import outdata
import dataset



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
    move_dist = (move_t[0] + turn_time) * enemy_speed
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
        seconds.append([point_local, i, 0])
        i += 1
    seconds_sorted = sorted(seconds, key=lambda item: item[0][1], reverse=True)#按y轴排序
    return seconds_sorted





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



#==============================================4.1.1.2===============================================

#1. 对第二波次进行分类，按照与base的阈值分为随第1波次转弯以及不随第1波次转弯，而是自己按最优航迹开始转弯，记录类别
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


# 2. 对于独立转弯或是延迟转弯点无人机均需要计算最小安全爬升高度
def calculate_min_safe_vertical_distance(enemy_positions_geo, uav_positions_geo, converter, safety_margin=100):
    # 获取敌机的最大高度
    enemy_max_altitude = float('-inf')
    for geo in enemy_positions_geo:
        enemy_max_altitude = max(enemy_max_altitude, float(geo[2]))

    # 获取我方无人机的最小高度
    uav_min_altitude = float('inf')
    for geo in uav_positions_geo:
        uav_min_altitude = min(uav_min_altitude, float(geo[2]))
    # 计算最小爬升高度 = 敌机最大高度 - 我方最小高度 + 安全余量
    min_climb_height = enemy_max_altitude - uav_min_altitude + safety_margin

    # 如果计算结果为负数，说明我方已经在足够高度，只需要安全余量
    if min_climb_height < safety_margin:
        min_climb_height = safety_margin

    return min_climb_height, enemy_max_altitude, uav_min_altitude


#3. 计算方向确定敌我前进方向（按理说其实就是沿着y轴前进，但保险起见还是转坐标系算一下）
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


#4. 占位按照每一层每一圈进行，当前函数计算当前层能排布的无人机最大圈数
def calculate_max_rings(total, a, b):
    n = 0
    sum_uav = 0

    while True:
        n += 1
        total_n = 2*a + 2*b - 8*n + 4
        if total_n <= 0:
            break
        if sum_uav + total_n > total:
            break
        sum_uav += total_n

    if n == 1:
        return n
    else:
        return n - 1


#5. 分配无人机占位是按照包围圈进行的，当前函数是分配一圈无人机的策略
def allocate_one_ring(flag, angle_deg, tmp_num, quantity_this_ring, coord_list,
                      min_x, max_x, min_y, max_y, min_z, max_z,
                      quantity_a, quantity_b, m):

    #print("quantity_this_ring", quantity_this_ring)
    for i in range(quantity_this_ring ):
        casei = i % 4
        num = i// 4
        #print("i and casei and num", i,casei,num)
        if flag == 1 or flag == 2: #只要flag不为0，那么说明某条边分配满了，顺延分配到下一条边
            casei += 1
        # 更新 flag
        if num + 1 >= quantity_a and num + 1 < quantity_b: #如果出现上下边能分配无人机大于左右边且左右边分配满了，那么flag为1
            flag = 1
        elif num + 1 >= quantity_b and num + 1 < quantity_a: #如果出现左右边能分配无人机大于上下边且上下边分配满了，那么flag为2
            flag = 2
        else:
            flag = 0

        success = False

        if casei == 0 and flag != 1:  # 右边
            target_x = min_x + detection_size[0] * 0.5 + num * detection_size[0]
            target_y = max_y + 1000 * (m + 1) * math.cos(math.radians(angle_deg)) - detection_size[0] * 0.5 * math.sin(math.radians(angle_deg))
            target_z = max_z - 1000 * (m + 1) * math.sin(math.radians(angle_deg)) - detection_size[0] * 0.5 * math.cos(math.radians(angle_deg))
            success = True
        elif casei == 1 and flag != 2:  # 下边
            target_x = max_x - detection_size[0] * 0.5
            target_y = max_y + 1000 * (m + 1) * math.cos(math.radians(angle_deg)) - detection_size[0] * 0.5 * math.sin(math.radians(angle_deg)) - num * 2000 * math.sin(math.radians(angle_deg))
            target_z = max_z - 1000 * (m + 1) * math.sin(math.radians(angle_deg)) - detection_size[0] * 0.5 * math.cos(math.radians(angle_deg)) - num * 2000 * math.cos(math.radians(angle_deg))
            success = True
        elif casei == 2 and flag != 1:  # 左边
            target_x = max_x - detection_size[0] * 0.5 - num * detection_size[0]
            target_y = min_y + 1000 * (m + 1) * math.cos(math.radians(angle_deg)) + detection_size[0] * 0.5 * math.sin(math.radians(angle_deg))
            target_z = min_z - 1000 * (m + 1) * math.sin(math.radians(angle_deg)) + detection_size[0] * 0.5 * math.cos(math.radians(angle_deg))
            success = True
        elif casei == 3 and flag != 2:  # 上边
            target_x = min_x + detection_size[0] * 0.5
            target_y = min_y + 1000 * (m + 1) * math.cos(math.radians(angle_deg)) + detection_size[0] * 0.5 * math.sin(math.radians(angle_deg)) + num * 2000 * math.sin(math.radians(angle_deg))
            target_z = min_z - 1000 * (m + 1) * math.sin(math.radians(angle_deg)) + detection_size[0] * 0.5 * math.cos(math.radians(angle_deg)) + num * 2000 * math.sin(math.radians(angle_deg))
            success = True

        if success:
            coord_list.append((target_x, target_y, target_z))
            tmp_num += 1
            print(f"占位第 {tmp_num} 个")

       # print(f"bianjiezhishi:{min_x, max_x, min_y, max_y, min_z, max_z}")

    return coord_list


#6. 占位总分配策略，包括如何分配哪一层，如何分配哪一圈
def allocation_policy_all(total, min_x, max_x, min_y, max_y, min_z, max_z,
                          angle_deg, quantity_a, quantity_b):

    coord_list = []
    tmp_num_total = 0
    flag = 0

    #print(f"*******total:{total}")
    m = math.ceil(total / (quantity_a * quantity_b))


    for j in range(m):
        #放多少层
        min_x_n = min_x
        max_x_n = max_x
        min_y_n = min_y
        max_y_n = max_y
        min_z_n = min_z
        max_z_n = max_z
        remainder_m = total - tmp_num_total
        #print(f"<UNK>{remainder_m}<UNK>")
        remainder_n = remainder_m
        n = calculate_max_rings(remainder_m, quantity_a, quantity_b)
        #print("max_n", n)
        for i in range(n):
            # 本圈还需要放多少个
            #print(f"i{i}")
            quantity_this_ring = 2 * quantity_a + 2 * quantity_b - 8 * (i + 1) + 4

            #print("remainder_n", remainder_n)

            if remainder_n < quantity_this_ring:
                quantity_this_ring = remainder_n

            # 本圈开始时的 tmp_num
            tmp_num = 0

            # 调用分配一圈函数
            coord_list = allocate_one_ring(flag, angle_deg, tmp_num, quantity_this_ring,
                                           coord_list, min_x_n, max_x_n, min_y_n, max_y_n, min_z_n, max_z_n,
                                           quantity_a, quantity_b, j)
            flag = 0

            tmp_num_total = len(coord_list)

            # 更新边界（内缩）
            min_x_n += detection_size[0]
            max_x_n -= detection_size[0]
            min_y_n += detection_size[1]
            max_y_n -= detection_size[1]
            min_z_n += detection_size[0]
            max_z_n -= detection_size[0]

            print(f"当前总数: {tmp_num_total}, 边界更新: min_x={min_x}, max_x={max_x}")

        remainder_n = remainder_m - quantity_this_ring

    return coord_list


#7. 确定第二波次无人机转弯占位位置，要考虑敌群在转弯时的补偿距离
def second_turning_position(enemy_geo, enemy_center, second_iuav_points, enemy_speed, turn_time,
                            distance_follow, detection_size, height, y_gap, converter):


    #坐标转换（无人机当前位置）
    seconds = []
    total = 0
    for i, point in enumerate(second_iuav_points):
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(point)
        local = converter.geodetic_to_local(lat, lon, alt)
        seconds.append([local, i, 0])  # 0 表示未转弯
        total += 1

    #转换敌群坐标
    enemys = []
    enemys_tmp = []
    for i, enemy in enumerate(enemy_geo):
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(enemy)
        local = converter.geodetic_to_local(lat, lon, alt)
        enemys.append([local, i, 0])
        enemys_tmp.append(local)


    # 转换敌群中心
    B_lat, B_lon, B_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    enemy_center_local = converter.geodetic_to_local(B_lat, B_lon, B_alt)

    #敌群转弯结束后位置补偿
    enemy_move_distance = enemy_speed * turn_time * 2 #独立转弯及延迟转弯两次转弯的补偿
    enemy_center_local[1] = enemy_center_local[1] - enemy_move_distance

    # 同步补偿敌群内每个敌机坐标
    for enemy in enemys:
        enemy[0][1] -= enemy_move_distance

    for enemy in enemys_tmp:
        enemy[1] = enemy[1]- enemy_move_distance

    enemys_afterfirst = []
    for enemy in enemys_tmp:
        test_local = converter.local_to_geodetic_dms(enemy)
        enemys_afterfirst.append(test_local)

    # 重新计算敌群边界（新中心）
    min_x, max_x, min_y, max_y, min_z, max_z = find_edge_points(enemys)

    print(f"理论的敌群边界:{min_x, max_x, min_y, max_y, min_z, max_z}")
    a = max_y - min_y
    b = max_z - min_z
    angle_deg = math.degrees(math.atan2(a, b))


    # 计算最外圈每一条边理论上最大无人机数以及每一层最大无人机数
    diagonal_length = math.sqrt((max_z - min_z) ** 2 + (max_y - min_y) ** 2)
    quantity_a = int((max_x - min_x - detection_size[0]) / detection_size[0]) + 1
    quantity_b = int((diagonal_length - detection_size [1]) / detection_size[1])  #因为起点在上下边缘上，可认为初始就能有两个点
    quantity = quantity_b * quantity_a

    print(f"显示圈上无人机数:{quantity} 左右两条边只能放{quantity_a}个，上下两条边只能放{quantity_b}个")

    #占位分配结束得到目标位置
    target= allocation_policy_all(total, min_x, max_x, min_y, max_y, min_z, max_z,
                               angle_deg, quantity_a, quantity_b)

    #转回经纬度
    uavstr = []
    for uav in target:
        result = converter.local_to_geodetic_dms(uav)
        uavstr.append(result)

    return uavstr,enemys_afterfirst


#8. 根据相遇时间（第一波与敌群相遇）获得相遇时敌方和第二波独立转弯无人机的位置
def update_ipositions_geo(enemy_positions_geo, uav_positions_geo, enemy_center_geo, our_center_geo,
                        enemy_speed, converter, dt, uav_distances, height_distances):
    #敌方经纬度，我方经纬度，敌方中心，我方中心，敌方速度，对象，我方预测飞行距离

    #计算敌我两方的方向向量
    direction_unit = calculate_direction_vector(enemy_center_geo, our_center_geo, converter)

    updated_enemy_positions = []
    updated_uav_positions = []
    turn_second_points = []

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
        new_pos_local = pos_local
        new_geo = converter.local_to_geodetic_dms(new_pos_local)
        # 竖直方向爬升
        new_geo[2] = str(float(new_geo[2]) + height_distances)
        updated_uav_positions.append(new_geo)
        turn_second_points.append(new_geo)


    # 更新我方中心坐标
    uav_center_lat, uav_center_lon, uav_center_alt = GeodeticConverter.decimal_dms_to_degrees(our_center_geo)
    pos_uav_center = converter.geodetic_to_local(uav_center_lat, uav_center_lon, uav_center_alt)

    pos_uav_center[1] = pos_uav_center[1] + uav_distances
    new_pos_uav_center = pos_uav_center
    new_geo_uav_center = converter.local_to_geodetic_dms(new_pos_uav_center)
    #竖直爬升距离
    new_geo_uav_center[2] = str(float(new_geo_uav_center[2]) + height_distances)


    return updated_enemy_positions, updated_uav_positions, new_geo_enemy_center, new_geo_uav_center, direction_unit, turn_second_points

#9. 因为敌我速度均很快，为了保证相遇过程中敌群一直在延迟转弯无人机视场内，紧急减速情况下的安全距离
def safety_distance (uavs_speed, enemy_speed):
    t_reduce = uavs_speed[4] / uavs_speed[2]
    dis_reduce = uavs_speed[4] ** 2 / (2 * uavs_speed[2]) + enemy_speed * t_reduce
    print(f"t_reduce:{t_reduce}  dis_reduce:{dis_reduce}")

    return t_reduce, dis_reduce


#10. 根据转弯时间+相遇时间的总时间（第一波已经完成转弯）获得相遇时第二波延迟转弯无人机的位置
def update_dposition_geo( duav_positions_geo, enemy_speed, converter, uav_distances, height_distances, uavs_speed, turn_second_points):
    updated_uav_positions = []

    # 延迟转弯的无人机减速到0需要多少时间以及往前走多少距离
    t_reduce, dis_reduce = safety_distance(uavs_speed,enemy_speed)
    #相对的，敌方会在我方减速的时候前进
    enemy_dis = t_reduce * enemy_speed

    # 我方经纬度转坐标系(更新到第一波无人机及独立转弯点无人机即将转弯时）
    for idx, geo in enumerate(duav_positions_geo):
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        pos_local = converter.geodetic_to_local(lat, lon, alt)

        # 我方无人机沿正方向飞行
        pos_local[1] = pos_local[1] + uav_distances - dis_reduce - enemy_dis
        new_pos_local = pos_local
        new_geo = converter.local_to_geodetic_dms(new_pos_local)
        # 竖直方向爬升
        new_geo[2] = str(float(new_geo[2]) + height_distances)
        updated_uav_positions.append(new_geo)
        turn_second_points.append(new_geo)

    return  updated_uav_positions, turn_second_points


def update_enemy_geo(enemy_position_geo, enemy_speed, converter, turn_time):
    enemy_end_points = []

    for i, enemy in enumerate(enemy_position_geo):
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(enemy)
        local = converter.geodetic_to_local(lat, lon, alt)
        local[1] = local[1] - enemy_speed * turn_time
        enemy_end_geo = converter.local_to_geodetic_dms(local)
        enemy_end_points.append(enemy_end_geo)

    return enemy_end_points



#判断第1波无人机何时即将与敌群交错，以便得到转弯时机(目前假设第1波无人机也是先加速后减速与敌方相遇，暂留此函数）
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


#10. 三维打印无人机相遇的情况以便判断
def plot_positions_with_centers(enemy_positions_geo_init, uav_positions_geo_init,
                                enemy_positions_geo_new, uav_positions_geo_new,
                                uav_end_geo,
                                enemy_center_geo_init, our_center_geo_init,
                                enemy_center_geo_new, our_center_geo_new,
                                converter):


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

    # 最终我方点
    uav_end_lats, uav_end_lons, uav_end_alts = [], [], []
    for geo in uav_end_geo:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        uav_end_lats.append(lat)
        uav_end_lons.append(lon)
        uav_end_alts.append(alt)

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

    #最后我方
    ax.scatter(uav_end_lons, uav_end_lats, uav_end_alts, c='green', marker='x', label='UAV End', s=50)

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
    # 敌机及第二波次无人机初始点数据（经纬度格式）
    enemy_geo, second_points = dataset.dataset()

    #第一波次无人机基点
    base = ["28:30:34.95W", "10:11:35.24N", "55.0"]
    # 敌机群中心位置
    enemy_center = ["28:11:34.95W", "10:30:35.24N", "55.0"]
    enemy_speed = 240
    uavs_speed = [180, 80, 80, 300, 200] #当前、加速、减速、最大、转弯速度
    turn_time = 15 #第二波无人机转弯时间
    first_turn_time = 10 #假设第一波无人机用10秒转弯
    distance_follow = 500 #无人机与无人机之间y轴上跟随距离
    detection_size = [1000, 1000, 1000] #分别表示无人机xyz三个方向能探查的距离
    height = 500
    y_gap = 2000 #跟随敌机距离
    dt = 1
    base_distance_threshold = 1000 #base距离阈值
    turn_distance_threshold = 500 #迎面距离阈值


    # 创建局部坐标转换器
    A_lat, A_lon, A_alt = GeodeticConverter.decimal_dms_to_degrees(base)
    B_lat, B_lon, B_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    converter = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)


    #对第2波次无人机进行分类，一部分延迟转弯，一部分独立转弯
    decision_array = classify_uav_turning(second_points, base, converter, base_distance_threshold)


    second_iuav_points = [] #独立转弯的无人机
    second_duav_points = [] #延迟转弯的无人机
    # 打印结果
    for idx, val in enumerate(decision_array):
        if val == 1:
            print(f"无人机 {idx}: 距离近，跟随第一波转弯 (1)")
            second_duav_points.append(second_points[idx])
        else:
            print(f"无人机 {idx}: 距离远，独立转弯 (0)")
            second_iuav_points.append(second_points[idx])

    #获得敌方中心转坐标系的点位
    enemy_center_local = converter.geodetic_to_local(B_lat, B_lon, B_alt)

    # #获得我方第一波次坐标系点位
    # uav_first_local = converter.geodetic_to_local(A_lat, A_lon, A_alt)
    # print(f"我方第一波次无人机位置:{uav_first_local[1]}")

    #获得预测时间和我方预期飞行距离
    predict_time, predict_distance, height_distances_max = calculate_distances_opposite(enemy_center_local[1], uavs_speed[0],
                                                                  uavs_speed[3], uavs_speed[4], uavs_speed[1], uavs_speed[2], enemy_speed) #因为原点是0，只需要输入敌方距离即可判定转弯时机

    print(f"预期飞行时间:{predict_time} 预期飞行距离:{predict_distance} 总距离：{enemy_center_local[1]} 敌方飞行距离:{enemy_speed*predict_time}")

    # 计算最小安全爬升高度
    print(f"\n=== 最小安全爬升高度计算 ===")
    min_climb_height, enemy_max_alt, uav_min_alt = calculate_min_safe_vertical_distance(
        enemy_geo, second_iuav_points, converter, safety_margin=100
    )
    print(f"敌机最大高度: {enemy_max_alt:.2f} 米")
    print(f"我方无人机最小高度: {uav_min_alt:.2f} 米") 
    print(f"要求最小爬升高度: {min_climb_height:.2f} 米")
    print(f"求解最大爬升距离:{height_distances_max:.2f} 米")

    height_distances = min(height_distances_max, min_climb_height)

    print(f"爬升高度为: {height_distances:.2f} 米")

    #获得相遇时敌方的坐标，我方的坐标，敌方的中心，我方第二波次独立转弯的无人机中心
    new_enemy_points, new_iuav_points, new_enemy_center, new_uav_center, direction_unit, turn_second_points = update_ipositions_geo(enemy_geo, second_iuav_points, enemy_center, base,
                                                                                              enemy_speed, converter, predict_time, predict_distance, height_distances)

    #获得第二波次延迟转弯点无人机转弯位置,第二波延迟转弯无人机在确认第一波完成转弯后再转
    new_duav_points, turn_second_points = update_dposition_geo(second_duav_points, enemy_speed, converter, predict_distance, height_distances, uavs_speed, turn_second_points)


    print("敌机相遇位置:")
    for e in new_enemy_points:
        print(e)

    print("\n无人机转弯位置:")
    for u in turn_second_points:
        print(u)


    uav_end_points,enemy_afterfirst = second_turning_position(new_enemy_points, new_enemy_center, turn_second_points, enemy_speed, turn_time, distance_follow, detection_size, height, y_gap, converter)
    print("\n无人机转弯后终点位置:")
    for v in uav_end_points:
        print(v)

    #当第二波延迟转弯的无人机也转弯了，那敌机的位置如下
    enemy_end_points = update_enemy_geo(enemy_afterfirst, enemy_speed, converter, turn_time)

    print("\n转弯后敌机移动终点位置:")
    for n in enemy_end_points:
        print(n)


    #数据输出存储
    outdata.save_uav_multi_positions(uavs_speed, enemy_speed, second_points, enemy_geo, turn_second_points, uav_end_points, enemy_end_points)


    # 画初始位置和第二波独立转弯后占位位置
    plot_positions_with_centers(enemy_geo, second_points, enemy_afterfirst, new_iuav_points, uav_end_points, enemy_center, base, new_enemy_center, new_uav_center, converter)







