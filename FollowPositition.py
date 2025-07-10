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

    # 变动位移 ：
    move_dist = (move_t + turn_time) * enemy_speed
    return move_dist


#在三维点集中，找出边界的最大值最小值
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

    # 确定边缘点（哪些点在边缘上）如果某个点的坐标是最小最大值，那就说明在边界点上
    edge_points = []
    for point in data:
        x, y, z = point[0]
        if (x == min_x or x == max_x) or (y == min_y or y == max_y):
            edge_points.append(point)

    return min_x, max_x, min_y, max_y, min_z, max_z


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



#实时更新敌我两方无人机经纬度坐标
def update_positions_geo(enemy_positions_geo, uav_positions_geo,
                         enemy_center_geo, our_center_geo,
                         enemy_speed, uav_speeds, converter, dt=1.0):
    #敌群经纬度，我方经纬度，敌群中心，我方中心，敌群速度，我方速度，对象，时间段

    # 敌群中心经纬度转坐标轴
    e_lat, e_lon, e_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center_geo)
    enemy_center_local = converter.geodetic_to_local(e_lat, e_lon, e_alt)

    # 我方中心经纬度转坐标轴
    o_lat, o_lon, o_alt = GeodeticConverter.decimal_dms_to_degrees(our_center_geo)
    our_center_local = converter.geodetic_to_local(o_lat, o_lon, o_alt)

    # 计算全局方向向量（敌群中心 - 我方中心）
    direction = np.array(enemy_center_local) - np.array(our_center_local)
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction_unit = direction / norm
    else:
        direction_unit = np.zeros(3)

    updated_enemy_positions = []
    updated_uav_positions = []

    # 更新敌机群
    for geo in enemy_positions_geo:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        pos_local = converter.geodetic_to_local(lat, lon, alt)

        # 敌机沿相反方向飞
        new_pos_local = pos_local - direction_unit * enemy_speed * dt

        # 转回经纬度
        new_geo = converter.local_to_geodetic_dms(new_pos_local)
        updated_enemy_positions.append(new_geo)

    # 更新我方无人机群
    for idx, geo in enumerate(uav_positions_geo):
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        pos_local = converter.geodetic_to_local(lat, lon, alt)

        # 我方无人机沿正方向飞
        new_pos_local = pos_local + direction_unit * uav_speeds[idx] * dt

        # 转回经纬度
        new_geo = converter.local_to_geodetic_dms(new_pos_local)
        updated_uav_positions.append(new_geo)

    return updated_enemy_positions, updated_uav_positions


#对第二波次进行分类，按照与base的阈值分为随第1波次转弯以及不随第1波次转弯，而是自己按最优航迹开始转弯，记录类别
def classify_uav_turning(uav_positions_geo, base_geo, converter, distance_threshold):
    #我方无人机经纬度、base、对象、阈值

    # 将 base 转换为局部坐标
    base_lat, base_lon, base_alt = GeodeticConverter.decimal_dms_to_degrees(base_geo)
    base_local = converter.geodetic_to_local(base_lat, base_lon, base_alt)

    decisions = []

    for idx, geo in enumerate(uav_positions_geo):
        # 将无人机经纬度转换为局部坐标
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        pos_local = converter.geodetic_to_local(lat, lon, alt)

        # 计算到 base 的直线距离
        dist = np.linalg.norm(np.array(pos_local) - np.array(base_local))

        # 判定是否小于等于阈值
        if dist <= distance_threshold:
            decisions.append(1)  # 跟随第一波一起转弯
        else:
            decisions.append(0)  # 独立按最优航迹转弯


        # print(f"无人机 {idx}: 距离 base = {dist:.2f} m, 判定 = {'跟随转弯' if dist <= distance_threshold else '独立转弯'}")

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

    # 获取无人机起点（局部坐标）
    uav_start_points_local = []
    for geo in uav_positions_geo:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        local = converter.geodetic_to_local(lat, lon, alt)
        uav_start_points_local.append(local)

    # 调用second_turning_position计算终点（占位位置）
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

    # 转成局部坐标
    uav_end_points_local = []
    for geo in final_positions_geo:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(geo)
        local = converter.geodetic_to_local(lat, lon, alt)
        uav_end_points_local.append(local)

    return uav_start_points_local, uav_end_points_local




















if __name__ == "__main__":
    #敌机数据
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

    # 打印结果
    for idx, val in enumerate(decision_array):
        if val == 1:
            print(f"无人机 {idx}: 距离近，跟随第一波转弯 (1)")
        else:
            print(f"无人机 {idx}: 距离远，独立转弯 (0)")

    #获得交错时间
    predicted_time, base_local_final, enemy_local_final = predict_cross_time(base, enemy_center, uavs_speed[0], enemy_speed, converter, turn_distance_threshold, dt)

    uav_start_points_local, uav_end_points_local = get_uav_turning_points(
        enemy_geo,
        enemy_center,
        second_points,
        base,
        enemy_speed,
        uavs_speed[0],  # 无人机飞行速度标量
        uavs_speed,  # 无人机速度参数列表
        converter,
        predicted_time,
        turn_time,
        distance_follow,
        detection_size,
        height,
        y_gap,
        dt
    )

    # 打印起点和终点
    for idx in range(len(uav_start_points_local)):
        print(f"无人机 {idx}: 起点 local = {uav_start_points_local[idx]}, 终点 local = {uav_end_points_local[idx]}")




    # enemy_positions_geo_new, uav_positions_geo_new = update_positions_geo(enemy_geo, second_points, enemy_center, base, enemy_speed, uav_speed, converter, dt)
    #
    # print("敌机新位置:")
    # for e in enemy_positions_geo_new:
    #     print(e)
    #
    # print("\n无人机新位置:")
    # for u in uav_positions_geo_new:
    #     print(u)

    # #计算交会时间和距离
    # results_sorted = calculate_meeting_time(second_points, base, enemy_center, enemy_speed, uavs_speed[0])
    #
    # #分配转弯策略
    # results_with_strategy = assign_turning_strategy(results_sorted)
    #
    # #初始化坐标转换器
    # A_lat, A_lon, A_alt = GeodeticConverter.decimal_dms_to_degrees(base)
    # B_lat, B_lon, B_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    # converter = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)
    #
    #
    # #计算最终占位位置
    # final_positions = get_final_positions(enemy_geo, enemy_center, second_points, base, enemy_speed, uavs_speed, turn_time, distance_follow, detection_size, height, y_gap, converter)
    #
    # #为每架无人机生成轨迹
    # all_traj = {}
    #
    # for res in results_with_strategy:
    #     uav_id = res['uav_id']
    #
    #     # 当前无人机到敌群的距离
    #     dist_to_enemy = res['distance']
    #
    #     if dist_to_enemy > turn_distance_threshold:
    #         print(f"无人机 {uav_id} 当前距离 {dist_to_enemy:.2f} m，未进入转弯阶段，保持迎面飞行。")
    #         continue  # 不生成轨迹，不进入转弯
    #     else:
    #         print(f"无人机 {uav_id} 距离 {dist_to_enemy:.2f} m，进入转弯阶段，执行转弯策略。")
    #
    #     # 当前无人机初始局部坐标
    #     lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(second_points[uav_id])
    #     start_local = converter.geodetic_to_local(lat, lon, alt)
    #
    #     # 占位终点局部坐标
    #     end_local = final_positions[uav_id]['final_local']
    #
    #     # 设置飞行参数（可根据策略微调）
    #     uav_speed_params = {
    #         'v_start': uavs_speed[0],
    #         'v_max': uavs_speed[3],
    #         'v_end': uavs_speed[4],
    #         'acc': uavs_speed[1],
    #         'dec': uavs_speed[2]
    #     }
    #
    #     traj = generate_trajectory(start_local, end_local, uav_speed_params)
    #     all_traj[uav_id] = traj
    #
    #     #冲突检查
    #     conflicts = check_trajectory_conflict(all_traj, distance_follow)
    #
    #     #输出结果
    #     print("转弯策略分配结果")
    #     for re in results_with_strategy:
    #         print(
    #             f"无人机 {re['uav_id']} - 距离: {re['distance']:.2f} m, 预计交会时间: {re['t_meet']:.2f} s, 策略: {re['turn_strategy']}")
    #
    #     print("\n【冲突检测结果】")
    #     if conflicts:
    #         for c in conflicts:
    #             print(f"时间 {c['t']} s - 无人机 {c['uav1']} & {c['uav2']} 相距 {c['distance']} m")
    #     else:
    #         print("未检测到冲突，安全。")
    #
    # plot_trajectories(all_traj)



