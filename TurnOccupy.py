import geopy
import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
from numpy.testing.print_coercion_tables import print_new_cast_table
from sympy.core.sympify import converter
from sympy.physics.units import acceleration
from scipy.optimize import linear_sum_assignment
from geopy.distance import geodesic


import velocity_recong
from sklearn.cluster import DBSCAN
from geopy.distance import distance
from geopy import Point
import plotly.graph_objs as go
import plotly.graph_objects as go


from numpy.ma.core import remainder

import GeodeticConverter
import outdata
from FollowPositition import safety_distance
from GeodeticConverter import dms_to_decimal
from data.dataset import dataset
from sympy import symbols, solve, Eq, sqrt

import plotly.graph_objects as go

GRAVITY_EARTH = 9.80665  # 地球表面重力加速度
R = 6371000  # 地球半径，单位：米



# 对第1批或第2批uav进行排序，由敌群的近到远或由远到近
def uav_sorted_distances_points(uav_points, uav_center, enemy_center, reverse):
    first = []
    first_sorted_dms = []
    i = 0

    # 将我方中心，敌方中心都转度数
    uav_center_lat, uav_center_lon, uav_center_alt = GeodeticConverter.decimal_dms_to_degrees(uav_center)
    enemy_center_lat, enemy_center_lon, enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)

    # 计算整体飞行方向
    flight_bearing = converter.calculate_flight_bearing(uav_center_lat, uav_center_lon, enemy_center_lat,enemy_center_lon)

    for point in uav_points:

        lat, lon ,alt = GeodeticConverter.decimal_dms_to_degrees(point)

        # 每架无人机的飞行方向，并将其投影到整体移动方向上（适用于几百公里内）
        uav_bearing = converter.calculate_flight_bearing(lat, lon, enemy_center_lat, enemy_center_lon)
        delta_angle = abs(flight_bearing - uav_bearing)
        delta_angle = min(delta_angle, 360 - delta_angle)

        # 计算每架无人机飞行球面距离
        dist = converter.calculate_spherical_distance(lat, lon, alt, enemy_center_lat, enemy_center_lon, alt)
        projected_dist = dist * math.cos(math.radians(delta_angle))

        first.append([projected_dist, i, 0, point])
        i = i + 1

    first_sorted = sorted(first, key=lambda item: item[0], reverse = reverse)#按投影后的距离由小到大（False）的顺序进行排序
    for point in first_sorted:
        first_sorted_dms.append(point[3])

    return first_sorted, first_sorted_dms


# 计算第1波次或第2波次无人机中心
def calculate_center_dms(uav_dms):

    decimal_positions = []

    # 遍历每个函数，得到度数分量
    for dms_pos in uav_dms:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(dms_pos)
        decimal_positions.append((lat, lon, alt))

    # 对所有分量求平均值
    lats = [p[0] for p in decimal_positions]
    lons = [p[1] for p in decimal_positions]
    alts = [p[2] for p in decimal_positions]

    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    center_alt = np.mean(alts)

    # 分量转local再转dms
    uav_center_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(center_lat, center_lon, center_alt))

    return uav_center_dms


# 聚类分析有多少个纵队并分别记录
def cluster_uavs_by_latitude(second_points, converter, eps = 20):
    # 先将second_point转为坐标系
    local_coords = []
    for dms in second_points:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(dms)
        local = converter.geodetic_to_local(lat, lon, alt)
        local_coords.append(local)

    local_coords = np.array(local_coords)

    # 使用局部坐标的x(东西方向）聚类
    local_x = local_coords[:, 0].reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(local_x)
    labels = clustering.labels_  #分别打上标签

    result = []
    for label in sorted(set(labels)):
        group = []
        for idx, l in enumerate(labels):
            if l == label:
                local_pos = local_coords[idx]
                dms = converter.local_to_geodetic_dms(local_pos)
                group.append(dms)
        result.append([int(label), group])  # 强制转为int，防止出现 np.int64

    return result #返回一个二元数组


# 计算敌群最远边界信息，获得敌机边界四个角的信息
def get_enemy_edges(enemy_center, lat_range, lon_range):
    ne = [lon_range[1], lat_range[1], enemy_center[2]]  # 北 + 东
    se = [lon_range[1], lat_range[0], enemy_center[2]]  # 南 + 东
    nw = [lon_range[0], lat_range[1], enemy_center[2]]  # 北 + 西
    sw = [lon_range[0], lat_range[0], enemy_center[2]]  # 南 + 西
    return ne, se, nw, sw


# 将dms坐标转化为local
def to_local(pos_dms, converter):
    lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(pos_dms)
    return converter.geodetic_to_local(lat, lon, alt)


# 计算第1波无人机距离敌群的最远距离
def max_distance(basepoint, enemy_center, lat_range, lon_range):
    # 敌方四个角dms格式
    en = [lon_range[1], lat_range[1], enemy_center[2]]  # 东北
    es = [lon_range[1], lat_range[0], enemy_center[2]]  # 东南
    wn = [lon_range[0], lat_range[1], enemy_center[2]]  # 西北
    ws = [lon_range[0], lat_range[0], enemy_center[2]]  # 西南
    corners = [ws, wn, es, en]

    # 转换base为度数
    base_lat, base_lon, base_alt = GeodeticConverter.decimal_dms_to_degrees(basepoint)

    # 初始化最小最大距离
    max_dist = -float('inf')
    min_dist = float('inf')
    max_dms = None
    min_dms = None

    for corner_dms in corners:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(corner_dms)
        dist = converter.calculate_spherical_distance(base_lat, base_lon, base_alt, lat, lon, alt)

        # 有最大或最小的的就记录
        if dist > max_dist:
            max_dist = dist
            max_dms = corner_dms

        if dist < min_dist:
            min_dist = dist
            min_dms = corner_dms

    return max_dist, max_dms, min_dist, min_dms


# 求解探测到敌方最后沿时的时间
def function_last_detection_time(uav_speed, enemy_speed, max_distances, detect_distances):
    relative_speed = uav_speed + enemy_speed
    time = (max_distances - detect_distances ) / relative_speed
    return time


# 计算最后一架无人机需要飞出的总距离(暂时用不上)
def last_uav_distance(basepoint, uavpoint, uav_speed, enemy_speed, max_distances, detect_distances, converter):
    pos_base = to_local(basepoint, converter)

    dis_uav_base = pos_base[1] - uavpoint
    time = function_last_detection_time(uav_speed, enemy_speed, max_distances, detect_distances)
    dis_last_uav = dis_uav_base + uav_speed * time + detect_distances
    return dis_last_uav


# 敌我同时推进，每 step_time 秒更新一次位置，重建 converter。
def simulate_relative_motion(uav_start, enemy_start, basepoint, uav_speed, enemy_speed, total_time, step_time=10):
    steps = total_time // step_time
    delta_uav = uav_speed * step_time
    delta_enemy = enemy_speed * step_time

    # 当前地理位置
    uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(uav_start)
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_start)
    base_lat, base_lon, base_alt = GeodeticConverter.decimal_dms_to_degrees(basepoint)

    base_positions = []
    enemy_positions = []
    converters = []

    for i in range(int(steps) + 1):
        # 创建当前局部坐标系
        converter = GeodeticConverter.GeodeticToLocalConverter(base_lat, base_lon, base_alt, enemy_lat, enemy_lon, enemy_alt)
        print("什么玩意:",converter)

        # 当前位置 → local
        base_local = converter.geodetic_to_local( base_lat, base_lon, base_alt)
        enemy_local = converter.geodetic_to_local(enemy_lat, enemy_lon, enemy_alt)

        # 按 y 轴推进（默认 y 轴方向是 base → enemy_center）
        base_local[1] += delta_uav
        enemy_local[1] -= delta_enemy

        # local → 地理坐标
        next_base_geo = converter.local_to_geodetic_dms(base_local)
        next_enemy_geo = converter.local_to_geodetic_dms(enemy_local)
        next_base_geo[2] = uav_start[2]
        next_enemy_geo[2] = enemy_start[2]

        # 记录每一步坐标与 converter
        base_positions.append(next_base_geo)
        enemy_positions.append(next_enemy_geo)
        converters.append(converter)

    return converter, base_positions, enemy_positions


# 计算转弯后的位置（有转弯度数）
def calculate_turning_position_with_bearing(lat, lon, alt, turning_radius, angle_deg, bearing_deg, direction='left'):

    # 计算圆心方向（偏移航向±90度）
    offset_bearing = (bearing_deg + (90 if direction == 'left' else -90)) % 360

    # 计算圆心点
    center_point = distance(meters=turning_radius).destination(Point(lat, lon), bearing=offset_bearing)

    # 计算终点方向：从圆心开始，按方向旋转角度
    arc_end_bearing = (offset_bearing + (angle_deg if direction == 'left' else -angle_deg)) % 360

    # 沿圆弧从圆心出发，回到圆周上
    end_point = distance(meters=turning_radius).destination(center_point, bearing=arc_end_bearing)

    # print("无人机方向，偏移", {bearing_deg}, offset_bearing)

    return end_point.latitude, end_point.longitude, alt


# 计算无人机在假设转弯弧度后的点位（并未进行追击）
def after_turn_position(turning_radius, uav_meet_dms, angle_deg, bearing_deg, direction='right'):
    # 先转度数
    uav_meet_lat, uav_meet_lon, uav_meet_alt = GeodeticConverter.decimal_dms_to_degrees(uav_meet_dms)

    # 角度制转弧度制
    angle_rad = math.radians(angle_deg)

    # 计算转弯后无人机的位置
    after_turning_uav_lat, after_turning_uav_lon, after_turning_uav_alt = calculate_turning_position_with_bearing(
        uav_meet_lat, uav_meet_lon, uav_meet_alt, turning_radius, angle_rad, bearing_deg, direction)

    # 角度转local再转经纬度，得到转弯某角度后的经纬度坐标
    after_turning_uav_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(after_turning_uav_lat, after_turning_uav_lon, after_turning_uav_alt))

    # print(
    #     f"转弯后无人机的位置: 纬度 = {after_turning_uav_lat}, 经度 = {after_turning_uav_lon}, 高度 = {after_turning_uav_alt}")
    return after_turning_uav_dms


# 计算转弯后进行追赶的时间距离及敌群移动距离
def after_turn_chase(uav_reltime_speed, uav_max_speed, enemy_speed, turning_time, turning_radius, acceleration):

    uav_max_speed = 400
    #加速时间及加速阶段前进距离
    time_acc = (uav_max_speed - uav_reltime_speed) / acceleration
    distance_add = (uav_max_speed ** 2 -uav_reltime_speed ** 2) / (2 * acceleration)


    #定义符号变量以便求解转弯方程
    chase_time, uniform_time, chase_distance, distance_enemy_chase = symbols('chase_time uniform_time, chase_distance distance_enemy_chase')

    #建立方程组
    equations = [
        Eq(time_acc + uniform_time, chase_time),  # 加速与匀速运动的时间关系
        Eq(distance_add + uniform_time * uav_max_speed, chase_distance), # 速度与追击距离的关系
        Eq((turning_time + chase_time) * enemy_speed, distance_enemy_chase),  # 敌人在追击期间移动距离
        Eq(sqrt(distance_enemy_chase ** 2 + (2 * turning_radius) ** 2), chase_distance)  # 追击距离几何关系
    ]

    #解方程组
    solutions = solve(equations, (chase_time, uniform_time, chase_distance, distance_enemy_chase))

    for sol in solutions:
        numeric_sol = [float(val.evalf()) for val in sol]
        chase_time_val, uniform_time_val, chase_distance_val, distance_enemy_chase_val = numeric_sol

    print(
        f"转弯后追赶时间，转弯后追赶距离，转弯及追赶过程中敌群移动距离：[{chase_time_val:.6f}, {uniform_time_val}, {chase_distance_val:.6f}, {distance_enemy_chase_val:.1f}]")

    return chase_time_val, uniform_time_val, chase_distance_val, distance_enemy_chase_val


# 纵队最后一架无人机的飞行策略
def last_uav_move_strategy(uav_speed, uav_max_speed, uav_deceleration_speed, enemy_speed, max_distances, detect_distances, uavpoint, basepoint, min_enemy_dms, acceleration):

    last_time_info = []
    # 飞行时间估计（第1波次无人机将敌方全纳入视场时间）
    time = function_last_detection_time(uav_speed, enemy_speed, max_distances, detect_distances)
    last_time_info.append(time) #匀速时间点

    # 匀加速时间与加速距离
    time_acc = (uav_max_speed - uav_speed) / acceleration  # 得到加速时间
    distance_acc = (uav_max_speed ** 2 - uav_speed ** 2) / (2 * acceleration)  # 得到加速期间前进的距离
    last_time_info.append(time_acc + last_time_info[0]) #加速时间点

    # 加速前已经飞行距离
    total_distances = uav_speed * time + distance_acc

    # 当前无人机位置
    lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(uavpoint)

    # 敌群中心位置
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(min_enemy_dms)

    # 求飞行方向
    bearing =  converter.calculate_flight_bearing(lat, lon, enemy_lat, enemy_lon)
    print(f"<UNK> = {bearing:.6f}")
    bearing_enemy = converter.calculate_flight_bearing(enemy_lat, enemy_lon, lat, lon)

    # 球面预测加速后飞行点
    new_lat, new_lon, new_alt = converter.calculate_destination_point(
        lat, lon, alt, bearing, total_distances, 0)

    # 敌群位置更新（反方向飞行）
    enemy_movedis = enemy_speed * (time + time_acc)
    new_enemy_lat, new_enemy_lon, new_enemy_alt = converter.calculate_destination_point(
        enemy_lat, enemy_lon, enemy_alt, bearing_enemy, enemy_movedis, 0)

    # 计算此时相对距离
    distance_move = converter.calculate_spherical_distance(new_lat, new_lon, new_alt, new_enemy_lat, new_enemy_lon, new_enemy_alt)

    safety_distance = 1000

    # 计算减速时间和减速距离
    deceleration_time = (uav_max_speed - uav_deceleration_speed) / acceleration
    uav_deceleration_distence = uav_max_speed * deceleration_time - 0.5 * acceleration *deceleration_time **2

    # 匀速减速再相遇所需要的总时间 = （不减速时的距离 - 安全距离 + 不减速时的距离与考虑减速时的距离之差） / 相对速度
    time_move = (distance_move - safety_distance + uav_max_speed * deceleration_time - uav_deceleration_distence) / (uav_max_speed + enemy_speed)

    # 无人机最大速度匀速前进距离
    distance_uav = uav_max_speed * (time_move - deceleration_time)

    # 匀速时间
    time_uniform = time_move - deceleration_time
    last_time_info.append(time_uniform + last_time_info[1]) #继续匀速得到时间点
    last_time_info.append(deceleration_time + last_time_info[2]) #得到减速时间点也是转弯时间点

    # 敌方移动总距离
    distance_enemy = enemy_speed * time_move

    # 相遇时敌机中心位置更新（反方向飞行）
    meet_enemy_lat, meet_enemy_lon, meet_enemy_alt = converter.calculate_destination_point(
        new_enemy_lat, new_enemy_lon, new_enemy_alt, bearing_enemy, distance_enemy, 0
    )

    # 在减速到达敌方前就爬升至敌方高度上方
    angle = converter.calculate_climb_angle(new_alt, new_enemy_alt + safety_distance, distance_uav)
    angle_rad = math.radians(angle)

    # 开始转弯的点位（相遇位置）
    uav_meet_lat, uav_meet_lon, uav_meet_alt, distance_uav_val= converter.calculate_destination_with_climb_angle(new_lat, new_lon, new_alt, bearing, distance_uav, angle)

    # 当前假设转弯180度
    angle_deg = 180
    angle_deg_rad = math.radians(angle_deg)

    # 计算转弯半径
    turning_radius = (uav_deceleration_speed**2)/ (GRAVITY_EARTH * math.tan(math.radians(45))) #  (uav_deceleration_speed**2) * (math.cos(angle_rad)**2))
    # print("半径", turning_radius)
    turning_time = (angle_deg_rad * turning_radius) / uav_deceleration_speed
    # print("转弯时间", turning_time)
    last_time_info.append(turning_time + last_time_info[3])

    #转弯时我方无人机及敌方无人机转dms
    last_begin_turn_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(uav_meet_lat, uav_meet_lon, uav_meet_alt))
    meet_last_enemy_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(meet_enemy_lat, meet_enemy_lon, meet_enemy_alt))


    bearing_rel =  converter.calculate_flight_bearing(uav_meet_lat, uav_meet_lon, enemy_lat, enemy_lon)
    # print("最开始敌群最近位置:",min_enemy_dms)

    # 得到转弯angle_deg后的无人机dms位置
    after_turning_last_uav_dms = after_turn_position(turning_radius, last_begin_turn_dms, angle_deg, bearing_rel, direction = 'right')
    turn_uav_lat, turn_uav_lon, turn_uav_alt = GeodeticConverter.decimal_dms_to_degrees(after_turning_last_uav_dms)

    # 得到转弯后追击敌方时所需要花费的时间距离等
    chase_time_val, uniform_time_val, chase_distance_val, distance_enemy_chase_val = after_turn_chase(
        uav_deceleration_speed, uav_max_speed, enemy_speed, turning_time, turning_radius, acceleration)

    # 得到追击到敌方时敌我两方位置
    chase_enemy_lat, chase_enemy_lon, chase_enemy_alt = converter.calculate_destination_point(
        meet_enemy_lat, meet_enemy_lon, meet_enemy_alt, bearing_enemy, distance_enemy_chase_val, 0)

    chase_last_uav_lat, chase_last_uav_lon, chase_last_uav_alt = converter.calculate_destination_point(
        meet_enemy_lat, meet_enemy_lon, meet_enemy_alt + safety_distance, bearing_enemy, distance_enemy_chase_val, 0)

    # 转弯后此时敌我两方的经纬度位置
    chase_enemy_dms = converter.local_to_geodetic_dms(
         converter.geodetic_to_local(chase_enemy_lat, chase_enemy_lon, chase_enemy_alt))
    chase_uav_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(chase_last_uav_lat, chase_last_uav_lon, chase_last_uav_alt))
    # 追击后的时间
    after_chase_time = time + time_move + turning_time

    return last_begin_turn_dms, after_turning_last_uav_dms, chase_uav_dms, meet_last_enemy_dms, chase_enemy_dms, turning_time, chase_time_val, last_time_info


# 计算第一批无人机与敌群相遇的时间
def first_meet_enemy_time(uav_sorted, max_enemy_dms, uav_speed, uav_deceleration_speed, enemy_speed, safe_distence, acceleration):
    meet_time_info = []
    # 敌机中心经纬度转度数
    e_lat, e_lon, e_alt = GeodeticConverter.decimal_dms_to_degrees(max_enemy_dms)

    # 对于已经排序的无人机
    for i,uav in enumerate(uav_sorted):

        # 取排序无人机的第一个元素
        uav_pos = uav[3]
        # 记录编号
        uav_number = uav [1]
        #print("uav_number",uav_number)

        # 无人机直角坐标转经纬度再转度数
        lat, lon ,alt = GeodeticConverter.decimal_dms_to_degrees(uav_pos)

        # 快到转弯点时要减速，求减速时间和距离
        deceleration_time = (uav_speed - uav_deceleration_speed) / acceleration
        uav_deceleration_distence = uav_speed * deceleration_time - 0.5 * acceleration * deceleration_time ** 2

        # 利用球面坐标系计算无人机与敌机之间的初始距离
        distance_between_uav_enemy = converter.calculate_spherical_distance(lat, lon, e_alt, e_lat, e_lon, e_alt)
        #print("无人机与敌机之间的初始距离", distance_between_uav_enemy)

        # 求匀速前行时的时间以及总时间
        uniform_time = (distance_between_uav_enemy - uav_deceleration_distence - enemy_speed * deceleration_time) / (enemy_speed + uav_speed)
        total_time = uniform_time + deceleration_time

        # 无人机移动距离
        distance_uav = uav_speed * uniform_time + uav_deceleration_distence

        # 添加到meet_time列表中
        meet_time_info.append((uav_number, total_time, uniform_time, distance_uav))
    # print("meet_time_all:",meet_time_info)

    return meet_time_info


# 第1波次或第2波次跟随飞行无人机飞行策略
def first_uav_move_strategy(uav_speed, uav_dec_speed, uav_max_speed, enemy_speed, first_sorted_uav, first_uav_center, max_enemy_dms, safety_distance, acceleration, time_detect):
    first_uav_time_info = []
    start_first_uav_dms = []
    meet_first_uav_dms = []
    after_turn_first_uav_dms = []

    # 第1波次无人机中心位置
    center_lat, center_lon, center_alt = GeodeticConverter.decimal_dms_to_degrees(first_uav_center)

    # 敌群中心位置
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(max_enemy_dms)

    # 求飞行方向
    bearing = converter.calculate_flight_bearing(center_lat, center_lon, enemy_lat, enemy_lon)
    bearing_enemy = converter.calculate_flight_bearing(enemy_lat, enemy_lon, center_lat, center_lon)

    # 得到将敌方纳入视场内时我方和敌方分别移动距离
    distance_uav = uav_speed * time_detect
    distance_enemy = enemy_speed * time_detect

    # 更新我方将敌方纳入视场时位置
    for i,uav in enumerate(first_sorted_uav):
        # 排序后无人机的dms坐标
        first_uav = uav[3]
        # dms转度数
        uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(first_uav)


        new_uav_lat, new_uav_lon, new_uav_alt = converter.calculate_destination_point(
                    uav_lat, uav_lon, uav_alt, bearing, distance_uav, 0)

        start_first_uav = converter.local_to_geodetic_dms(
                    converter.geodetic_to_local(new_uav_lat, new_uav_lon, new_uav_alt))
        start_first_uav_dms.append((uav[0] - distance_uav, uav[1], uav[2],start_first_uav))
        # print("start_first_uav_dmsstart_first_uav_dms", start_first_uav_dms)

    # 更新敌方被纳入视场时位置
    start_enemy_lat, start_enemy_lon, start_enemy_alt = converter.calculate_destination_point(
                    enemy_lat, enemy_lon, enemy_alt, bearing_enemy, distance_enemy, 0)
    start_enemy = converter.local_to_geodetic_dms(
                    converter.geodetic_to_local(start_enemy_lat, start_enemy_lon, start_enemy_alt))

    # 更新第1波或第2波次无人机中心位置
    start_center_lat, start_center_lon, start_center_alt = converter.calculate_destination_point(
                    center_lat, center_lon, center_alt, bearing, distance_uav, 0)
    start_center_uav = converter.local_to_geodetic_dms(
                    converter.geodetic_to_local(start_center_lat, start_center_lon, start_center_alt))

    # 获得排序后第1波或第2波次无人机相遇时间（含减速到100的时间）
    meet_time_info = first_meet_enemy_time(start_first_uav_dms, start_enemy, uav_speed, uav_dec_speed, enemy_speed, safety_distance, acceleration)

    # 从时间信息表中获取平均匀速时间
    time2_values = [entry[2] for entry in meet_time_info]
    avg_time2 = sum(time2_values) / len(time2_values)
    time_uniform = avg_time2 + time_detect

    first_uav_time_info.append(time_uniform) #获得匀速时间点
    first_uav_time_info.append(first_uav_time_info[0]) #没有加速，加速时间点=匀速时间点
    first_uav_time_info.append(first_uav_time_info[1]) #没有加速后匀速，加速后匀速时间点=匀速时间点

    # 从信息表中获取平均匀速再减速总时间
    time1_values = [entry[1] for entry in meet_time_info]
    avg_time1 = sum(time1_values) / len(time1_values)
    time_del = avg_time1 - avg_time2 #减速了几秒的平均时间

    first_uav_time_info.append(time_del + first_uav_time_info[2])

    # 飞行方向
    bearing_next = converter.calculate_flight_bearing(start_center_lat, start_center_lon, start_enemy_lat, start_enemy_lon)
    bearing_enemy_next = converter.calculate_flight_bearing(start_enemy_lat, start_enemy_lon, start_center_lat, start_center_lon)

    print("我方移动方向:",bearing_next)

    for i,uav in enumerate(start_first_uav_dms):

        # 排序后无人机的dms坐标
        first_uav = uav[3]

        # dms转度数
        uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(first_uav)

        # 求无人机转弯开始时的点位（输出）
        meet_uav_lat, meet_uav_lon, meet_uav_alt = converter.calculate_destination_point(
            uav_lat, uav_lon, uav_alt, bearing_next, meet_time_info[i][3], 0)
        meet_uav_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(meet_uav_lat, meet_uav_lon, meet_uav_alt))
        meet_first_uav_dms.append(meet_uav_dms)
        # print("转弯前的无人机坐标：", meet_uav_dms)

        # 当前假设转弯180度
        angle_deg = 180
        angle_deg_rad = math.radians(angle_deg)

        # 计算转弯半径（目前转弯速度和转弯半径均相同）(输出)
        turning_radius = (uav_deceleration_speed ** 2) / (GRAVITY_EARTH * math.tan(math.radians(45)))  # (uav_deceleration_speed**2) * (math.cos(angle_rad)**2))
        turning_time = (angle_deg_rad * turning_radius) / uav_deceleration_speed


        # 由于每架无人机遇到敌机的时机可能都不一样，所以认为飞行航向是当前无人机的飞行航向，而不再以整体飞行航向作为标准
        bearing_rel = converter.calculate_flight_bearing(meet_uav_lat, meet_uav_lon, enemy_lat, enemy_lon)

        # 得到转弯angle_deg后的无人机dms位置
        after_turn_uav_dms = after_turn_position(turning_radius, meet_uav_dms, angle_deg, bearing_rel, direction='right')
        after_turn_first_uav_dms.append(after_turn_uav_dms)
        # print("转弯后的无人机坐标", after_turn_uav_dms)

        first_uav_time_info.append(turning_time + first_uav_time_info[3])  # 转弯完成时间
    # 每一架无人机从进入视场开始直到转弯结束的时间，仅时间列表，time_detect是开始到进入视场，turning_time是转弯时间
    meet_single_time = [(entry[1] + time_detect + turning_time) for entry in meet_time_info]
    # print("每架无人机从开始到转弯结束的时间", meet_single_time)
    return meet_first_uav_dms, after_turn_first_uav_dms, first_uav_time_info, bearing_enemy_next, meet_single_time


# 合成时间信息表（3个批次的）
def generate_uav_time_info (last_uav_time_info, first_uav_time_info, second_uav_time_info):
    time_uav_info = {}

    # 依次插入到字典中
    time_uav_info[0] = last_uav_time_info
    time_uav_info[1] = first_uav_time_info
    time_uav_info[2] = second_uav_time_info

    return time_uav_info


# 获取同一时间三波次的状态,1是第一波次，2是第二波次，3是第二波次最后一架（前置转弯）
def get_uav_state(time, state_end_times, uav_number):
    if uav_number == 3:
        # 第一波次无人机有5个状态
        accel_act, cruise_act, decel_act, turn_act, chase_act, after_chase= state_end_times
        if time <= cruise_act:
            return "加速前进"
        elif cruise_act < time <= decel_act:
            return "匀速前进"
        elif decel_act < time <= turn_act:
            return "减速前进"
        elif turn_act < time <= chase_act:
            return "转弯"
        elif chase_act < time <= after_chase:
            return "追击"
        else:
            return "追击结束，跟随敌群"


    elif uav_number == 1 or uav_number == 2:
        # 第二波次和第三波次无人机都有4个状态
        accel_act, cruise_act, decel_act, turn_act, chase_act, after_chase = state_end_times
        if time <= decel_act:
            return "匀速前进"
        elif decel_act < time <= turn_act:
            return "减速前进"
        elif turn_act < time <= chase_act:
            return "转弯"
        elif chase_act < time <= after_chase:
            return "追击"
        else:
            return "追击结束，跟随敌群"


# 计算坐标点投影值
def dot_projection(p, vx, vy):
    return p[0] * vx + p[1] * vy


# 计算敌群前进方向的前沿边和后沿边，并进行无人机占位
def generate_placements_with_bearing(
    enemy_center_dms, enemy_latrange_dms, enemy_lonrange_dms,
    distance_m,  bearing_deg, exclusion_radius_m, preplaced_dms,
    start_side='left', max_rows=999, first_uav_num = 45, second_uav_num = 30
):
    # 先对中心进行转化
    lat_c, lon_c, alt_c = GeodeticConverter.decimal_dms_to_degrees(enemy_center_dms)
    alt = float(alt_c)

    # 得到四个角的角点位置
    en = [enemy_lonrange_dms[1], enemy_latrange_dms[1], enemy_center_dms[2]]  # 东北
    es = [enemy_lonrange_dms[1], enemy_latrange_dms[0], enemy_center_dms[2]]  # 东南
    wn = [enemy_lonrange_dms[0], enemy_latrange_dms[1], enemy_center_dms[2]]  # 西北
    ws = [enemy_lonrange_dms[0], enemy_latrange_dms[0], enemy_center_dms[2]]  # 西南
    corners = [ws, wn, en, es]

    # 以西南角为中心，东南角为x轴建立局部坐标系
    ws_lat, ws_lon, ws_alt = GeodeticConverter.decimal_dms_to_degrees(ws)
    es_lat, es_lon, es_alt = GeodeticConverter.decimal_dms_to_degrees(es)

    # 重新以中心建立局部坐标系，便于进行坐标计算
    converter_enu = GeodeticConverter.GeodeticToLocalConverter(ws_lat, ws_lon, ws_alt, es_lat, es_lon, es_alt)

    # 将四个角点转度数
    corners_degree = []
    for corner in corners:
        corner_lat, corner_lon, corner_alt = GeodeticConverter.decimal_dms_to_degrees(corner)
        corners_degree.append([corner_lat, corner_lon, corner_alt])

    # 四个角点的度数转enu坐标系
    corners_xy = []
    for (la, lo, al) in corners_degree:
        x, y, z = converter_enu.geodetic_to_local(la, lo, al)  # East=x, North=y
        corners_xy.append((x, y))

    # 补回首点，便于边遍历
    corners_xy.append(corners_xy[0])

    # 对方向进行规划，如果是-45到45就当作是北方向，依次类推
    bearing_front = 0
    if 315 < bearing_deg <= 360 or 0 < bearing_deg <=45 :
        bearing_front = 0
    elif 45 < bearing_deg <= 135 :
        bearing_front = 90
    elif 135 < bearing_deg <= 225 :
        bearing_front = 180
    elif 225 < bearing_deg <= 315 :
        bearing_front = 270

    # 飞行前进的方向需要先转弧度制
    theta = math.radians(bearing_front)
    fx, fy = math.sin(theta), math.cos(theta)  #ENU中朝向
    f = (fx, fy) #得到方向向量

    # 计算各角点在f方向上的投影
    projs = [fx*px + fy*py for (px, py) in corners_xy[:-1]]  #就是在方向上的投影距离，原点-方向
    Smax = max(projs)
    Smin = min(projs)

    tol_rel = 1e-4  #假设误差大概是4个数量级
    inters_front, inters_back = [], []
    alpha = 0
    beta = 0

    # 扫描四条边,依次对边进行判断，得到前沿边和后沿边
    for i in range(4):
        x0, y0 = corners_xy[i]
        x1, y1 = corners_xy[i+1]
        dx, dy = x1 - x0, y1 - y0 #这条边的方向向量

        # 根据fx，fy的情况判断应该增加的误差值，如果是fy接近0，说明南北朝向的边可能存在x轴上的误差，如果fx接近0，说明东西朝向的边可能存在y轴上的误差
        if fy < tol_rel and (i == 0 or i == 2):
            alpha = abs(dx)
        elif fx < tol_rel and (i == 1 or i == 3):
            beta = abs(dy)

        denom = fx * dx + fy * dy  # 与前向 f 的点积

        # 是否平行（与支撑线平行 <=> 与 f 垂直）
        is_parallel = abs(denom) <= tol_rel + abs(alpha * fx) + abs(beta * fy)

        if is_parallel :
            # 判是否共线（端点都在支撑线上）
            proj0 = fx * x0 + fy * y0
            proj1 = fx * x1 + fy * y1
            if max(proj0, proj1) == Smax: #记录前沿坐标点
                inters_front.append((x0, y0, 0))
                inters_front.append((x1, y1, 0))

            if min(proj0, proj1) == Smin: #记录后沿坐标点
                inters_back.append((x0, y0, 0))
                inters_back.append((x1, y1, 0))
            # 不共线则没有交点，continue
            continue

    inters_front = [(float(x), float(y), float(z)) for x, y, z in inters_front]
    inters_back = [(float(x), float(y), float(z)) for x, y, z in inters_back]

    # 前沿坐标点转dms（好像没用上）
    front_dms = []
    for inter in inters_front:
        inter_dms = converter_enu.local_to_geodetic_dms(inter)
        inter_dms[2] = alt
        front_dms.append(inter_dms)

    # 计算左手位置的方向向量
    lx, ly = -fy, fx  # 左方向（面向 f 时左手边）

    # 如果左手端点的投影值小于右手端点的投影值，那么说明左右端点存反了，重新存放
    if dot_projection(inters_front[0], lx, ly) < dot_projection(inters_front[1], lx, ly):
        inters_front[0], inters_front[1] = inters_front[1], inters_front[0]  # 交换，确保 pL_raw 是左端
    if dot_projection(inters_back[0], lx, ly) < dot_projection(inters_back[1], lx, ly):
        inters_back[0], inters_back[1] = inters_back[1], inters_back[0]

    pL_raw, pR_raw = inters_front[0], inters_front[1]
    # 计算方向单位向量u，沿着u走其对应的xy的分量
    ux, uy = pR_raw[0] - pL_raw[0], pR_raw[1] - pL_raw[1]
    seg_len = math.hypot(ux, uy)
    if seg_len < 1e-6:
        return [], corners
    ux, uy = ux / seg_len, uy / seg_len

    # 设默认冲突半径，如果没有，就默认是探测距离的一半
    if exclusion_radius_m is None:
        exclusion_radius_m = 0.5 * float(distance_m[0])

    # 把预放置位置DMS转到ENU，便于冲突检测
    preplaced_local = []
    pre_lat, pre_lon, pre_alt = GeodeticConverter.decimal_dms_to_degrees(preplaced_dms)
    local = converter_enu.geodetic_to_local(pre_lat, pre_lon ,pre_alt)
    preplaced_local.append(local)

    # 调用占位策略
    placed_local, placements_dms = occupation_strategy(start_side, f, distance_m, first_uav_num, second_uav_num, seg_len, max_rows, inters_front, inters_back, ux, uy,
                                                       converter_enu, preplaced_local, exclusion_radius_m, alt)
    return front_dms, corners, placements_dms


# 判断当前放置点位与已经放置点位或者提前放置点位是否太近，如果太近小于安全距离，那么当前点位就不再放置新无人机
def too_close(pt, placed_local, preplaced_local, exclusion_radius_m):
    px, py = pt[0], pt[1]
    # 已放
    for qx, qy, _ in placed_local:
        if math.hypot(px - qx, py - qy) < exclusion_radius_m:
            return True
    # 预放
    for qx, qy, _ in preplaced_local:
        if math.hypot(px - qx, py - qy) < exclusion_radius_m:
            return True
    return False


# 计算占位策略，按照第1波次放置在前沿，第2波次放置在后沿的顺序来放，如果第1波次已经放置到后沿，那么第2波次无人机接着放置
def occupation_strategy(start_side, f, distance_m, first_uav_num, second_uav_num, seg_len, max_rows, inters_front, inters_back, ux, uy,
                        converter_enu, preplaced_local, exclusion_radius_m, alt):
    fx, fy = f[0], f[1]
    pL_raw, pR_raw = inters_front[0], inters_front[1]
    pL_back, pR_back = inters_back[0], inters_back[1]

    projs = [fx * p[0] + fy * p[1] for p in inters_back]  # 就是在方向上的投影距离，原点-方向
    Smin = min(projs)
    print("最小值:", Smin)

    # 放置顺序控制
    left_first = (str(start_side).lower() == 'left')

    # 逐行推进
    placements_dms = []
    placed_local = []
    f_back = (-fx, -fy)  # 向后移动一行

    u_margin = float(distance_m[0])  # 与边界的“退让”距离
    step_along = float(distance_m[0])  # 同一行内左右向内推进步长
    row_offset = float(distance_m[0])  # 行间距（向后）

    uavs_budget = first_uav_num
    uavs_budget2 = second_uav_num

    second_replace = False
    min_projection = float('inf')

    # 对行数进行遍历，目前并没有限制行数
    for r in range(int(max_rows)):
        if uavs_budget <= 0:
            break

        # 本行的左右端点（整体向后平移 r*row_offset）
        row_shift_x = f_back[0] * (r * row_offset)
        row_shift_y = f_back[1] * (r * row_offset)
        row_L = (pL_raw[0] + row_shift_x, pL_raw[1] + row_shift_y, 0.0)
        row_R = (pR_raw[0] + row_shift_x, pR_raw[1] + row_shift_y, 0.0)

        # 有效可用长度
        usable_len = seg_len - u_margin
        if usable_len < step_along - 1e-6:
            break  # 这一行已经放不下任意一点

        # 交替放置：k=0,1,2,... 左右向内推进
        k_max = int((seg_len - u_margin) // (2.0 * step_along))
        if k_max < 0 :
            break

        # 一行中能放置的最大对数
        for k in range(k_max + 1):
            # 左侧候选
            Lx = row_L[0] + ux * (0.5 * u_margin + k * step_along)
            Ly = row_L[1] + uy * (0.5 * u_margin + k * step_along)

            # 右侧候选
            Rx = row_R[0] - ux * (0.5 * u_margin + k * step_along)
            Ry = row_R[1] - uy * (0.5 * u_margin + k * step_along)

            # 已经相遇或交叉，结束本行
            if dot_projection((Rx - Lx, Ry - Ly, 0.0), ux, uy) < 0:
                break

            lpoint_projection = Lx * fx + Ly * fy
            rpoint_projection = Rx * fx + Ry * fy

            # 如果当前放置的点位判断出来与已经放置的点位或者预放置的点位距离太近，那么就不接受当前无人机放置该点位

            if(not second_replace) and (min(lpoint_projection, rpoint_projection) < Smin):
                uavs_budget += second_uav_num
                second_replace = True

            pair = [("left", (Lx, Ly, 0.0)), ("right", (Rx, Ry, 0.0))]
            # print("第", r, "行中的第", k, "个具体坐标是：", pair)

            # 如果不是左边先放，那就交换
            if not left_first:
                pair.reverse()

            # 依次尝试放置本对（左右）
            for side, cand in pair:
                if uavs_budget <= 0:
                    break  # 结束本对，随后结束本行与所有行

                if not too_close(cand, placed_local, preplaced_local, exclusion_radius_m):
                    dms = converter_enu.local_to_geodetic_dms(cand)
                    dms[2] = distance_m[1] + alt # 固定高度；如需要每行变化，这里改
                    placements_dms.append(dms)
                    placed_local.append(cand)
                    uavs_budget -= 1

                    if not second_replace:
                        min_projection = min(min_projection, lpoint_projection, rpoint_projection)

                if uavs_budget <= 0:
                    break  # 本行提前结束

            if uavs_budget <= 0:
                break

    # 如果标志位还是false，说明第2波次不需要接着第1波次放置无人机，第2波次无人机放置在敌群后沿
    if second_replace == False:
        for r in range(int(max_rows)):
            if uavs_budget2 <= 0:
                break
                # 计算从前沿移动到后沿需要的“行”数量

            # 在后沿处新起一行
            row_shift_x = f_back[0] * (r * row_offset)
            row_shift_y = f_back[1] * (r * row_offset)
            row_L = (pL_back[0] + row_shift_x, pL_back[1] + row_shift_y, 0.0)
            row_R = (pR_back[0] + row_shift_x, pR_back[1] + row_shift_y, 0.0)


            k_max = int((seg_len - u_margin) // (2.0 * step_along))
            for k in range(k_max + 1):
                if uavs_budget2 <= 0:
                    break
                offset = (0.5 * u_margin + k * step_along)
                Lx = row_L[0] + ux * offset
                Ly = row_L[1] + uy * offset
                Rx = row_R[0] - ux * offset
                Ry = row_R[1] - uy * offset

                pair = [("left", (Lx, Ly, 0.0)), ("right", (Rx, Ry, 0.0))]
                if not left_first:
                    pair.reverse()

                for side, cand in pair:
                    if uavs_budget2 <= 0:
                        break

                    if not too_close(cand, placed_local, preplaced_local, exclusion_radius_m):
                        dms = converter_enu.local_to_geodetic_dms(cand)
                        dms[2] = distance_m[1] + alt
                        placements_dms.append(dms)
                        placed_local.append(cand)
                        uavs_budget2 -= 1

    return placed_local, placements_dms


#以当前占位点构建局部坐标系中心点
def build_local_converter(point_dms):

    A_lat, A_lon, A_alt = GeodeticConverter.decimal_dms_to_degrees(point_dms)
    B_lat, B_lon, B_alt = A_lat + 1e-4, A_lon, A_alt
    converter_point = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, B_lat, B_lon, B_alt)

    return converter_point


#计算敌群朝向在前/右/上方向上的分量向量
def bearing_to_basis(bearing_deg: float):
    b = math.radians(bearing_deg)
    f = np.array([math.sin(b), math.cos(b), 0.0])     # 前（沿航向）
    r = np.array([math.cos(b), -math.sin(b), 0.0])    # 右
    u = np.array([0.0, 0.0, 1.0])                     # 上
    return f, r, u


#判断加速减速情况，是加速后直接减速还是加速后匀速再减速
def speed_situation_time_lower_bound(direction_vector, v0, v_max, v_target, a_acc, a_dec):

    #梯形速度，先加速再匀速再减速
    d_acc = max(0.0, (v_max**2 - v0**2) / (2*a_acc))
    d_dec = max(0.0, (v_max**2 - v_target**2) / (2*a_dec))

    if direction_vector > d_acc + d_dec:  # 梯形速度型：有巡航
        t_acc = max(0.0, (v_max - v0) / a_acc)
        t_cruise = (direction_vector - d_acc - d_dec) / max(v_max, 1e-6)
        t_dec = max(0.0, (v_max - v_target) / a_dec)

        return True, t_acc, t_cruise, t_dec, v_max

    # 三角速度型：解峰值速度
    denom = (1/(2*a_acc) + 1/(2*a_dec))
    v_peak_sq = (direction_vector + v0**2/(2*a_acc) + v_target**2/(2*a_dec)) / max(denom, 1e-9)
    v_peak = math.sqrt(max(v_peak_sq, 0.0))
    t_acc = max(0.0, (v_peak - v0) / a_acc)
    t_dec = max(0.0, (v_peak - v_target) / a_dec)

    return False, t_acc, 0.0, t_dec, v_peak


# 设定追击策略，第1波次无人机相当于追击第1波次占位点，第2波次无人机追击第2波次的对应占位点
def chase_strategy(occpy_dms, uav_after_turn_dms, first_uav_num, second_uav_num, enemy_bearing_deg, enemy_speed,
                   uav_start_speed=100.0,  # 我方起始空速 m/s
                   uav_max_speed=500.0,  # 我方最大空速 m/s
                   a_acc=80.0,  # 加速度
                   a_dec=80.0,  # 减速度
                   pos_tol=3.0,  # 空间位置收敛阈值m
                   vel_tol=1.0,  # 速度收敛阈值m/s（接近敌速）
                   dt=0.1,  # 仿真步长s
                   k_lead=0.6,  # 前视系数（0.4~0.8 常用）
                   distance_margin=20.0,  # 刹车安全距离 m
                   max_steps=200000):
    uav_tag = "None"
    if len(uav_after_turn_dms) == first_uav_num:
        uav_tag = "first"
    elif len(uav_after_turn_dms) == second_uav_num:
        uav_tag = "second"

    for uav, i in enumerate(uav_after_turn_dms):
        if uav_tag == "first":
            chase_info = pursue_moving_point(uav, occpy_dms[i], enemy_bearing_deg, enemy_speed, uav_start_speed, uav_max_speed, a_acc, a_dec,
                                             pos_tol, vel_tol, dt, k_lead, distance_margin, max_steps)
        if uav_tag == "second":
            chase_info = pursue_moving_point(uav, occpy_dms[i+first_uav_num], enemy_bearing_deg, enemy_speed, uav_start_speed, uav_max_speed, a_acc, a_dec,
                                             pos_tol, vel_tol, dt, k_lead, distance_margin, max_steps)

    return chase_info


# 追击函数，用于求解转弯后无人机到自己点位的最佳追击航向等
def pursue_moving_point(
        uav_dms,  # 我方无人机初始 DMS
        target_dms,  # 敌群西南角（或任一角/中心）初始 DMS
        enemy_bearing_deg,  # 敌群航向（正北0°、顺时针）
        enemy_speed,  # 敌群地速 m/s
        uav_start_speed=100.0,  # 我方起始空速 m/s
        uav_max_speed=500.0,  # 我方最大空速 m/s
        a_acc=80.0,  # 加速度 m/s^2
        a_dec=80.0,  # 减速度 m/s^2
        pos_tol=3.0,  # 空间位置收敛阈值 m
        vel_tol=1.0,  # 速度收敛阈值 m/s（接近敌速）
        dt=0.1,  # 仿真步长 s
        k_lead=0.6,  # 前视系数（0.4~0.8 常用）
        distance_margin=20.0,  # 刹车距离安全裕度 m
        max_steps=200000  # 最长仿真步数（防止死循环）
):

    # 以当前敌群点为中心建立右手坐标系
    converter_point = build_local_converter(target_dms)

    # 将当前无人机位置以及目标占位位置转local
    uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(uav_dms)
    uav_local = converter_point.geodetic_to_local(uav_lat, uav_lon, uav_alt)
    target_lat, target_lon, target_alt = GeodeticConverter.decimal_dms_to_degrees(target_dms)
    target_local = converter_point.geodetic_to_local(target_lat, target_lon, target_alt)

    # 计算敌群飞行方向的三维向量
    f_target, _, _ = bearing_to_basis(enemy_bearing_deg)
    target_speed = float(enemy_speed)

    # 估计时间下界，也就是直接沿着敌群方向进行追击时的时间值
    start_direction_vector = np.linalg.norm(target_local - uav_local)
    tri_sign, time_acc_est, time_cruise_est, time_dec_est, speed_peak = speed_situation_time_lower_bound(
        start_direction_vector, uav_start_speed, uav_max_speed, target_speed, a_acc, a_dec
    )

    # 闭环追击
    t = 0.0
    v = float(uav_start_speed)
    S_uav = 0.0
    t_acc = t_cruise = t_dec = 0.0
    state = "acc"
    traj_uav = [uav_local.copy()]  # 仅需可视化时使用

    for _ in range(max_steps):
        # 敌点位置
        target_t_local = target_local + enemy_speed * t * f_target

        # 到目标的相对向量
        dir_vector_r = target_t_local - uav_local
        dist = float(np.linalg.norm(dir_vector_r))

        # 收敛判据：位置逼近 & 速度逼近敌速
        if dist < pos_tol and abs(v - target_speed) < vel_tol:
            break

        # 前视引导
        t_lead = float(np.clip(k_lead * dist / max(v, 1e-3), 0.3, 3.0))
        target_lead = target_local + enemy_speed * (t + t_lead) * f_target
        dvec = target_lead - uav_local
        dnorm = float(np.linalg.norm(dvec))
        if dnorm < 1e-6:
            break
        dir_hat = dvec / dnorm

        # 刹车距离（减到敌速）
        d_stop = 0.0 if v <= target_speed else (v * v - target_speed * target_speed) / (2 * a_dec)

        # 状态机：加/巡/减
        if dist <= d_stop + distance_margin:
            # 减速
            v_new = max(v - a_dec * dt, target_speed)
            state = "dec"
            t_dec += dt
        else:
            if v < uav_max_speed:
                v_new = min(v + a_acc * dt,uav_max_speed)
                state = "acc"
                t_acc += dt
            else:
                v_new = v
                state = "cruise"
                t_cruise += dt

        # 位置更新（我方）
        U = U + v_new * dt * dir_hat
        traj_uav.append(U.copy())

        # 时间/路程累计
        S_uav += v_new * dt
        v = v_new
        t += dt

    t_total = t
    S_e = enemy_speed * t_total

    # 可达性粗检查：若跑满步数仍未收敛，多半是几何+速度不可达或容差太严
    reached = (t < max_steps * dt)

    return {
        "reached": reached,
        "t_total": t_total,
        "t_acc": t_acc,
        "t_cruise": t_cruise,
        "t_dec": t_dec,
        "S_uav": S_uav,
        "S_enemy": S_e,
        "estimate_lower_bound": {
            "triangular": not tri_sign,
            "t_acc_est": time_acc_est,
            "t_cruise_est": time_cruise_est,
            "t_dec_est": time_dec_est,
            "v_peak_est": speed_peak,
            "R0": start_direction_vector
        },
        "traj_u_local": np.array(traj_uav)  # ENU轨迹，需可视化时使用
    }





# 第三批次（一架无人机）先进行一段加速之后的位置求取，加速结束后再用uav_timed_position计算位置
def third_uav_first_speed_up_timed_position(center_pos, enemy_center, uav_speed, uav_max_speed, acceleration):
    uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(center_pos)
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)

    #加速方向为与敌群迎头的方向，计算加速时间和加速距离
    speed_up_bearing = converter.calculate_flight_bearing(uav_lat, uav_lon, enemy_lat, enemy_lon)
    speed_up_time = (uav_max_speed - uav_speed) / acceleration
    speed_up_distance = uav_speed * speed_up_time + (1/2) * acceleration *speed_up_time ** 2

    #根据上面的信息计算加速过后的位置
    uav_speed_over_lat, uav_speed_over_lon, uav_speed_over_alt = converter.calculate_destination_point(uav_lat, uav_lon, uav_alt, speed_up_bearing,
                                                                                                 speed_up_distance, 0)
    uav_speed_up_over_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(uav_speed_over_lat, uav_speed_over_lon, uav_speed_over_alt))
    return uav_speed_up_over_dms


# 第一波次从起始点的定时位置（关键时间：【加速结束（匀速开始）】，匀速结束（减速开始），减速结束（相遇），转弯结束（追击开始），追击结束）
def uav_timed_position(first_center_pos, uav_speed, uav_max_speed, uav_deceleration_speed, enemy_center,enemy_chase_center, acceleration, state_end_times):
    #通过前两个时间点是否一致，来判断是第一二波还是第三波无人机，不一致时第三波次，有加速阶段
    if state_end_times[0] != state_end_times[1]: #前两个状态时间不一样，说明有加速的过程
        flag = 1
    else:
        flag = 0
    if flag == 1:
        #重新由第三波次初始位置计算加速阶段过后的位置，作为此函数的first_center_pos
        first_center_pos = third_uav_first_speed_up_timed_position(first_center_pos, enemy_center, uav_speed, uav_max_speed, acceleration)
    else:
        pass

    #转换波次中心、敌群中心、敌群被追击的位置、敌群朝向
    timed_position = []
    first_lat, first_lon, first_alt = GeodeticConverter.decimal_dms_to_degrees(first_center_pos)
    print(f"first_pos:",first_lat, first_lon, first_alt)
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    enemy_chase_lat, enemy_chase_lon,enemy_chase_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_chase_center)
    bearing = converter.calculate_flight_bearing(first_lat, first_lon, enemy_lat, enemy_lon)

    # 匀速结束的位置
    uniform_distance = uav_speed * (state_end_times[2] - state_end_times[1])
    uniform_over_lat, uniform_over_lon, uniform_over_alt = converter.calculate_destination_point(first_lat, first_lon,
                                                                                                 first_alt, bearing,
                                                                                                 uniform_distance, 0)
    #减速结束的位置
    dec_over_time = state_end_times[3] - state_end_times[2]
    dec_over_distance = uav_speed * dec_over_time - (1 / 2) * acceleration * dec_over_time ** 2
    dec_over_lat, dec_over_lon, dec_over_alt = converter.calculate_destination_point(uniform_over_lat, uniform_over_lon,
                                                                      uniform_over_alt, bearing, dec_over_distance, 0)
    # 转弯结束的位置
    # 总转弯角度/弧度，半径及航向角
    angle_deg = 180
    angle_deg_rad = math.radians(angle_deg)
    turning_radius = (uav_deceleration_speed ** 2) / (GRAVITY_EARTH * math.tan(math.radians(45)))
    bearing_deg = converter.calculate_flight_bearing(dec_over_lat, dec_over_lon, enemy_lat, enemy_lon)
    turn_over_lat, turn_over_lon, turn_over_alt = calculate_turning_position_with_bearing(
        dec_over_lat, dec_over_lon, dec_over_alt, turning_radius, angle_deg_rad, bearing_deg, direction='right')

    # 不同行动状态对应位置求取，用无人机状态的最大时间来设定标准时间
    standard_time = list(range(0, max(state_end_times)))

    # 遍历每个标准时间点，判断三波次无人机的状态
    for time in standard_time:
        if time <= state_end_times[2]: #匀速阶段
            # 计算位移
            uniform_process_distance = uav_speed * time
            #匀速阶段time处的位置
            uav_chase_lat, uav_chase_lon, uav_chase_alt = converter.calculate_destination_point(first_lat, first_lon, first_alt, bearing, uniform_process_distance, 0)
           # timed_position.append((time, uav_chase_lat, uav_chase_lon, uav_chase_alt))
        elif state_end_times[2] < time <= state_end_times[3]:#减速阶段
            dec_time = time - state_end_times[2]
            dec_distance = uav_speed * dec_time - (1/2) * acceleration * dec_time ** 2

            #减速阶段time处的位置
            uav_chase_lat, uav_chase_lon, uav_chase_alt = converter.calculate_destination_point(uniform_over_lat, uniform_over_lon, uniform_over_alt,
                                                                                          bearing, dec_distance, 0)
           # timed_position.append((time, uav_chase_lat, uav_chase_lon, uav_chase_alt))
        elif state_end_times[3] < time <= state_end_times[4]: #转弯阶段 不接收数据不传出数据(开始转弯到开始追击)
            #当前时间以及移动的弧度
            turning_time = (angle_deg_rad * turning_radius) / uav_deceleration_speed
            elapsed_turn_time = time - state_end_times[3]
            angle_turned_rad = (elapsed_turn_time / turning_time) * angle_deg_rad #用时间比例计算当前位置

            # 转弯阶段time处的位置
            uav_chase_lat, uav_chase_lon, uav_chase_alt = calculate_turning_position_with_bearing(
                dec_over_lat, dec_over_lon, dec_over_alt, turning_radius, angle_turned_rad, bearing_deg, direction='right')
            #timed_position.append((time, uav_chase_lat, uav_chase_lon, uav_chase_alt))
        elif state_end_times[4] < time <= state_end_times[5]:  # 追击阶段 不接收数据不传出数据
            #追击加速阶段(暂时设定追击时间大于加速时间)
            chase_speed_up_time = (uav_max_speed-uav_deceleration_speed) / acceleration
            chase_bearing = converter.calculate_flight_bearing(turn_over_lat, turn_over_lon, enemy_chase_lat,
                                                               enemy_chase_lon)
            #时间在无人机加速时间段内的距离及位置计算
            if time <= state_end_times[4] + chase_speed_up_time:
                chase_distance = uav_deceleration_speed * (time - state_end_times[4]) + (1/2) * acceleration * (time - state_end_times[4]) ** 2
                uav_chase_lat, uav_chase_lon, uav_chase_alt = converter.calculate_destination_point(turn_over_lat, turn_over_lon, turn_over_alt,
                                                                                          chase_bearing, chase_distance, 0)
               # timed_position.append((time, uav_chase_lat, uav_chase_lon, uav_chase_alt))

            #加速至max之后的追击位置，需要先求出加速结束时刻的位置，再结合max速度计算后面的追击距离
            else:
                chase_speed_up_distance = uav_deceleration_speed * (chase_speed_up_time) + (1/2) * acceleration * (chase_speed_up_time) ** 2
                chase_speed_up_uav_lat, chase_speed_up_uav_lon, chase_speed_up_uav_alt = converter.calculate_destination_point(turn_over_lat,
                                                                                                    turn_over_lon,
                                                                                                    turn_over_alt,
                                                                                                    chase_bearing,
                                                                                                    chase_speed_up_distance, 0)
                chase_after_speed_up_distance = chase_speed_up_distance + uav_max_speed * (time - state_end_times[4]-chase_speed_up_time)
                uav_chase_lat, uav_chase_lon, uav_chase_alt = converter.calculate_destination_point(chase_speed_up_uav_lat, chase_speed_up_uav_lon, chase_speed_up_uav_alt,
                                                                                                    chase_bearing, chase_after_speed_up_distance, 0)
                #timed_position.append((time, uav_chase_lat, uav_chase_lon, uav_chase_alt))
        else:
            print(f"目前已完成追击任务")
        timed_position.append((time, uav_chase_lat, uav_chase_lon, uav_chase_alt))
    # print(f"某时刻无人机位置", timed_position)
    return timed_position

# 敌群定时输出位置
def enemy_timed_position(enemy_center, enemy_speed, uav_center, state_end_times):
    #转换敌群和无人机的位置形式
    output_enemy_timed_position = []
    uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(uav_center)
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    enemy_bearing = converter.calculate_flight_bearing(uav_lat, uav_lon, enemy_lat, enemy_lon)

    #设定标准时间，以无人机时间点最大值设定
    standard_time = list(range(0, max(state_end_times)))

    # 遍历每个标准时间点，判断三架无人机的状态
    for time in standard_time:
        enemy_action_distance = enemy_speed * time
        enemy_action_lat, enemy_action_lon, enemy_action_alt = converter.calculate_destination_point(enemy_lat, enemy_lon, enemy_alt, enemy_bearing, enemy_action_distance, 0)
        output_enemy_timed_position.append((time, enemy_action_lat, enemy_action_lon, enemy_action_alt))
    return output_enemy_timed_position


# 敌群在无人机转弯结束后的位置
# def uav_turned_enemy_timed_position(enemy_center, enemy_speed, uav_center, all_begin_to_turn_time):
#     #转换敌群和无人机的位置形式
#     output_enemy_timed_position = []
#     uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(uav_center)
#     enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
#     enemy_bearing = converter.calculate_flight_bearing(uav_lat, uav_lon, enemy_lat, enemy_lon)
#
#     # 遍历每个标准时间点，判断三架无人机的状态
#     for time in all_begin_to_turn_time:
#         enemy_action_distance = enemy_speed * time
#         enemy_action_lat, enemy_action_lon, enemy_action_alt = converter.calculate_destination_point(enemy_lat, enemy_lon, enemy_alt, enemy_bearing, enemy_action_distance, 0)
#         output_enemy_timed_position.append((time, enemy_action_lat, enemy_action_lon, enemy_action_alt))
#     return output_enemy_timed_position


#每一架无人机根据其转弯结束的时间确定追击后占位——具体位置（占位整体结构以确定，即与敌群的相对位置确定）
'''由于placements_dms为初始占位，第一架无人机转弯之后的占位结构（去除第二波次最后一架前置转弯的无人机）
    所以在这个函数中需要根据初始占位来确定每一架无人机转弯结束后需要追击的位置
    以placements_dms[0]，time_info[0]为起始位置/时间，第n个点随第n个时间差移动
    移动方式需要敌群速度和行动方向'''
def uav_turned_specific_position(enemy_center, enemy_speed, uav_center, time_info, placements_dms):
    #转换敌群和无人机的位置形式
    output_enemy_timed_position = []
    uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(uav_center)
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    enemy_bearing = converter.calculate_flight_bearing(uav_lat, uav_lon, enemy_lat, enemy_lon)

    #改变数值，由小到大排序，默认False
    time_info = sorted(time_info)

    # 遍历每个时间差，计算每一个架无人机应到过顶占位的位置
    for position_dms, time in zip(placements_dms, time_info):# 使用zip将两个列表中的项一一配对（占位结构和时间差）
        pos_lat, pos_lon, pos_alt = GeodeticConverter.decimal_dms_to_degrees(position_dms)

        # 计算敌群行动距离
        enemy_action_distance = enemy_speed * time

        # 计算我方每架无人机应在的目标位置
        enemy_action_lat, enemy_action_lon, enemy_action_alt = converter.calculate_destination_point(
            pos_lat, pos_lon, pos_alt, enemy_bearing, enemy_action_distance, 0
        )

        # 将结果添加到输出列表中，一开始也打印了time
        output_enemy_timed_position.append((enemy_action_lat, enemy_action_lon, enemy_action_alt))

    return output_enemy_timed_position
# 判断每个时间点下三波次无人机的状态,某一波次转弯过程中接收待转弯无人机中心（或完成追击的无人机中心）和敌群中心位置
# enemy_chase_center这个还没想好
def drone_state(uav1_state_end_times, uav2_state_end_times, uav3_state_end_times, uav_speed, uav_max_speed, enemy_speed, basepoint, enemy_center,
                enemy_chase_center, first_center_pos, second_center_pos, third_center_pos, chase_time_val=None):#max_distances, detect_distances, uavpoint,
    result = []
    information = []
    standard_time = list(range(0, max(uav2_state_end_times)))

    # 遍历每个标准时间点，判断三架无人机的状态
    for time in standard_time:
        # 记录状态
        uav3_status = get_uav_state(time, uav3_state_end_times, uav_number=3)
        uav1_status = get_uav_state(time, uav1_state_end_times, uav_number=1)
        uav2_status = get_uav_state(time, uav2_state_end_times, uav_number=2)

        #第三波次转弯且第一波次和第二波次未转弯时，需要接收1，2,e位置
        if uav3_status == "转弯" and uav1_status != "转弯" and uav2_status != "转弯":
            uav1_position = uav_timed_position(first_center_pos, uav_speed, uav_max_speed, uav_deceleration_speed, enemy_center, enemy_chase_center, acceleration, uav1_state_end_times)
            uav2_position = uav_timed_position(second_center_pos, uav_speed, uav_max_speed, uav_deceleration_speed, enemy_center, enemy_chase_center, acceleration, uav2_state_end_times)
            output_enemy_position = enemy_timed_position(enemy_center, enemy_speed, basepoint, uav2_state_end_times)
            # print(f"正在转弯，定时接收我方1,2和敌群中心位置",uav1_position, uav2_position, enemy_position)
            print(f"At time {time} UAV3 is turning, UAV1's position is {uav1_position}, UAV2's position is {uav2_position}, enemy's position is {output_enemy_position}")
            information.append(uav1_position)
            information.append(uav2_position)
            information.append(output_enemy_position)

        # 第一波次转弯且第二波次未转弯时，需要接收2,e位置
        elif uav1_status == "转弯" and uav2_status != "转弯": #需要接收2，e位置
            #担心第一批次正在转的时候，第二批次也开始转了，同时第三批次还没追上呢，那么要接收谁的位置信息呢？？？ 暂定全接收
            uav2_position = uav_timed_position(second_center_pos, uav_speed, uav_max_speed,uav_deceleration_speed, enemy_center,enemy_chase_center,acceleration,uav2_state_end_times)
            uav3_position = uav_timed_position(third_center_pos, uav_speed, uav_max_speed,uav_deceleration_speed, enemy_center,enemy_chase_center,acceleration,uav3_state_end_times)
            output_enemy_position = enemy_timed_position(enemy_center, enemy_speed, basepoint, uav2_state_end_times)
            print( f"At time {time} UAV1 is turning, UAV2's position is {uav2_position}, UAV3's position is {uav3_position}, enemy's position is {output_enemy_position}")
            information.append(uav2_position)
            information.append(output_enemy_position)

        # 第二波次转弯时，需要接收3,e位置
        elif uav2_status == "转弯": #需要接收3，e位置
            #这时候第三批追上了吗？第一批转完了吗？要接收谁的位置信息呢？暂定接收第三批次和敌群位置，但是这会儿敌群位置是谁给的呢？
            uav3_position = uav_timed_position(third_center_pos, uav_speed, uav_max_speed,uav_deceleration_speed, enemy_center,enemy_chase_center,acceleration,uav3_state_end_times)
            output_enemy_position = enemy_timed_position(enemy_center, enemy_speed, basepoint, uav2_state_end_times)
            information.append(uav3_position)
            information.append(output_enemy_position)
            print( f"At time {time} UAV2 is turning, UAV3's position is {uav3_position}, enemy's position is {output_enemy_position}")
        else:
            information = []

    result.append((time, uav1_status, uav2_status, uav3_status, information))#列表后面加位置
    return result


# 航向调整1.追赶过程中敌群速度高于我方转弯后速度；2.敌群速度与我方转弯后速度一致；3.敌群速度低于我方转弯后速度
# 航速调整：转弯之后与敌群成角度追击，需要在重点调整航向与敌群一致
# enemy_center是敌群被追上的位置，uav_pos追上敌群在其上方，海拔不确定
def fine_tuning(uav_speed, enemy_speed, acceleration,bearing_enemy, enemy_center, uav_pos):
    #航向和航速调整(航向参考模型3的整体航行方向)
    uav_bearing = bearing_enemy
    speed_diff = abs(enemy_speed - uav_speed)
    tune_distance = speed_diff * (enemy_speed + uav_speed) / (2 * acceleration)
    if enemy_speed != uav_speed:
        fine_speed_time = speed_diff / acceleration
        uav_speed = enemy_speed
    
    # 将我方敌群的海拔调整到敌群海拔上方的合理区间位置
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)#当前敌群位置
    uav_chase_lat, uav_chase_lon, uav_chase_alt = GeodeticConverter.decimal_dms_to_degrees(uav_pos)#当前我方位置（已完成追赶）
    min_height_diff = 50 #需要修改
    max_height_diff = 500 #需要修改
    #激光雷达实时测量敌我垂直高度current_diff，这里先通过计算得到差值
    # 情况1：高度差过小，需要上升
    current_diff = uav_chase_alt - enemy_alt
    if current_diff < min_height_diff:
        adjustment = min_height_diff - current_diff
        new_uav_chase_alt = uav_chase_alt + adjustment
    # 情况2：高度差过大，需要下降
    elif current_diff > max_height_diff:
        adjustment = current_diff - max_height_diff
        new_uav_chase_alt = uav_chase_alt - adjustment
    # 情况3：高度差合适，保持当前高度
    else:
        new_uav_chase_alt = uav_chase_alt
    #高度控制
    uav_chase_new_pos = [uav_chase_lat, uav_chase_lon,new_uav_chase_alt]
    print("uav_chase_new_pos:", uav_chase_new_pos)

    return uav_bearing, uav_speed, uav_chase_new_pos





#3D绘图便于观察
def plot_positions_with_centers(uav_first_geo_init, uav_second_geo_init,
                                enemy_center_init, last_uav,
                                meet_last_enemy_center, meet_last_uav_point,
                                after_turn_last_uav, chase_last_point, chase_enemy_center,converter):

    #==================================点位转坐标=============================================
    # 初始第1波点位 uav_dms, enemy_dms,
    first_uav_init_lats, first_uav_init_lons, first_uav_init_alts = geo_to_degrees(uav_first_geo_init)
    # 初始第2波点位
    second_uav_init_lats, second_uav_inti_lons, second_uav_init_alts = geo_to_degrees(uav_second_geo_init)


    #===================================敌群中心转坐标============================================
    # 初始中心（已知）
    enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center_init)#开始敌群中心
    last_enemy_center_lat, last_enemy_center_lon, last_enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(meet_last_enemy_center)#遇到最后一架无人机开始转弯敌群中心
    last_chase_enemy_lat, last_chase_enemy_lon, last_chase_enemy_alt = GeodeticConverter.decimal_dms_to_degrees(chase_enemy_center)#被最后一架无人机追上时的位置


    # ==================================最后一架无人机============================================
    last_lat, last_lon, last_alt = GeodeticConverter.decimal_dms_to_degrees(last_uav)#开始
    meet_last_lat, meet_last_lon, meet_last_alt = GeodeticConverter.decimal_dms_to_degrees(meet_last_uav_point)#刚与敌群相遇时的点位
    after_turn_last_lat, after_turn_last_lon, after_turn_last_alt = GeodeticConverter.decimal_dms_to_degrees(after_turn_last_uav)#转弯之后的点位
    chase_last_lat, chase_last_lon, chase_last_alt = GeodeticConverter.decimal_dms_to_degrees(chase_last_point)#追击后的点位


    #=====================================绘图==============================================
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # # 初始第1波
    # ax.scatter(first_uav_init_lats, first_uav_init_lons, first_uav_init_alts, c='cyan', marker='x', label='First UAV Init', s=50)
    #
    # # 初始第2波
    # ax.scatter(second_uav_init_lats, second_uav_inti_lons, second_uav_init_alts, c='blue', marker='o', label='Second UAV Init', s=50)
    #
    # # 初始敌群
    ax.scatter(enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt, c='red', marker='*', label='Enemy Init', s=50)
    ax.scatter(last_enemy_center_lat, last_enemy_center_lon, last_enemy_center_alt, c='green', marker='*', label='Turning Enemy Init', s=50)
    ax.scatter(last_chase_enemy_lat, last_chase_enemy_lon, last_chase_enemy_alt, c='orange', marker='*', label='Last chase enemy', s=50)

    #最后一架无人机
    # ax.scatter(last_lat, last_lon, last_alt, c='pink', marker='x', label='Second last UAV', s=50)
    ax.scatter(meet_last_lat, meet_last_lon, meet_last_alt, c='deepskyblue', marker='x', label='Meet last UAV', s=50)
    ax.scatter(after_turn_last_lat, after_turn_last_lon, after_turn_last_alt, c='navy', marker='x', label='After Turn last UAV', s=50)
    ax.scatter(chase_last_lat, chase_last_lon, chase_last_alt, c='purple', marker='x', label='Chase last UAV', s=50)


    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('3D Positions: Initial & New with Centers')
    ax.legend()
    plt.tight_layout()
    plt.show()
#无人机路径记录



#纬经高转化为坐标轴分量
def geo_to_degrees(geo):
    lats, lons, alts = [], [], [] #纬经高
    for i in geo:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(i)
        lats.append(lat)
        lons.append(lon)
        alts.append(alt)
    return lats, lons, alts
# 记录路径点


# 绘制群体散点+轨迹
# def plot_point_group(dms_list, label, color='orange', mode='markers+lines'):
#     lats, lons, alts = geo_to_degrees(dms_list)
#     return go.Scatter3d(
#         x=lons,
#         y=lats,
#         z=alts,
#         mode=mode,
#         marker=dict(size=4, color=color),
#         name=label,
#         text=[f"{label} {i}" for i in range(len(lats))],
#         hovertemplate=
#             f"<b>{label}</b><br>" +
#             "Lon: %{x}<br>Lat: %{y}<br>Alt: %{z} m<br><extra></extra>"
#     )


# 每架无人机三阶段航迹
def plot_uav_trajectories(init_geo, meet_geo, turn_geo, label_prefix="UAV", color="green"):
    traces = []
    #初始位置到相遇位置的直线航迹
    for i in range(2): #len(init_geo)
        points = [init_geo[i], meet_geo[i]] #, turn_geo[i]
        lats, lons, alts = [], [], []
        #将每一个点转为度数，分别转换每一架无人机的初始和相遇位置
        for dms in points:
            lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(dms)
            lats.append(lat)
            lons.append(lon)
            alts.append(alt)

        line_trace = go.Scatter3d(
            x=lons,
            y=lats,
            z=alts,
            mode='lines+markers',
            name=f"{label_prefix}_{i}",
            marker=dict(size=3, color=color),
            line=dict(color=color, width=2),
            text=[f"{label_prefix}_{i}"],
            hovertemplate=
                f"<b>{label_prefix}_{i}</b><br>" +
                "Lon: %{x}<br>Lat: %{y}<br>Alt: %{z} m<br><extra></extra>"
        )
        traces.append(line_trace)
    #生成半圆航迹

    for ii in range(2):
        lats, lons, alts = [], [], []
        # 提取相遇位置和转弯位置，分别转换为角度
        points = [meet_geo[ii], turn_geo[ii]]
        meet_lat, meet_lon, meet_alt = GeodeticConverter.decimal_dms_to_degrees(points[0])
        turn_lat, turn_lon, turn_alt = GeodeticConverter.decimal_dms_to_degrees(points[1])

        # 计算两点的中点
        midpoint_lat = (meet_lat + turn_lat) / 2
        midpoint_lon = (meet_lon + turn_lon) / 2

        # 计算两点之间的距离
        distance = converter.calculate_spherical_distance(meet_lat, meet_lon, meet_alt, turn_lat, turn_lon, turn_alt)  # 单位：公里
        radius = distance / 2  # 半径为两点之间的距离的一半
        # print("半径半径", radius)
        # 创建半圆轨迹

        for theta in np.linspace(0, 180, 10):  # 半圆角度从 0 到 180 度
            # 使用圆的参数方程计算经纬度，角度转弧度，“米”转经纬方向距离
            delta_lat = radius * np.cos(math.radians(theta)) / 111320   # 纬度
            delta_lon = radius * np.sin(math.radians(theta)) / (111320 * np.cos(np.radians(midpoint_lat)))  # 经度
            # print("度数变化:", theta, delta_lat, delta_lon)

            # 计算每个点的经纬度，从圆心开始计算
            lat = midpoint_lat + delta_lat
            lon = midpoint_lon + delta_lon
            alt = (turn_alt + meet_alt) / 2  # 假设高度是中点的高度

            #步长为10，所以这里append后一共是十个点
            lats.append(lat)
            lons.append(lon)
            alts.append(alt)

        # 创建轨迹图形
        # print("理应得到的点:",(lats[0], lons[0], alts[0]))
        arc_trace = go.Scatter3d(
            x=lons, y=lats, z=alts,
            mode='lines+markers',
            name=f"{label_prefix}_trajectory",
            marker=dict(size=3, color=color),
            line=dict(color=color, width=2),
            text=[f"{label_prefix}_point_{n}" for n in range(len(lats))],
            hovertemplate=f"<b>{label_prefix}</b><br>" + "Lon: %{x}<br>Lat: %{y}<br>Alt: %{z} m<br><extra></extra>"
        )
        traces.append(arc_trace)

    return traces

#从无人机或敌群位置中获取可视化散点位置
def create_uav_scatter(points, color):
    scatter_points = []

    # 从字典中提取经纬度和高度信息
    xs = [pt[1] for pt in points.values()]  # 经度
    ys = [pt[0] for pt in points.values()]  # 纬度
    zs = [pt[2] for pt in points.values()]  # 高度
    labels = list(points.keys())  # 获取所有的标签
    sizes = [3] * len(points)  # 设置所有点的大小
    colors = [color] * len(points)  # 设置所有点的颜色

    # 创建散点图
    for i in range(len(xs)):
        scatter_points.append(go.Scatter3d(
            x=[xs[i]],
            y=[ys[i]],
            z=[zs[i]],
            mode='markers+text',
            marker=dict(size=sizes[i], color=colors[i]),
            name=labels[i],
            # text=[labels[i]],
            textposition="top center",
            hovertemplate=(
                    f"<b>{labels[i]}</b><br>" +
                    "Lon: %{x}<br>Lat: %{y}<br>Alt: %{z} m<br><extra></extra>"
            )
        ))

    return scatter_points


def plot_positions(uav_first_geo_init, uav_second_geo_init,
                    enemy_center_init, meet_last_enemy_center, chase_enemy_center,
                    last_uav, meet_last_uav_point, after_turn_last_uav, chase_last_point,
                    meet_first_uav_point, after_turn_first_uav, meet_second_uav_point, after_turn_second_uav, corners, placements_dms, uav_base):


    # ==================================点位转坐标=============================================
    # 初始第1波点位 uav_dms, enemy_dms,
    first_uav_init_lats, first_uav_init_lons, first_uav_init_alts = geo_to_degrees(uav_first_geo_init)
    # 初始第2波点位
    # uav_second_geo_init.remove(last_uav)
    second_uav_init_lats, second_uav_init_lons, second_uav_init_alts = geo_to_degrees(uav_second_geo_init)
    base_lat, base_lon, base_alt = GeodeticConverter.decimal_dms_to_degrees(uav_base)

    # 使用 zip 函数将三个列表打包
    first_uav_positions = {
        f"First_UAV{i + 1}": [lat, lon, alt]  # 键名为 "First_UAVi"，值为 [纬度, 经度, 海拔]
        for i, (lat, lon, alt) in enumerate(zip(first_uav_init_lats, first_uav_init_lons, first_uav_init_alts))
    }

    # 输出第一波次字典
    # print("第一波次无人机位置字典：", first_uav_positions)

    # 使用 zip 函数将三个列表打包
    second_uav_positions = {
        f"Second_UAV{i + 1}": [lat, lon, alt]  # 键名为 "Second_UAVi"，值为 [纬度, 经度, 海拔]
        for i, (lat, lon, alt) in enumerate(zip(second_uav_init_lats, second_uav_init_lons, second_uav_init_alts))
    }

    # 输出第二波次字典
    # print("第二波次无人机位置字典：", second_uav_positions)
    # ===================================敌群中心转坐标============================================
    # 初始中心（已知）
    enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center_init)  # 开始敌群中心
    last_enemy_center_lat, last_enemy_center_lon, last_enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(meet_last_enemy_center)  # 遇到最后一架无人机开始转弯敌群中心
    last_chase_enemy_lat, last_chase_enemy_lon, last_chase_enemy_alt = GeodeticConverter.decimal_dms_to_degrees(chase_enemy_center)  # 被最后一架无人机追上时的位置

    # front_lat, front_lon, front_alt = GeodeticConverter.decimal_dms_to_degrees(placements[0])
    # p0_lat, p0_lon, p0_alt = GeodeticConverter.decimal_dms_to_degrees(placements[0][0])
    # p1_lat, p1_lon, p1_alt = GeodeticConverter.decimal_dms_to_degrees(placements[0][1])

    ws_lat, ws_lon, ws_alt = GeodeticConverter.decimal_dms_to_degrees(corners[0])
    wn_lat, wn_lon, wn_alt = GeodeticConverter.decimal_dms_to_degrees(corners[1])
    en_lat, en_lon, en_alt = GeodeticConverter.decimal_dms_to_degrees(corners[2])
    es_lat, es_lon, es_alt = GeodeticConverter.decimal_dms_to_degrees(corners[3])

    placement_lat, placement_lon, placement_alt = geo_to_degrees(placements_dms)




    # ==================================最后一架无人机============================================
    last_lat, last_lon, last_alt = GeodeticConverter.decimal_dms_to_degrees(last_uav)  # 开始
    meet_last_lat, meet_last_lon, meet_last_alt = GeodeticConverter.decimal_dms_to_degrees(meet_last_uav_point)  # 刚与敌群相遇时的点位
    after_turn_last_lat, after_turn_last_lon, after_turn_last_alt = GeodeticConverter.decimal_dms_to_degrees(after_turn_last_uav)  # 转弯之后的点位
    chase_last_lat, chase_last_lon, chase_last_alt = GeodeticConverter.decimal_dms_to_degrees(chase_last_point)  # 追击后的点位

    # ==================================第1波无人机相遇和转弯============================================
    meet_first_uav_lat, meet_first_uav_lon, meet_first_alt = geo_to_degrees(meet_first_uav_point)
    after_turn_first_lat, after_turn_first_lon, after_turn_first_alt = geo_to_degrees(after_turn_first_uav)
    # 使用 zip 函数将三个列表打包
    meet_first_uav_positions = {
        f"Meet_first_UAV{i + 1}": [lat, lon, alt]  # 键名为 "Meet_first_UAVi"，值为 [纬度, 经度, 海拔]
        for i, (lat, lon, alt) in enumerate(zip(meet_first_uav_lat, meet_first_uav_lon, meet_first_alt))
    }

    # 使用 zip 函数将三个列表打包
    turn_first_uav_positions = {
        f"Turn_first_UAV{i + 1}": [lat, lon, alt]  # 键名为 "Turn_first_UAVi"，值为 [纬度, 经度, 海拔]
        for i, (lat, lon, alt) in enumerate(zip(after_turn_first_lat, after_turn_first_lon, after_turn_first_alt))
    }

    # ==================================第2波无人机相遇和转弯============================================
    meet_second_uav_lat, meet_second_uav_lon, meet_second_alt = geo_to_degrees(meet_second_uav_point)
    after_turn_second_lat, after_turn_second_lon, after_turn_second_alt = geo_to_degrees(after_turn_second_uav)
    # 使用 zip 函数将三个列表打包
    meet_second_uav_positions = {
        f"Meet_second_UAV{i + 1}": [lat, lon, alt]  # 键名为 "Meet_first_UAVi"，值为 [纬度, 经度, 海拔]
        for i, (lat, lon, alt) in enumerate(zip(meet_second_uav_lat, meet_second_uav_lon, meet_second_alt))
    }

    # 使用 zip 函数将三个列表打包
    turn_second_uav_positions = {
        f"Turn_second_UAV{i + 1}": [lat, lon, alt]  # 键名为 "Turn_first_UAVi"，值为 [纬度, 经度, 海拔]
        for i, (lat, lon, alt) in enumerate(zip(after_turn_second_lat, after_turn_second_lon, after_turn_second_alt))
    }

    all_lons, all_lats, all_alts = [], [], []
    points = {
        # 'Last Enemy center ': [last_enemy_center_lat, last_enemy_center_lon, last_enemy_center_alt],
        # 'Last Chase Enemy': [last_chase_enemy_lat, last_chase_enemy_lon, last_chase_enemy_alt],
        # 'Meet Last UAV': [meet_last_lat, meet_last_lon, meet_last_alt],
        # 'After Turn Last UAV': [after_turn_last_lat, after_turn_last_lon, after_turn_last_alt],
        # 'Chase Last UAV': [chase_last_lat, chase_last_lon, chase_last_alt],
        # 'uav base': [base_lat, base_lon, base_alt],
        # 'fornt point':[front_lat, front_lon, front_alt],
        # 'p0':[p0_lat, p0_lon, p0_alt],
        # 'p1':[p1_lat, p1_lon, p1_alt],
        #下面三个点要继续改
        'Last uav':[last_lat, last_lon, last_alt],
        'Last uav meet enemy': [meet_last_lat, meet_last_lon, meet_last_alt],
        'Last uav turn':[after_turn_last_lat, after_turn_last_lon, after_turn_last_alt],
        'Enemy center init':[enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt],
        'ws':[ws_lat, ws_lon, ws_alt],
        'es':[es_lat, es_lon, es_alt],
        'en':[en_lat, en_lon, en_alt],
        'wn':[wn_lat, wn_lon, wn_alt],
    }

    # 提取坐标 这里如果有再多波次，需要重构字典
    scatter_points_0 = create_uav_scatter(points, color='magenta')#points里面的点，包括东南西北
    scatter_points_1 = create_uav_scatter(first_uav_positions, color='red')#第一波次无人机初始位置
    scatter_points_2 = create_uav_scatter(second_uav_positions, color='orange')#第二波次无人机初始位置
    scatter_points_3 = create_uav_scatter(meet_first_uav_positions, color='pink')#第一波次无人机与敌群相遇
    scatter_points_4 = create_uav_scatter(turn_first_uav_positions, color='yellow')  # 第一波次无人机转弯之后
    scatter_points_5 = create_uav_scatter(meet_second_uav_positions, color='gray')  # 第二波次无人机与敌群相遇
    scatter_points_6 = create_uav_scatter(turn_second_uav_positions, color='brown')  # 第二波次无人机转弯之后

    scatter_points = scatter_points_0 + scatter_points_1 + scatter_points_2 + scatter_points_3 + scatter_points_4 + scatter_points_5 + scatter_points_6#合并字典

    #存储所有点，为了找坐标端点
    merged_dict = {**points, **first_uav_positions, **second_uav_positions}#解压
    # print("merged_dict位置字典", merged_dict)
    # 创建空的列表来分别存储纬度、经度和高度
    lats = []
    lons = []
    alts = []

    # 遍历字典并提取每个 UAV 的位置
    for key, value in merged_dict.items():
        a, b, c = value  # 每个 UAV 的位置值是一个包含纬度、经度和高度的列表
        lats.append(a)
        lons.append(b)
        alts.append(c)

    # 2) 生成标签文本（显示 UAV 编号）
    pl_labels = [f"U {i + 1}" for i in range(len(placement_lat))]

    # 3) 合并成一个 Scatter3d（效率高）
    scatter_points.append(go.Scatter3d(
        x=placement_lon,
        y=placement_lat,
        z=placement_alt,
        mode='markers+text',  # 如果太挤，可改 'markers'
        marker=dict(size=4, color='green'),  # 统一样式；也可用 colorscale
        name='UAV placements',
        text=pl_labels,  # 点旁边显示编号
        textposition="top center",
        hovertemplate=(
            "<b>%{text}</b><br>"  # %{text} 就是 pl_labels
            "Lon: %{x}<br>"
            "Lat: %{y}<br>"
            "Alt: %{z} m<br><extra></extra>"
        )
    ))

    # ===== 多无人机轨迹（三阶段）=====
    # 绘制第二波次无人机的航迹()
    uav_first_trajectories = plot_uav_trajectories(
        uav_first_geo_init, meet_first_uav_point, after_turn_first_uav,
        label_prefix="First UAV", color='green')

    # 删除 last_uav 后的 uav_second_geo_init
    # uav_second_geo_init.remove(last_uav)
    # 绘制第二波次无人机的航迹
    uav_second_trajectories = plot_uav_trajectories(
        uav_second_geo_init, meet_second_uav_point, after_turn_second_uav,
        label_prefix="Second UAV", color='blue')

    # 合并两波次的轨迹
    all_uav_trajectories = uav_first_trajectories + uav_second_trajectories
    #     # 示例轨迹线：你可以换成更复杂的路径
    # path = go.Scatter3d(
    #     x=[points['After Turn Last UAV'][1], points['Chase Last UAV'][1]],
    #     y=[points['After Turn Last UAV'][0], points['Chase Last UAV'][0]],
    #     z=[points['After Turn Last UAV'][2], points['Chase Last UAV'][2]],
    #     mode='lines',
    #     line=dict(color='black', width=4),
    #     name='UAV Turn Path'
    # )
    #
    # # 提取 First UAV 轨迹坐标
    # for group in [uav_first_geo_init, meet_first_uav_point, after_turn_first_uav]:
    #     for dms in group:
    #         lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(dms)
    #         all_lats.append(lat)
    #         all_lons.append(lon)
    #         all_alts.append(alt)

    # 加入关键点坐标（所有的经纬海拔区间，为了可视化的坐标端点）
    all_lats.extend(lats)
    all_lons.extend(lons)
    all_alts.extend(alts)
    print(all_lons)
    # print("显示最大最小值:",min(all_lons), max(all_lons), min(all_lats), max(all_lats))

    # 绘制图形
    # print("scatter points:", scatter_points)
    fig = go.Figure(scatter_points + all_uav_trajectories)   #+ [path]
    # 设置显示参数
    fig.update_layout(
        scene=dict(
            xaxis_title='Longitude (°E)',
            yaxis_title='Latitude (°N)',
            zaxis_title='Altitude (m)',

            xaxis=dict(range=[min(lons) - 0.1, max(lons) + 0.1]),#  124.1497,124.1897
            yaxis=dict(range=[min(lats) - 0.1, max(lats) + 0.1]),#   29.67, 29.71
            zaxis=dict(range=[min(all_alts) - 1000, max(all_alts) + 1000]),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        title='Geodetic UAV vs Enemy Visualization',
        showlegend=True
    )

    fig.show()


if __name__ == "__main__":
    #=============================初始化数据==================================
    data = dataset()
    acceleration = 80 #加速度减速度均设为80
    uav_deceleration_speed = 100 #假设无人机减速减到100
    #============================创建局部坐标系================================
    #最开始的局部坐标系（局部坐标系用于判断一些细节问题）
    converter, uav_traj, enemy_traj = simulate_relative_motion(
        data['basepoint'],
        data['enemy_approx'],
        data['basepoint'],
        data['minimum_speed'],
        data['speed'],
        0,
        10
    )
    #=======================处理第2波次无人机纵队最后一架无人机===============================
    #对第2波次无人机进行聚类，得到纵队情况
    cluster_second_uav = cluster_uavs_by_latitude(data['second_uavs'], converter) #聚类
    num_columns = len(cluster_second_uav) #得到类别数

    #求第2波次无人机中心
    second_uav_center = calculate_center_dms(data['second_uavs'])
    print(f"第二波次中心：", second_uav_center)
    #对第2波次无人机按y从小到大进行排序，得到sorted_second[0]就是末尾那架无人机
    second_uav_sorted, second_sorted_dms = uav_sorted_distances_points(data['second_uavs'], second_uav_center, data['enemy_approx'], reverse = True)
    # print("最后一架无人机:", second_uav_sorted)

    #已知敌机中心和经纬度范围求得敌机距离base的最大距离，即最远边界值
    max_distance, max_enemy_dms, min_distance, min_enemy_dms = max_distance(data['basepoint'], data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange'])
    # print("输出最大距离最小距离:",max_distance, max_dms, min_distance, min_dms)

    #计算当第1波次无人机能将敌方全纳入探测范围时的时间
    time_detect = function_last_detection_time(data['minimum_speed'], data['speed'], max_distance, data['detect_distance'])
    # print("time_detect = ", time_detect)

    last_point = second_sorted_dms[0]
    print("*******", last_point)
   
    #计算最后一架无人机转弯起点位置、转弯180度后位置、追赶位置以及敌方中心在我方无人机开始转弯时位置、被追赶上位置；转弯时间，追逐时间

    last_begin_turn_dms, after_turning_last_uav_dms, chase_uav_dms, meet_last_enemy_dms, chase_enemy_dms, turning_time, chase_time, last_time_info = last_uav_move_strategy(
        data['minimum_speed'], data['maximum_speed'], uav_deceleration_speed, data['speed'],
        max_distance, data['detect_distance'],last_point, data['basepoint'], min_enemy_dms, acceleration)
    # print("输出时间信息:", last_time_info)

    # =================================处理第1波次无人机===========================================

    #求第1波次无人机中心
    first_uav_center = calculate_center_dms(data['first_uavs'])
    print(f"第一波次中心：", first_uav_center)
    # 对第一批次无人机进行排序，距离敌群由近到远，并计算相遇时间（包含安全距离）
    first_uav_sorted, first_sorted_dms = uav_sorted_distances_points(data['first_uavs'], first_uav_center, max_enemy_dms, reverse = False)

    meet_first_uav_dms, after_turn_first_uav_dms, first_uav_time_info, bearing_enemy, meet_single_time1 = first_uav_move_strategy(data['minimum_speed'], uav_deceleration_speed, data['maximum_speed'], data['speed'], first_uav_sorted, first_uav_center,
                             max_enemy_dms, 1000, acceleration, time_detect)
    # print("第一波次与敌群相遇点：", meet_first_uav_dms)
    # print("第一波次转完之后的时间", first_uav_time_info[4])
    # =================================处理第2波次无人机===========================================

    remain_second_uav_sorted = second_uav_sorted[1:]
    second_rest_uav  = [entry[3] for entry in remain_second_uav_sorted]
    remain_second_uav_sorted, remain_second_sorted_dms = uav_sorted_distances_points(second_rest_uav, second_uav_center,
                                                                       data['enemy_approx'], reverse=False)
    # print("第二波次去除最后一架之后的排序:", remain_second_uav_sorted, remain_second_sorted_dms)
    meet_second_uav_dms, after_turn_second_uav_dms, second_uav_time_info, bearing_enemy2, meet_single_time2= first_uav_move_strategy(data['minimum_speed'], uav_deceleration_speed, data['maximum_speed'], data['speed'], remain_second_uav_sorted, second_uav_center,
                             max_enemy_dms, 1000, acceleration, time_detect)

    #占位策略，得到无人机上的占位信息，便于追击
    front_dms, corners, placements_dms = generate_placements_with_bearing(
        data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange'],
        distance_m = data['near_detect_distance'],
        bearing_deg = bearing_enemy,  #敌方从东向西
        exclusion_radius_m = None,
        preplaced_dms=data['enemy_approx'],
        start_side='left',  #先左
        max_rows=100, first_uav_num=data['first_num'], second_uav_num=data['second_num']
    )
    # print("根据敌群初始位置求得固定占位", placements_dms)
    #===============计算两波次无人机从敌群进入视场到转弯结束的时间点====================
    enter_to_meet_time = meet_single_time1 + meet_single_time2
    #print("两波次无人机从敌群进入视场到转弯结束的时间点", enter_to_meet_time)
    output_each_enemy_pos = uav_turned_specific_position(data['enemy_approx'], data['speed'], data['basepoint'], enter_to_meet_time, placements_dms)
    print("每一架无人机转弯后应追击的占位", output_each_enemy_pos)

    time_uav_info = generate_uav_time_info(last_time_info, first_uav_time_info, second_uav_time_info)
    print("总时间信息表！！！！！:", time_uav_info) #包括时间，纬度/经度/海拔







    plot_positions(first_sorted_dms, remain_second_sorted_dms,
                   data['enemy_approx'], meet_last_enemy_dms, chase_enemy_dms,
                   last_point, last_begin_turn_dms, after_turning_last_uav_dms, chase_uav_dms,
                   meet_first_uav_dms, after_turn_first_uav_dms, meet_second_uav_dms, after_turn_second_uav_dms,corners, placements_dms, data['basepoint'])




    #数据输出
    outdata.save_uav_multi_positions(data['minimum_speed'], acceleration, data['maximum_speed'], 100, data['speed'],
                                     data['first_uavs'], data['second_uavs'], data['enemy_approx'],
                                     last_begin_turn_dms, after_turning_last_uav_dms, chase_uav_dms,
                                     time_detect, turning_time, chase_time,
                                     None, None, None,
                                     None, None, None,
                                     None, None, None,
                                     None, None, None)

    #===================================相对位置===============================================

    # # 第3波次无人机（加速、匀速、减速、转弯、追击上的结束时间） 现在少一个
    # uav3_state_end_times = [3, 7, 9, 11, 13, 15]  # 第3波次无人机的加速、匀速、减速、转弯的结束时间
    # # 第1/2波次无人机（匀速、减速、转弯、追击上的结束时间） 现在少一个
    # uav1_state_end_times = [5, 10, 15, 16, 17, 18]  # 第1波次无人机的匀速、减速、转弯的结束时间
    # uav2_state_end_times = [6, 12, 17, 18, 19, 20]  # 第二波次无人机的匀速、减速、转弯的结束时间
    # enemy_chase_center = [
    #         "120:38:15.75E",
    #          "29:46:10.10N",
    #         "3976.86"
    #     ]#后面商量怎么改
    # first_center_pos = [
    #         "120:38:15.75E",
    #          "29:46:10.10N",
    #         "3976.86"
    #     ] #已知
    # second_center_pos = [
    #         "120:38:15.75E",
    #          "29:46:10.10N",
    #         "3976.86"
    #     ] #已知
    # third_center_pos = [
    #     "120:38:14.82E",
    #     "29:47:22.84N",
    #      "5005.0"
    #     ] #已知
    # # 调用函数
    # #######################测试相对位置函数#########################
    # speed_up_over_dms = third_uav_first_speed_up_timed_position(third_center_pos, data['enemy_approx'],
    #                                                             data['minimum_speed'], data['maximum_speed'],
    #                                                             acceleration)
    # print(f"加速之后的位置:", speed_up_over_dms,first_center_pos, third_center_pos,data['basepoint'])
    # timed_pos = uav_timed_position(data['basepoint'], data['minimum_speed'], data['maximum_speed'], uav_deceleration_speed, data['enemy_approx'],
    #                    enemy_chase_center, acceleration, uav3_state_end_times)
    # print(f"timed_pos定时位置:", timed_pos)#维度为负数的问题需不需要解决？
    # output_enemy_timed_position = enemy_timed_position(data['enemy_approx'], data['speed'], data['basepoint'], uav2_state_end_times)
    # print(f"enemy_timed_position定时位置:", output_enemy_timed_position)
    # relative_position = drone_state(uav1_state_end_times, uav2_state_end_times, uav3_state_end_times, data['minimum_speed'], data['maximum_speed'], data['speed'],
    #             data['basepoint'], data['enemy_approx'],
    #             enemy_chase_center, first_center_pos, second_center_pos, third_center_pos,
    #             chase_time_val=None) # max_distances, detect_distances, uavpoint,
    # print(f"relative_position相对位置:", relative_position)
    ####################上面在测试#############################
    # state_result = drone_state(uav1_state_end_times, uav2_state_end_times, uav3_state_end_times, data['minimum_speed'], data['maximum_speed'], data['speed'], data['basepoint'], data['enemy_approx'],
    #             enemy_chase_center, first_center_pos, second_center_pos, third_center_pos, chase_time_val=None)
    # # 输出结果
    # for time, uav1_status, uav2_status, uav3_status, information in state_result:
    #     print(f"Time {time}: UAV1 is {uav1_status}, UAV2 is {uav2_status}, UAV3 is {uav3_status},convey information is {information}")





