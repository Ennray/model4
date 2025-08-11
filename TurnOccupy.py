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


GRAVITY_EARTH = 9.80665  # 地球表面重力加速度
R = 6371000  # 地球半径，单位：米



#对第1批或第2批uav进行排序，由敌群的近到远或由远到近
def uav_sorted_distances_points(uav_points, uav_center, enemy_center, reverse):
    first = []
    first_sorted_dms = []
    i = 0

    #将我方中心，敌方中心都转度数
    uav_center_lat, uav_center_lon, uav_center_alt = GeodeticConverter.decimal_dms_to_degrees(uav_center)
    enemy_center_lat, enemy_center_lon, enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)

    #计算整体飞行方向
    flight_bearing = converter.calculate_flight_bearing(uav_center_lat, uav_center_lon, enemy_center_lat,enemy_center_lon)

    for point in uav_points:

        lat, lon ,alt = GeodeticConverter.decimal_dms_to_degrees(point)

        #每架无人机的飞行方向，并将其投影到整体移动方向上（适用于几百公里内）
        uav_bearing = converter.calculate_flight_bearing(lat, lon, enemy_center_lat, enemy_center_lon)
        delta_angle = abs(flight_bearing - uav_bearing)
        delta_angle = min(delta_angle, 360 - delta_angle)

        #计算每架无人机飞行球面距离
        dist = converter.calculate_spherical_distance(lat, lon, alt, enemy_center_lat, enemy_center_lon, alt)
        projected_dist = dist * math.cos(math.radians(delta_angle))

        first.append([projected_dist, i, 0, point])
        i = i + 1

    first_sorted = sorted(first, key=lambda item: item[0], reverse = reverse)#按投影后的距离由小到大（False）的顺序进行排序
    for point in first_sorted:
        first_sorted_dms.append(point[3])

    return first_sorted, first_sorted_dms

#计算第1波次或第2波次无人机中心
def calculate_center_dms(uav_dms):

    decimal_positions = []

    #遍历每个函数，得到度数分量
    for dms_pos in uav_dms:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(dms_pos)
        decimal_positions.append((lat, lon, alt))

    #对所有分量求平均值
    lats = [p[0] for p in decimal_positions]
    lons = [p[1] for p in decimal_positions]
    alts = [p[2] for p in decimal_positions]

    center_lat = np.mean(lats)
    center_lon = np.mean(lons)
    center_alt = np.mean(alts)

    #分量转local再转dms
    uav_center_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(center_lat, center_lon, center_alt))

    return uav_center_dms


#聚类分析有多少个纵队并分别记录
def cluster_uavs_by_latitude(second_points, converter, eps = 20):
    #先将second_point转为坐标系
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


#计算敌群最远边界信息，获得敌机边界四个角的信息
def get_enemy_edges(enemy_center, lat_range, lon_range):
    ne = [lon_range[1], lat_range[1], enemy_center[2]]  # 北 + 东
    se = [lon_range[1], lat_range[0], enemy_center[2]]  # 南 + 东
    nw = [lon_range[0], lat_range[1], enemy_center[2]]  # 北 + 西
    sw = [lon_range[0], lat_range[0], enemy_center[2]]  # 南 + 西
    return ne, se, nw, sw



def to_local(pos_dms, converter):
    lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(pos_dms)
    return converter.geodetic_to_local(lat, lon, alt)


#计算第1波无人机距离敌群的最远距离
def max_distance(basepoint, enemy_center, lat_range, lon_range):
    #敌方四个角dms格式
    ws = [lon_range[1], lat_range[1], enemy_center[2]]  # 西+南
    wn = [lon_range[1], lat_range[0], enemy_center[2]]  # 西+北
    es = [lon_range[0], lat_range[1], enemy_center[2]]  # 东+南
    en = [lon_range[0], lat_range[0], enemy_center[2]]  # 东+北
    corners = [ws, wn, es, en]

    #转换base为度数
    base_lat, base_lon, base_alt = GeodeticConverter.decimal_dms_to_degrees(basepoint)

    #初始化最小最大距离
    max_dist = -float('inf')
    min_dist = float('inf')
    max_dms = None
    min_dms = None

    for corner_dms in corners:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(corner_dms)
        dist = converter.calculate_spherical_distance(base_lat, base_lon, base_alt, lat, lon, alt)

        #有最大或最小的的就记录
        if dist > max_dist:
            max_dist = dist
            max_dms = corner_dms

        if dist < min_dist:
            min_dist = dist
            min_dms = corner_dms

    return max_dist, max_dms, min_dist, min_dms


#求解探测到敌方最后沿时的时间
def function_last_detection_time(uav_speed, enemy_speed, max_distances, detect_distances):
    relative_speed = uav_speed + enemy_speed
    time = (max_distances - detect_distances ) / relative_speed
    return time


#计算最后一架无人机需要飞出的总距离(暂时用不上)
def last_uav_distance(basepoint, uavpoint, uav_speed, enemy_speed, max_distances, detect_distances, converter):
    pos_base = to_local(basepoint, converter)

    dis_uav_base = pos_base[1] - uavpoint
    time = function_last_detection_time(uav_speed, enemy_speed, max_distances, detect_distances)
    dis_last_uav = dis_uav_base + uav_speed * time + detect_distances
    return dis_last_uav


#敌我同时推进，每 step_time 秒更新一次位置，重建 converter。
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


def calculate_turning_position_with_bearing(lat, lon, alt, turning_radius, angle_deg, bearing_deg, direction='left'):

    # Step 1: 计算圆心方向（偏移航向±90度）
    offset_bearing = (bearing_deg + (90 if direction == 'left' else -90)) % 360

    # Step 2: 计算圆心点
    center_point = distance(meters=turning_radius).destination(Point(lat, lon), bearing=offset_bearing)

    # Step 3: 计算终点方向：从圆心开始，按方向旋转角度
    arc_end_bearing = (offset_bearing + (angle_deg if direction == 'left' else -angle_deg)) % 360

    # Step 4: 沿圆弧从圆心出发，回到圆周上
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


#计算转弯后进行追赶的时间距离及敌群移动距离
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


#纵队最后一架无人机的飞行策略
def last_uav_move_strategy(uav_speed, uav_max_speed, uav_deceleration_speed, enemy_speed, max_distances, detect_distances, uavpoint, basepoint, min_enemy_dms, acceleration):

    last_time_info = []
    #飞行时间估计（第1波次无人机将敌方全纳入视场时间）
    time = function_last_detection_time(uav_speed, enemy_speed, max_distances, detect_distances)
    last_time_info.append(time) #匀速时间点

    #匀加速时间与加速距离
    time_acc = (uav_max_speed - uav_speed) / acceleration  # 得到加速时间
    distance_acc = (uav_max_speed ** 2 - uav_speed ** 2) / (2 * acceleration)  # 得到加速期间前进的距离
    last_time_info.append(time_acc + last_time_info[0]) #加速时间点

    #加速前已经飞行距离
    total_distances = uav_speed * time + distance_acc

    #当前无人机位置
    lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(uavpoint)

    #敌群中心位置
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(min_enemy_dms)

    #求飞行方向
    bearing =  converter.calculate_flight_bearing(lat, lon, enemy_lat, enemy_lon)
    print(f"<UNK> = {bearing:.6f}")
    bearing_enemy = converter.calculate_flight_bearing(enemy_lat, enemy_lon, lat, lon)

    #球面预测加速后飞行点
    new_lat, new_lon, new_alt = converter.calculate_destination_point(
        lat, lon, alt, bearing, total_distances, 0)

    #敌群位置更新（反方向飞行）
    enemy_movedis = enemy_speed * (time + time_acc)
    new_enemy_lat, new_enemy_lon, new_enemy_alt = converter.calculate_destination_point(
        enemy_lat, enemy_lon, enemy_alt, bearing_enemy, enemy_movedis, 0)

    #计算此时相对距离
    distance_move = converter.calculate_spherical_distance(new_lat, new_lon, new_alt, new_enemy_lat, new_enemy_lon, new_enemy_alt)

    safety_distance = 1000

    #计算减速时间和减速距离
    deceleration_time = (uav_max_speed - uav_deceleration_speed) / acceleration
    uav_deceleration_distence = uav_max_speed * deceleration_time - 0.5 * acceleration *deceleration_time **2

    #匀速减速再相遇所需要的总时间 = （不减速时的距离 - 安全距离 + 不减速时的距离与考虑减速时的距离之差） / 相对速度
    time_move = (distance_move - safety_distance + uav_max_speed * deceleration_time - uav_deceleration_distence) / (uav_max_speed + enemy_speed)

    #无人机最大速度匀速前进距离
    distance_uav = uav_max_speed * (time_move - deceleration_time)

    #匀速时间
    time_uniform = time_move - deceleration_time
    last_time_info.append(time_uniform + last_time_info[1]) #继续匀速得到时间点
    last_time_info.append(deceleration_time + last_time_info[2]) #得到减速时间点也是转弯时间点

    #敌方移动总距离
    distance_enemy = enemy_speed * time_move

    #相遇时敌机中心位置更新（反方向飞行）
    meet_enemy_lat, meet_enemy_lon, meet_enemy_alt = converter.calculate_destination_point(
        new_enemy_lat, new_enemy_lon, new_enemy_alt, bearing_enemy, distance_enemy, 0
    )

    #在减速到达敌方前就爬升至敌方高度上方
    angle = converter.calculate_climb_angle(new_alt, new_enemy_alt + safety_distance, distance_uav)
    angle_rad = math.radians(angle)

    #开始转弯的点位（相遇位置）
    uav_meet_lat, uav_meet_lon, uav_meet_alt, distance_uav_val= converter.calculate_destination_with_climb_angle(new_lat, new_lon, new_alt, bearing, distance_uav, angle)

    # 当前假设转弯180度
    angle_deg = 180
    angle_deg_rad = math.radians(angle_deg)

    #计算转弯半径
    turning_radius = (uav_deceleration_speed**2)/ (GRAVITY_EARTH * math.tan(math.radians(45))) #  (uav_deceleration_speed**2) * (math.cos(angle_rad)**2))
    print("半径", turning_radius)
    turning_time = (angle_deg_rad * turning_radius) / uav_deceleration_speed
    print("转弯时间", turning_time)
    last_time_info.append(turning_time + last_time_info[3])

    #转弯时我方无人机及敌方无人机转dms
    last_begin_turn_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(uav_meet_lat, uav_meet_lon, uav_meet_alt))
    meet_last_enemy_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(meet_enemy_lat, meet_enemy_lon, meet_enemy_alt))


    bearing_rel =  converter.calculate_flight_bearing(uav_meet_lat, uav_meet_lon, enemy_lat, enemy_lon)
    print("输出两个坐标",uav_meet_lon,enemy_lon)
    print("开始转弯时无人机的位置：", last_begin_turn_dms)
    print("最开始敌群最近位置:",min_enemy_dms)
    print("航向角：", bearing_rel)

    #得到转弯angle_deg后的无人机dms位置
    after_turning_last_uav_dms = after_turn_position(turning_radius, last_begin_turn_dms, angle_deg, bearing_rel, direction = 'right')
    turn_uav_lat, turn_uav_lon, turn_uav_alt = GeodeticConverter.decimal_dms_to_degrees(after_turning_last_uav_dms)

    #得到转弯后追击敌方时所需要花费的时间距离等
    chase_time_val, uniform_time_val, chase_distance_val, distance_enemy_chase_val = after_turn_chase(
        uav_deceleration_speed, uav_max_speed, enemy_speed, turning_time, turning_radius, acceleration)

    #得到追击到敌方时敌我两方位置
    chase_enemy_lat, chase_enemy_lon, chase_enemy_alt = converter.calculate_destination_point(
        meet_enemy_lat, meet_enemy_lon, meet_enemy_alt, bearing_enemy, distance_enemy_chase_val, 0)

    chase_last_uav_lat, chase_last_uav_lon, chase_last_uav_alt = converter.calculate_destination_point(
        meet_enemy_lat, meet_enemy_lon, meet_enemy_alt + safety_distance, bearing_enemy, distance_enemy_chase_val, 0)

    #转弯后此时敌我两方的经纬度位置
    chase_enemy_dms = converter.local_to_geodetic_dms(
         converter.geodetic_to_local(chase_enemy_lat, chase_enemy_lon, chase_enemy_alt))
    chase_uav_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(chase_last_uav_lat, chase_last_uav_lon, chase_last_uav_alt))
    #追击后的时间
    after_chase_time = time + time_move + turning_time



    return last_begin_turn_dms, after_turning_last_uav_dms, chase_uav_dms, meet_last_enemy_dms, chase_enemy_dms, turning_time, chase_time_val, last_time_info



#计算第一批无人机与敌群相遇的时间
def first_meet_enemy_time(uav_sorted, max_enemy_dms, uav_speed, uav_deceleration_speed, enemy_speed, safe_distence, acceleration):
    meet_time_info = []
    #敌机中心经纬度转度数
    e_lat, e_lon, e_alt = GeodeticConverter.decimal_dms_to_degrees(max_enemy_dms)

    #对于已经排序的无人机
    for i,uav in enumerate(uav_sorted):

        #取排序无人机的第一个元素
        uav_pos = uav[3]
        #记录编号
        uav_number = uav [1]
        #print("uav_number",uav_number)

        #无人机直角坐标转经纬度再转度数
        lat, lon ,alt = GeodeticConverter.decimal_dms_to_degrees(uav_pos)

        #快到转弯点时要减速，求减速时间和距离
        deceleration_time = (uav_speed - uav_deceleration_speed) / acceleration
        uav_deceleration_distence = uav_speed * deceleration_time - 0.5 * acceleration * deceleration_time ** 2

        #利用球面坐标系计算无人机与敌机之间的初始距离
        distance_between_uav_enemy = converter.calculate_spherical_distance(lat, lon, alt, e_lat, e_lon, e_alt)

        #求匀速前行时的时间以及总时间
        uniform_time = (distance_between_uav_enemy - uav_deceleration_distence - enemy_speed * deceleration_time) / (enemy_speed + uav_speed)
        total_time = uniform_time + deceleration_time

        #无人机移动距离
        distance_uav = uav_speed * uniform_time + uav_deceleration_distence

        # 添加到meet_time列表中
        meet_time_info.append((uav_number, total_time, uniform_time, distance_uav))
    #print("meet_time_all:",meet_time_info)
    return meet_time_info


#第1波次无人机飞行策略
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

    #得到将敌方纳入视场内时我方和敌方分别移动距离
    distance_uav = uav_speed * time_detect
    distance_enemy = enemy_speed * time_detect

    #更新我方将敌方纳入视场时位置
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

    #更新敌方被纳入视场时位置
    start_enemy_lat, start_enemy_lon, start_enemy_alt = converter.calculate_destination_point(
                    enemy_lat, enemy_lon, enemy_alt, bearing_enemy, distance_enemy, 0)
    start_enemy = converter.local_to_geodetic_dms(
                    converter.geodetic_to_local(start_enemy_lat, start_enemy_lon, start_enemy_alt))

    #更新第1波或第2波次无人机中心位置
    start_center_lat, start_center_lon, start_center_alt = converter.calculate_destination_point(
                    center_lat, center_lon, center_alt, bearing, distance_uav, 0)
    start_center_uav = converter.local_to_geodetic_dms(
                    converter.geodetic_to_local(start_center_lat, start_center_lon, start_center_alt))

    #获得排序后第1波或第2波次无人机相遇时间（含减速到100的时间）
    meet_time_info = first_meet_enemy_time(start_first_uav_dms, start_enemy, uav_speed, uav_dec_speed, enemy_speed, safety_distance, acceleration)

    #从时间信息表中获取平均匀速时间
    time2_values = [entry[2] for entry in meet_time_info]
    avg_time2 = sum(time2_values) / len(time2_values)
    time_uniform = avg_time2 + time_detect

    first_uav_time_info.append(time_uniform) #获得匀速时间点
    first_uav_time_info.append(first_uav_time_info[0]) #没有加速，加速时间点=匀速时间点
    first_uav_time_info.append(first_uav_time_info[1]) #没有加速后匀速，加速后匀速时间点=匀速时间点

    #从信息表中获取平均匀速再减速总时间
    time1_values = [entry[1] for entry in meet_time_info]
    avg_time1 = sum(time1_values) / len(time1_values)
    time_del = avg_time1 - avg_time2 #减速了几秒的平均时间

    first_uav_time_info.append(time_del + first_uav_time_info[2])

    #飞行方向
    bearing_next = converter.calculate_flight_bearing(start_center_lat, start_center_lon, start_enemy_lat, start_enemy_lon)
    bearing_enemy_next = converter.calculate_flight_bearing(start_enemy_lat, start_enemy_lon, start_center_lat, start_center_lon)

    print("我方移动方向:",bearing_next)

    for i,uav in enumerate(start_first_uav_dms):

        #排序后无人机的dms坐标
        first_uav = uav[3]

        #dms转度数
        uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(first_uav)

        #求无人机转弯开始时的点位（输出）
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


        #由于每架无人机遇到敌机的时机可能都不一样，所以认为飞行航向是当前无人机的飞行航向，而不再以整体飞行航向作为标准
        bearing_rel = converter.calculate_flight_bearing(meet_uav_lat, meet_uav_lon, enemy_lat, enemy_lon)



        # 得到转弯angle_deg后的无人机dms位置
        after_turn_uav_dms = after_turn_position(turning_radius, meet_uav_dms, angle_deg, bearing_rel, direction='right')
        after_turn_first_uav_dms.append(after_turn_uav_dms)
        # print("转弯后的无人机坐标", after_turn_uav_dms)

    first_uav_time_info.append(turning_time + first_uav_time_info[3])  # 转弯完成时间

    return meet_first_uav_dms, after_turn_first_uav_dms, first_uav_time_info, bearing_enemy_next




#合成时间信息表（3个批次的）
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

# 进行占位计算

# —— 小工具：方位角 → ENU 单位向量（x=东, y=北）——
def bearing_to_unit_enu(bearing_deg):
    rad = math.radians(bearing_deg % 360.0)
    # ENU：x=East=sin(bearing), y=North=cos(bearing)
    return np.array([math.sin(rad), math.cos(rad)], dtype=float)

# —— 矩形线段求交：与“前沿直线” u·p = c 的交点 ——
def segment_line_intersections(p1, p2, u, c):
    # 线段参数：p(t)=p1+t*(p2-p1), t∈[0,1]
    v = p2 - p1
    denom = np.dot(u, v)
    if abs(denom) < 1e-12:
        # 平行：整段都在同一侧或重合
        if abs(np.dot(u, p1) - c) < 1e-9 and abs(np.dot(u, p2) - c) < 1e-9:
            # 共线：返回整个线段两端
            return [p1.copy(), p2.copy()]
        return []
    t = (c - np.dot(u, p1)) / denom
    if -1e-9 <= t <= 1 + 1e-9:
        return [p1 + t * v]
    return []

def line_rect_intersection(rect_xy, u, c):
    """
    rect_xy: 4个角(顺时针/逆时针) in ENU 2D, e.g. [SW, SE, NE, NW]
    u: forward 单位向量(2,)
    c: 标量，使得 u·p=c 的直线与矩形相交
    返回：与矩形相交得到的“前沿线段”的两个端点（2个点）
    """
    pts = []
    for i in range(4):
        p1 = rect_xy[i]
        p2 = rect_xy[(i + 1) % 4]
        pts += segment_line_intersections(p1, p2, u, c)
    # 去重并只取两个端点
    uniq = []
    for p in pts:
        if not any(np.linalg.norm(p - q) < 1e-6 for q in uniq):
            uniq.append(p)
    if len(uniq) > 2:
        # 共线情况会出现多点，取投影最小/最大两个
        projs = [np.dot(p, u) for p in uniq]
        idx_min = int(np.argmin(projs)); idx_max = int(np.argmax(projs))
        return [uniq[idx_min], uniq[idx_max]]
    return uniq  # 可能是0,1或2个点

def generate_placements_with_bearing(
    enemy_center_dms, enemy_latrange_dms, enemy_lonrange_dms,
    distance_m,  bearing_deg,
    start_side='left', max_uavs=None, max_rows=999
):

    """
    输入：
      - 敌群中心（DMS）
      - 纬度范围、经度范围（DMS，两个端点）
      - 敌群移动方向 bearing（度，正北为0，顺时针）
    输出：
      - front_point_llh: (lat, lon, alt)  单个“前沿点”
      - front_edge_llh: [(lat,lon,alt), (lat,lon,alt)]  “前沿线段”两个端点（若退化为点，两端相同）
    """
    # 1) 中心 & alt
    lat_c, lon_c, alt_c = GeodeticConverter.decimal_dms_to_degrees(enemy_center_dms)
    print('lat_c', alt_c)
    alt = float(alt_c)

    # 2) 四角（DMS→deg）
    lat1 = GeodeticConverter.dms_to_decimal(enemy_latrange_dms[0])
    lat2 = GeodeticConverter.dms_to_decimal(enemy_latrange_dms[1])
    lon1 = GeodeticConverter.dms_to_decimal(enemy_lonrange_dms[0])
    lon2 = GeodeticConverter.dms_to_decimal(enemy_lonrange_dms[1])

    lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
    lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)

    # 角点（顺时针）：WS, ES, EN, WN（西南，东南，东北，西北）
    corners_llh = [
        (lat_min, lon_min, alt),
        (lat_min, lon_max, alt),
        (lat_max, lon_max, alt),
        (lat_max, lon_min, alt),
    ]

    # 3) 建 ENU 转换器（以中心为基准）

    # 4) 角点转 ENU
    corners_xy = []
    for (la, lo, al) in corners_llh:
        x, y, z = converter.geodetic_to_local(la, lo, al)  # East=x, North=y
        corners_xy.append((x, y))
    # 补回首点，便于边遍历
    corners_xy.append(corners_xy[0])
    print("corners_xy", corners_xy)

    # 5) 前向单位向量 f（bearing: 北=0，顺时针）
    theta = math.radians(bearing_deg)
    fx, fy = math.sin(theta), math.cos(theta)  # ENU 中朝向
    f = (fx, fy)

    # 6) 计算各角点在 f 上的投影，取 Smax
    projs = [fx*px + fy*py for (px, py) in corners_xy[:-1]]
    Smax = max(projs)

    # 7) 求“支撑线” dot(f, x)=Smax 与四条边的交点
    eps = 1e-9
    inters = []
    for i in range(4):
        x0, y0 = corners_xy[i]
        x1, y1 = corners_xy[i+1]
        dx, dy = x1 - x0, y1 - y0
        denom = fx*dx + fy*dy  # dot(f, p1-p0)

        if abs(denom) < eps:
            # 与支撑线平行，可能整条边都在支撑线上（极罕见，矩形与方向正好对齐）
            # 判断端点是否在支撑线上
            if abs(fx*x0 + fy*y0 - Smax) < 1e-6 and abs(fx*x1 + fy*y1 - Smax) < 1e-6:
                # 整条边是前沿，记录两个端点
                inters.append((x0, y0))
                inters.append((x1, y1))
            # 否则没有交点
            continue

        t = (Smax - (fx*x0 + fy*y0)) / denom
        if -eps <= t <= 1+eps:
            # 裁剪到[0,1]
            t = max(0.0, min(1.0, t))
            xi, yi = x0 + t*dx, y0 + t*dy
            # 避免重复点
            if not inters or (abs(xi - inters[-1][0]) > 1e-6 or abs(yi - inters[-1][1]) > 1e-6):
                inters.append((xi, yi))

    # 8) 规范化交点数量
    if len(inters) == 0:
        # 理论上不会发生；回退到投影最大的顶点
        idx = projs.index(Smax)
        inters = [corners_xy[idx], corners_xy[idx]]
    elif len(inters) == 1:
        # 退化为前沿顶点
        inters = [inters[0], inters[0]]
    else:
        # 最多留下两端点（若因共线加入了4点，取端点投影在法向的最远两点）
        if len(inters) > 2:
            # 用与 f 垂直的方向区分端点
            nx, ny = -fy, fx
            inters.sort(key=lambda p: nx*p[0] + ny*p[1])
            inters = [inters[0], inters[-1]]

    # 9) 取“前沿点” = 把原点沿 f 推到 Smax 的点，并裁剪到线段
    # 原点在 ENU 是 (0,0)
    x_star, y_star = fx*Smax, fy*Smax

    # 将 x_star 在线段 inters[0]-inters[1] 上投影并裁剪
    (xA, yA), (xB, yB) = inters[0], inters[1]
    vx, vy = xB - xA, yB - yA
    seg_len2 = vx*vx + vy*vy
    if seg_len2 < 1e-12:
        # 线段退化为点
        xf, yf = xA, yA
    else:
        t = ((x_star - xA)*vx + (y_star - yA)*vy) / seg_len2
        t = max(0.0, min(1.0, t))
        xf, yf = xA + t*vx, yA + t*vy

    # 10) ENU → 经纬高
    front_point_llh = converter.local_to_geodetic((xf, yf, 0.0))
    p0_llh = converter.local_to_geodetic((xA, yA, 0.0))
    p1_llh = converter.local_to_geodetic((xB, yB, 0.0))

    # 替换 alt
    front_point_llh = (front_point_llh[0], front_point_llh[1], alt)
    p0_llh = (p0_llh[0], p0_llh[1], alt)
    p1_llh = (p1_llh[0], p1_llh[1], alt)

    front_point_dms1 = converter.local_to_geodetic_dms(front_point_llh)
    print("front_point_dms1", front_point_dms1)

    return front_point_dms1, [p0_llh, p1_llh]













# 第三批次先进行一段加速之后的位置求取，加速结束后再用first_second_timed_position计算位置
def third_uav_first_speed_up_timed_position(center_pos, enemy_center, uav_speed, uav_max_speed, acceleration):
    uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(center_pos)
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    speed_up_bearing = converter.calculate_flight_bearing(uav_lat, uav_lon, enemy_lat, enemy_lon)
    speed_up_time = (uav_max_speed - uav_speed) / acceleration
    speed_up_distance = uav_speed * speed_up_time + (1/2) * acceleration *speed_up_time ** 2
    uav_speed_over_lat, uav_speed_over_lon, uav_speed_over_alt = converter.calculate_destination_point(uav_lat, uav_lon, uav_alt, speed_up_bearing,
                                                                                                 speed_up_distance, 0)
    uav_speed_up_over_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(uav_speed_over_lat, uav_speed_over_lon, uav_speed_over_alt))
    return uav_speed_up_over_dms

# 第一、二波次从起始点的定时位置（关键时间：匀速结束（减速开始），减速结束（相遇），转弯结束（追击开始），追击结束）
def uav_timed_position(first_center_pos, uav_speed, uav_max_speed, uav_deceleration_speed, enemy_center,enemy_chase_center, acceleration,time,state_end_times):
    if state_end_times[0] != state_end_times[1]: #前两个状态时间不一样，说明有加速的过程
        flag = 1
    else:
        flag = 0
    if flag == 1:
        first_center_pos = third_uav_first_speed_up_timed_position(first_center_pos, enemy_center, uav_speed, uav_max_speed, acceleration)
    else:
        pass
    timed_position = []
    first_lat, first_lon, first_alt = GeodeticConverter.decimal_dms_to_degrees(first_center_pos)
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    enemy_chase_lat, enemy_chase_lon,enemy_chase_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_chase_center)
    bearing = converter.calculate_flight_bearing(first_lat, first_lon, enemy_lat, enemy_lon)
    uniform_distance = uav_speed * (state_end_times[2] - state_end_times[1])
    # 匀速结束的位置
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
    # 不同行动状态对应位置求取
    if time <= state_end_times[2]: #匀速阶段
        # 计算位移
        uniform_process_distance = uav_speed * time
        #匀速阶段time处的位置
        uav_chase_lat, uav_chase_lon, uav_chase_alt = converter.calculate_destination_point(first_lat, first_lon, first_alt, bearing, uniform_process_distance, 0)
       # timed_position.append((time, uav_chase_lat, uav_chase_lon, uav_chase_alt))
    # return uniform_timed_position
    elif state_end_times[2] < time <= state_end_times[3]:#减速阶段
        dec_time = time - state_end_times[2]
        dec_distance = uav_speed * dec_time - (1/2) * acceleration * dec_time ** 2

        #减速阶段time处的位置
        uav_chase_lat, uav_chase_lon, uav_chase_alt = converter.calculate_destination_point(uniform_over_lat, uniform_over_lon, uniform_over_alt,
                                                                                      bearing, dec_distance, 0)
       # timed_position.append((time, uav_chase_lat, uav_chase_lon, uav_chase_alt))
    # return dec_timed_position
    elif state_end_times[3] < time <= state_end_times[4]: #转弯阶段 不接收数据不传出数据(开始转弯到开始追击)
        #当前时间以及移动的弧度
        turning_time = (angle_deg_rad * turning_radius) / uav_deceleration_speed
        elapsed_turn_time = time - state_end_times[3]
        angle_turned_rad = (elapsed_turn_time / turning_time) * angle_deg_rad

        # 转弯阶段time处的位置
        uav_chase_lat, uav_chase_lon, uav_chase_alt = calculate_turning_position_with_bearing(
            dec_over_lat, dec_over_lon, dec_over_alt, turning_radius, angle_turned_rad, bearing_deg, direction='right')
        #timed_position.append((time, uav_chase_lat, uav_chase_lon, uav_chase_alt))
    # return turning_timed_position
    elif state_end_times[4] < time <= state_end_times[5]:  # 追击阶段 不接收数据不传出数据
        #追击加速阶段(暂时设定追击时间大于加速时间)
        chase_speed_up_time = (uav_max_speed-uav_deceleration_speed) / acceleration
        chase_bearing = converter.calculate_flight_bearing(turn_over_lat, turn_over_lon, enemy_chase_lat,
                                                           enemy_chase_lon)
        if time <= state_end_times[4] + chase_speed_up_time:
            chase_distance = uav_deceleration_speed * (time - state_end_times[4]) + (1/2) * acceleration * (time - state_end_times[4]) ** 2
            uav_chase_lat, uav_chase_lon, uav_chase_alt = converter.calculate_destination_point(turn_over_lat, turn_over_lon, turn_over_alt,
                                                                                      chase_bearing, chase_distance, 0)
           # timed_position.append((time, uav_chase_lat, uav_chase_lon, uav_chase_alt))
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
    # return turning_timed_position
    else:
        print(f"目前已完成追击任务")
    timed_position.append((time, uav_chase_lat, uav_chase_lon, uav_chase_alt))
    print(f"某时刻无人机位置", timed_position)
    return timed_position

# 敌群定时输出位置
def enemy_timed_position(enemy_center, enemy_speed, uav_center, time):
    enemy_timed_position = []
    uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(uav_center)
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    enemy_bearing = converter.calculate_flight_bearing(uav_lat, uav_lon, enemy_lat, enemy_lon)
    # time = range(0, uav2_state_end_times[3]+1) #总时间：起始——第二批次无人机追击上敌群
    enemy_action_distance = enemy_speed * time
    enemy_action_lat, enemy_action_lon, enemy_action_alt = converter.calculate_destination_point(enemy_lat, enemy_lon, enemy_alt, enemy_bearing, enemy_action_distance, 0)
    enemy_timed_position.append((time, enemy_action_lat, enemy_action_lon, enemy_action_alt))
    return enemy_timed_position

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

        if uav3_status == "转弯" and uav1_status != "转弯" and uav2_status != "转弯": #需要接收1，2,e位置
            uav1_position = uav_timed_position(first_center_pos, uav_speed, uav_max_speed, uav_deceleration_speed, enemy_center, enemy_chase_center, acceleration, time, uav1_state_end_times)
            uav2_position = uav_timed_position(second_center_pos, uav_speed, uav_max_speed, uav_deceleration_speed, enemy_center, enemy_chase_center, acceleration, time, uav2_state_end_times)
            enemy_position = enemy_timed_position(enemy_center, enemy_speed, basepoint, time)
            # print(f"正在转弯，定时接收我方1,2和敌群中心位置",uav1_position, uav2_position, enemy_position)
            print(f"At time {time} UAV3 is turning, UAV1's position is {uav1_position}, UAV2's position is {uav2_position}, enemy's position is {enemy_position}")
            information.append(uav1_position, uav2_position, enemy_position)
        elif uav1_status == "转弯" and uav2_status != "转弯": #需要接收2，e位置
            #担心第一批次正在转的时候，第二批次也开始转了，同时第三批次还没追上呢，那么要接收谁的位置信息呢？？？ 暂定全接收
            uav2_position = uav_timed_position(second_center_pos, uav_speed, uav_max_speed,uav_deceleration_speed, enemy_center,enemy_chase_center,acceleration,time,uav2_state_end_times)
            uav3_position = uav_timed_position(third_center_pos, uav_speed, uav_max_speed,uav_deceleration_speed, enemy_center,enemy_chase_center,acceleration,time,uav3_state_end_times)
            enemy_position = enemy_timed_position(enemy_center, enemy_speed, basepoint, time)
            print( f"At time {time} UAV1 is turning, UAV2's position is {uav2_position}, UAV3's position is {uav3_position}, enemy's position is {enemy_position}")
            information.append(uav2_position, enemy_position)
        elif uav2_status == "转弯": #需要接收3，e位置
            #这时候第三批追上了吗？第一批转完了吗？要接收谁的位置信息呢？暂定接收第三批次和敌群位置，但是这会儿敌群位置是谁给的呢？
            uav3_position = uav_timed_position(third_center_pos, uav_speed, uav_max_speed,uav_deceleration_speed, enemy_center,enemy_chase_center,acceleration,time,uav3_state_end_times)
            enemy_position = enemy_timed_position(enemy_center, enemy_speed, basepoint, time)
            information.append(uav3_position, enemy_position)
            print( f"At time {time} UAV2 is turning, UAV3's position is {uav3_position}, enemy's position is {enemy_position}")
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
def plot_point_group(dms_list, label, color='orange', mode='markers+lines'):
    lats, lons, alts = geo_to_degrees(dms_list)
    return go.Scatter3d(
        x=lons,
        y=lats,
        z=alts,
        mode=mode,
        marker=dict(size=4, color=color),
        name=label,
        text=[f"{label} {i}" for i in range(len(lats))],
        hovertemplate=
            f"<b>{label}</b><br>" +
            "Lon: %{x}<br>Lat: %{y}<br>Alt: %{z} m<br><extra></extra>"
    )


# 每架无人机三阶段航迹
def plot_uav_trajectories(init_geo, meet_geo, turn_geo, label_prefix="UAV", color="green"):
    traces = []
    for i in range(len(init_geo)):
        points = [init_geo[i], meet_geo[i], turn_geo[i]]
        lats, lons, alts = [], [], []
        for dms in points:
            lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(dms)
            lats.append(lat)
            lons.append(lon)
            alts.append(alt)

        trace = go.Scatter3d(
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
        traces.append(trace)
    return traces



def plot_positions(uav_first_geo_init, uav_second_geo_init,
                    enemy_center_init, meet_last_enemy_center, chase_enemy_center,
                    last_uav, meet_last_uav_point, after_turn_last_uav, chase_last_point,
                    meet_first_uav_point, after_turn_first_uav):

    # ==================================点位转坐标=============================================
    # 初始第1波点位 uav_dms, enemy_dms,
    first_uav_init_lats, first_uav_init_lons, first_uav_init_alts = geo_to_degrees(uav_first_geo_init)
    # 初始第2波点位
    second_uav_init_lats, second_uav_inti_lons, second_uav_init_alts = geo_to_degrees(uav_second_geo_init)

    # ===================================敌群中心转坐标============================================
    # 初始中心（已知）
    enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center_init)  # 开始敌群中心
    last_enemy_center_lat, last_enemy_center_lon, last_enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(meet_last_enemy_center)  # 遇到最后一架无人机开始转弯敌群中心
    last_chase_enemy_lat, last_chase_enemy_lon, last_chase_enemy_alt = GeodeticConverter.decimal_dms_to_degrees(chase_enemy_center)  # 被最后一架无人机追上时的位置

    # ==================================最后一架无人机============================================
    last_lat, last_lon, last_alt = GeodeticConverter.decimal_dms_to_degrees(last_uav)  # 开始
    meet_last_lat, meet_last_lon, meet_last_alt = GeodeticConverter.decimal_dms_to_degrees(meet_last_uav_point)  # 刚与敌群相遇时的点位
    after_turn_last_lat, after_turn_last_lon, after_turn_last_alt = GeodeticConverter.decimal_dms_to_degrees(after_turn_last_uav)  # 转弯之后的点位
    chase_last_lat, chase_last_lon, chase_last_alt = GeodeticConverter.decimal_dms_to_degrees(chase_last_point)  # 追击后的点位

    # ==================================第1波无人机============================================
    # meet_first_uav_lat, meet_first_uav_lon, meet_first_alt = GeodeticConverter.decimal_dms_to_degrees(meet_first_uav_point)
    # after_turn_first_lat, after_turn_last_lon, after_turn_last_alt = GeodeticConverter.decimal_dms_to_degrees(after_turn_first_uav)

    all_lons, all_lats, all_alts = [], [], []

    points = {
        'Last Enemy center ': [last_enemy_center_lat, last_enemy_center_lon, last_enemy_center_alt],
        'Last Chase Enemy': [last_chase_enemy_lat, last_chase_enemy_lon, last_chase_enemy_alt],
        'Meet Last UAV': [meet_last_lat, meet_last_lon, meet_last_alt],
        'After Turn Last UAV': [after_turn_last_lat, after_turn_last_lon, after_turn_last_alt],
        'Chase Last UAV': [chase_last_lat, chase_last_lon, chase_last_alt],
    }

    # 提取坐标
    xs = [pt[1] for pt in points.values()]  # 经度
    ys = [pt[0] for pt in points.values()]  # 纬度
    zs = [pt[2] for pt in points.values()]  # 高度
    labels = list(points.keys())
    colors = ['red', 'pink', 'purple', 'blue', 'cyan']
    sizes = [5, 5, 5, 5, 5]

    # 创建各个点
    scatter_points = []
    for i in range(len(xs)):
        scatter_points.append(go.Scatter3d(
            x=[xs[i]],
            y=[ys[i]],
            z=[zs[i]],
            mode='markers+text',
            marker=dict(size=sizes[i], color=colors[i]),
            name=labels[i],
            text=[labels[i]],
            textposition="top center",
            hovertemplate=
            f"<b>{labels[i]}</b><br>" +
            "Lon: %{x}<br>Lat: %{y}<br>Alt: %{z} m<br><extra></extra>"
        ))

    # ===== 多无人机轨迹（三阶段）=====
    uav_trajectories = plot_uav_trajectories(
        uav_first_geo_init, meet_first_uav_point, after_turn_first_uav,
        label_prefix="First UAV", color='green')

        # 示例轨迹线：你可以换成更复杂的路径
    path = go.Scatter3d(
        x=[points['After Turn Last UAV'][1], points['Chase Last UAV'][1]],
        y=[points['After Turn Last UAV'][0], points['Chase Last UAV'][0]],
        z=[points['After Turn Last UAV'][2], points['Chase Last UAV'][2]],
        mode='lines',
        line=dict(color='black', width=4),
        name='UAV Turn Path'
    )

    # 提取 First UAV 轨迹坐标
    for group in [uav_first_geo_init, meet_first_uav_point, after_turn_first_uav]:
        for dms in group:
            lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(dms)
            all_lats.append(lat)
            all_lons.append(lon)
            all_alts.append(alt)

    # 加入关键点坐标（原来的xs、ys、zs）
    all_lats.extend(ys)
    all_lons.extend(xs)
    all_alts.extend(zs)
    print("显示最大最小值:",min(all_lons), max(all_lons), min(all_lats), max(all_lats))



    # 绘制图形
    fig = go.Figure(uav_trajectories + scatter_points + [path])

    # 设置显示参数
    fig.update_layout(
        scene=dict(
            xaxis_title='Longitude (°E)',
            yaxis_title='Latitude (°N)',
            zaxis_title='Altitude (m)',

            xaxis=dict(range=[min(all_lons) - 0.1, max(all_lons) + 0.1]),
            yaxis=dict(range=[min(all_lats) - 0.1, max(all_lats) + 0.1]),
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
    #===================================相对位置===============================================
    # 第3波次无人机（加速、匀速、减速、转弯、追击上的结束时间） 现在少一个
    uav3_state_end_times = [3, 7, 9, 11]  # 第3波次无人机的加速、匀速、减速、转弯的结束时间
    # 第1/2波次无人机（匀速、减速、转弯、追击上的结束时间） 现在少一个
    uav1_state_end_times = [5, 10, 15]  # 第1波次无人机的匀速、减速、转弯的结束时间
    uav2_state_end_times = [6, 12, 17]  # 第二波次无人机的匀速、减速、转弯的结束时间
    enemy_chase_center = [0,0,0] #后面商量怎么改
    first_center_pos = [] #已知
    second_center_pos = [] #已知
    third_center_pos = [] #已知
    # 调用函数
    state_result = drone_state(uav1_state_end_times, uav2_state_end_times, uav3_state_end_times, data['minimum_speed'], data['maximum_speed'], data['speed'], data['basepoint'], data['enemy_approx'],
                enemy_chase_center, first_center_pos, second_center_pos, third_center_pos, chase_time_val=None)
    # 输出结果
    for time, uav1_status, uav2_status, uav3_status, information in state_result:
        print(f"Time {time}: UAV1 is {uav1_status}, UAV2 is {uav2_status}, UAV3 is {uav3_status},convey information is {information}")

    #=======================处理第2波次无人机纵队最后一架无人机===============================
    #对第2波次无人机进行聚类，得到纵队情况
    cluster_second_uav = cluster_uavs_by_latitude(data['second_uavs'], converter) #聚类
    num_columns = len(cluster_second_uav) #得到类别数

    #求第2波次无人机中心
    second_uav_center = calculate_center_dms(data['second_uavs'])

    #对第2波次无人机按y从小到大进行排序，得到sorted_second[0]就是末尾那架无人机
    second_uav_sorted, second_sorted_dms = uav_sorted_distances_points(data['second_uavs'], second_uav_center, data['enemy_approx'], reverse = True)
    # print("最后一架无人机:", second_sorted_dms[0])

    #已知敌机中心和经纬度范围求得敌机距离base的最大距离，即最远边界值
    max_distance, max_enemy_dms, min_distance, min_enemy_dms = max_distance(data['basepoint'], data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange'])
    # print("输出最大距离最小距离:",max_distance, max_dms, min_distance, min_dms)

    #计算当第1波次无人机能将敌方全纳入探测范围时的时间
    time_detect = function_last_detection_time(data['minimum_speed'], data['speed'], max_distance, data['detect_distance'])
    # print("time_detect = ", time_detect)

    last_point = second_sorted_dms[0]
    # print("*******", last_point)
   
    #计算最后一架无人机转弯起点位置、转弯180度后位置、追赶位置以及敌方中心在我方无人机开始转弯时位置、被追赶上位置；转弯时间，追逐时间

    last_begin_turn_dms, after_turning_last_uav_dms, chase_uav_dms, meet_last_enemy_dms, chase_enemy_dms, turning_time, chase_time, last_time_info = last_uav_move_strategy(

        data['minimum_speed'], data['maximum_speed'], uav_deceleration_speed, data['speed'],
        max_distance, data['detect_distance'],last_point, data['basepoint'], min_enemy_dms, acceleration)

    # print("输出时间信息:", last_time_info)

    # =================================处理第1波次无人机===========================================

    #求第1波次无人机中心
    first_uav_center = calculate_center_dms(data['first_uavs'])

    # 对第一批次无人机进行排序，距离敌群由近到远，并计算相遇时间（包含安全距离）
    first_uav_sorted, first_sorted_dms = uav_sorted_distances_points(data['first_uavs'], first_uav_center, max_enemy_dms, reverse = False)

    meet_first_uav_dms, after_turn_first_uav_dms, first_uav_time_info, bearing_enemy = first_uav_move_strategy(data['minimum_speed'], uav_deceleration_speed, data['maximum_speed'], data['speed'], first_uav_sorted, first_uav_center,
                             max_enemy_dms, 1000, acceleration, time_detect)


    # =================================处理第2波次无人机===========================================

    remain_second_uav_sorted = second_uav_sorted[1:]
    # print("全部的以及剩余的:", second_uav_sorted)
    # print("剩余的:", remain_second_uav_sorted)

    meet_second_uav_dms, after_turn_second_uav_dms, second_uav_time_info, bearing_enemy2= first_uav_move_strategy(data['minimum_speed'], uav_deceleration_speed, data['maximum_speed'], data['speed'], remain_second_uav_sorted, second_uav_center,
                             max_enemy_dms, 1000, acceleration, time_detect)


    print("敌群飞行方向:", bearing_enemy)

    placements = generate_placements_with_bearing(
        data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange'],
        distance_m=400,
        bearing_deg=270.0,  # 敌方从东向西
        start_side='left',  # 先左（相对 forward 的左侧=南/北取决于bearing）
        max_uavs=30, max_rows=10
    )
    print("试试就逝世:", placements)

    for p in placements[:8]:
        print("先试试",p)




    time_uav_info = generate_uav_time_info(last_time_info, first_uav_time_info, second_uav_time_info)
    print("总时间信息表！！！！！:", time_uav_info)






    plot_positions(first_sorted_dms, data['second_uavs'],
                   data['enemy_approx'], meet_last_enemy_dms, chase_enemy_dms,
                   last_point, last_begin_turn_dms, after_turning_last_uav_dms, chase_uav_dms,
                   meet_first_uav_dms, after_turn_first_uav_dms)











    #数据输出
    outdata.save_uav_multi_positions(data['minimum_speed'], acceleration, data['maximum_speed'], 100, data['speed'],
                                     data['first_uavs'], data['second_uavs'], data['enemy_approx'],
                                     last_begin_turn_dms, after_turning_last_uav_dms, chase_uav_dms,
                                     time_detect, turning_time, chase_time,
                                     None, None, None,
                                     None, None, None,
                                     None, None, None,
                                     None, None, None)






