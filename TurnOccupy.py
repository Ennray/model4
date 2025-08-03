import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
from sympy.physics.units import acceleration


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

#先对UAV2按照y轴，即距离敌方的远近进行排序
def sorted_y_points(second_points, converter):
    seconds = []
    i = 0
    for point in second_points:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(point)
        point_local = converter.geodetic_to_local(lat, lon, alt)
        seconds.append([point_local, i, 0])
        i += 1
    seconds_sorted = sorted(seconds, key=lambda item: item[0][1], reverse=False)#按y轴升序排序
    return seconds_sorted


#对第1批或第2批uav进行排序，由敌群的近到远
def first_sorted_y_points(first_points,converter):
    first = []
    i = 0
    for point in first_points:
        lat, lon ,alt = GeodeticConverter.decimal_dms_to_degrees(point)
        point_local = converter.geodetic_to_local(lat,lon,alt)
        first.append([point_local, i, 0])
        i = i + 1
    first_sorted = sorted(first, key=lambda item: item[0][1], reverse=True)#按y轴降序排序
    return first_sorted


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
    #定义敌方四个角dms格式
    ne = [lon_range[1], lat_range[1], enemy_center[2]]  # 北 + 东
    se = [lon_range[1], lat_range[0], enemy_center[2]]  # 南 + 东
    nw = [lon_range[0], lat_range[1], enemy_center[2]]  # 北 + 西
    sw = [lon_range[0], lat_range[0], enemy_center[2]]  # 南 + 西

    #转为十进制度
    base_lat, base_lon, base_alt = GeodeticConverter.decimal_dms_to_degrees(basepoint)
    corners = [ne, se, nw, sw]

    #四个角转经纬度数值，用球面坐标计算到base的最远距离
    max_dist = 0
    for corner_dms in corners:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(corner_dms)
        dist = converter.calculate_spherical_distance(base_lat, base_lon, base_alt, lat, lon, alt)
        max_dist = max(max_dist, dist)

    return max_dist


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



def angle_with_latitude_line(bearing):
    """
    返回无人机方向与纬线（东西方向）之间的夹角（单位：度）
    结果范围在 [0, 180]
    """

    diff = abs(bearing - 90)  # 纬线是东西方向，参考角为90°
    if diff > 180:
        diff = 360 - diff
    return diff





def calculate_turning_position_with_bearing(lat, lon, alt, turning_radius, angle_deg, bearing_deg, direction='left'):

    angle_diff = angle_with_latitude_line(bearing_deg)
    print("与纬度的夹角：", {angle_diff})
    # Step 1: 计算圆心方向（偏移航向±90度）
    offset_bearing = (bearing_deg + (90 if direction == 'left' else -90)) % 360

    # Step 2: 计算圆心点
    center_point = distance(meters=turning_radius).destination(Point(lat, lon), bearing=offset_bearing)

    # Step 3: 计算终点方向：从圆心开始，按方向旋转角度
    arc_end_bearing = (offset_bearing + (angle_deg if direction == 'left' else -angle_deg)) % 360

    # Step 4: 沿圆弧从圆心出发，回到圆周上
    end_point = distance(meters=turning_radius).destination(center_point, bearing=arc_end_bearing)

    print("无人机方向，偏移", {bearing_deg}, offset_bearing)

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

    print(
        f"转弯后无人机的位置: 纬度 = {after_turning_uav_lat}, 经度 = {after_turning_uav_lon}, 高度 = {after_turning_uav_alt}")
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
def last_uav_move_strategy(uav_speed, uav_max_speed, uav_deceleration_speed, enemy_speed, max_distances, detect_distances, uavpoint, basepoint, enemy_center, acceleration):

    #飞行时间估计（第1波次无人机将敌方全纳入视场时间）
    time = function_last_detection_time(uav_speed, enemy_speed, max_distances, detect_distances)

    #匀加速时间与加速距离
    time_acc = (uav_max_speed - uav_speed) / acceleration  # 得到加速时间
    distance_acc = (uav_max_speed ** 2 - uav_speed ** 2) / (2 * acceleration)  # 得到加速期间前进的距离

    #加速前已经飞行距离
    total_distances = uav_speed * time + distance_acc

    #当前无人机位置
    lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(uavpoint)

    #敌群中心位置
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)

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

    #匀速减速再相遇所需要的总时间 = （不减速时的距离 - 安全距离 + 不减速时的距离与考虑减速时的距离之差） / 相对时间
    time_move = (distance_move - safety_distance + uav_max_speed * deceleration_time - uav_deceleration_distence) / (uav_max_speed + enemy_speed)

    #无人机最大速度匀速前进距离
    distance_uav = uav_max_speed * (time_move - deceleration_time)

    #敌方移动总时间
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

    #转弯时我方无人机及敌方无人机转dms
    last_begin_turn_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(uav_meet_lat, uav_meet_lon, uav_meet_alt))
    meet_last_enemy_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(meet_enemy_lat, meet_enemy_lon, meet_enemy_alt))


    bearing_rel =  converter.calculate_flight_bearing(uav_meet_lat, uav_meet_lon, enemy_lat, enemy_lon)
    print("输出两个坐标",uav_meet_lon,enemy_lon)
    print("开始转弯时无人机的位置：", last_begin_turn_dms)
    print("敌群最开始中心位置:",enemy_center)
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


    return last_begin_turn_dms, after_turning_last_uav_dms, chase_uav_dms, meet_last_enemy_dms, chase_enemy_dms, turning_time, chase_time_val


#计算第一批无人机与敌群相遇的时间
def first_meet_enemy_time(uav_sorted, enemy_pos, uav_speed, uav_deceleration_speed, enemy_speed, safe_distence, acceleration):
    meet_time_info = []
    #敌机中心经纬度转度数
    e_lat, e_lon ,e_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_pos)

    #对于已经排序的无人机
    for i,uav in enumerate(uav_sorted):

        #取排序无人机的第一个元素
        uav_pos = uav[0].tolist()
        #记录编号
        uav_number = uav [1]
        print("uav_number",uav_number)

        #无人机直角坐标转经纬度再转度数
        dms_uav_pos = converter.local_to_geodetic_dms(uav_pos)
        lat, lon ,alt = GeodeticConverter.decimal_dms_to_degrees(dms_uav_pos)

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
    print("meet_time_all:",meet_time_info)
    return meet_time_info

#第1波次无人机飞行策略
def first_uav_move_strategy(uav_speed, uav_dec_speed, uav_max_speed, enemy_speed, first_sorted_uav, first_uav_center, enemy_center, safety_distance, acceleration):
    #获得排序后第1波次无人机相遇时间（含减速到100的时间）
    meet_time_info = first_meet_enemy_time(first_sorted_uav, enemy_center, uav_speed, uav_dec_speed, enemy_speed, safety_distance, acceleration)

    #敌群中心位置
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)

    #第1波次无人机中心位置
    first_uav_center_lat, first_uav_center_lon, first_uav_center_alt = GeodeticConverter.decimal_dms_to_degrees(first_uav_center)

    #飞行方向
    bearing = converter.calculate_flight_bearing(first_uav_center_lat, first_uav_center_lon, enemy_lat, enemy_lon)
    bearing_enemy = converter.calculate_flight_bearing(enemy_lat, enemy_lon, first_uav_center_lat, first_uav_center_lon)

    for i,uav in enumerate(first_sorted_uav):

        #排序取无人机的第一个元素
        first_uav = converter.local_to_geodetic_dms(uav[0].tolist())
        uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(first_uav)

        #求无人机转弯时的点位
        meet_uav_lat, meet_uav_lon, meet_uav_alt = converter.calculate_destination_point(
            uav_lat, uav_lon, uav_alt, bearing, meet_time_info[i][3], 0)


        print("meet_time", meet_time_info[i][3])




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

def plot_positions(uav_first_geo_init, uav_second_geo_init,
                                enemy_center_init, last_uav,
                                meet_last_enemy_center, meet_last_uav_point,
                                after_turn_last_uav, chase_last_point, chase_enemy_center):
    # 示例点：可替换为你自己的经纬度+海拔数据（单位：° + m）
    # ==================================点位转坐标=============================================
    # 初始第1波点位 uav_dms, enemy_dms,
    first_uav_init_lats, first_uav_init_lons, first_uav_init_alts = geo_to_degrees(uav_first_geo_init)
    # 初始第2波点位
    second_uav_init_lats, second_uav_inti_lons, second_uav_init_alts = geo_to_degrees(uav_second_geo_init)

    # ===================================敌群中心转坐标============================================
    # 初始中心（已知）
    enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(
        enemy_center_init)  # 开始敌群中心
    last_enemy_center_lat, last_enemy_center_lon, last_enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(
        meet_last_enemy_center)  # 遇到最后一架无人机开始转弯敌群中心
    last_chase_enemy_lat, last_chase_enemy_lon, last_chase_enemy_alt = GeodeticConverter.decimal_dms_to_degrees(
        chase_enemy_center)  # 被最后一架无人机追上时的位置

    # ==================================最后一架无人机============================================
    last_lat, last_lon, last_alt = GeodeticConverter.decimal_dms_to_degrees(last_uav)  # 开始
    meet_last_lat, meet_last_lon, meet_last_alt = GeodeticConverter.decimal_dms_to_degrees(
        meet_last_uav_point)  # 刚与敌群相遇时的点位
    after_turn_last_lat, after_turn_last_lon, after_turn_last_alt = GeodeticConverter.decimal_dms_to_degrees(
        after_turn_last_uav)  # 转弯之后的点位
    chase_last_lat, chase_last_lon, chase_last_alt = GeodeticConverter.decimal_dms_to_degrees(
        chase_last_point)  # 追击后的点位

    points = {
        'Last Enemy center ': [last_enemy_center_lat, last_enemy_center_lon, last_enemy_center_alt],
        'Last Chase Enemy': [last_chase_enemy_lat, last_chase_enemy_lon, last_chase_enemy_alt],
        'Meet Last UAV': [meet_last_lat, meet_last_lon, meet_last_alt],
        'After Turn Last UAV': [after_turn_last_lat, after_turn_last_lon, after_turn_last_alt],
        'Chase Last UAV': [chase_last_lat, chase_last_lon, chase_last_alt],
    }

    # 提取坐标
    xs = [pt[0] for pt in points.values()]  # 经度
    ys = [pt[1] for pt in points.values()]  # 纬度
    zs = [pt[2] for pt in points.values()]  # 高度
    labels = list(points.keys())
    colors = ['red', 'pink', 'purple', 'blue', 'cyan']
    sizes = [10, 10, 10, 10, 10]

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

    # 示例轨迹线：你可以换成更复杂的路径
    path = go.Scatter3d(
        x=[points['Meet Last UAV'][0], points['Meet Last UAV'][0]],
        y=[points['Meet Last UAV'][1], points['Meet Last UAV'][1]],
        z=[points['Meet Last UAV'][2], points['Meet Last UAV'][2]],
        mode='lines',
        line=dict(color='black', width=4),
        name='UAV Turn Path'
    )

    # 绘制图形
    fig = go.Figure(data=scatter_points + [path])

    # 设置显示参数
    fig.update_layout(
        scene=dict(
            xaxis_title='Longitude (°E)',
            yaxis_title='Latitude (°N)',
            zaxis_title='Altitude (m)',
            xaxis=dict(range=[min(xs) - 0.1, max(xs) + 0.1]),
            yaxis=dict(range=[min(ys) - 0.1, max(ys) + 0.1]),
            zaxis=dict(range=[min(zs) - 1000, max(zs) + 1000]),
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

    #对第2波次无人机按y从小到大进行排序，得到sorted_second[0]就是末尾那架无人机
    sorted_second = sorted_y_points(data['second_uavs'], converter)

    #已知敌机中心和经纬度范围求得敌机距离base的最大距离，即最远边界值
    max_distance = max_distance(data['basepoint'], data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange'])
    # print(f"max_distance, distances2 = {max_distance} ")

    #计算当第1波次无人机能将敌方全纳入探测范围时的时间
    time_detect = function_last_detection_time(data['minimum_speed'], data['speed'], max_distance, data['detect_distance'])
    # print("time_detect = ", time_detect)

    last_point = converter.local_to_geodetic_dms(sorted_second[0][0])
    print("*******", last_point)

    #计算最后一架无人机转弯起点位置、转弯180度后位置、追赶位置以及敌方中心在我方无人机开始转弯时位置、被追赶上位置；转弯时间，追逐时间
    last_begin_turn_dms, after_turning_last_uav_dms, chase_uav_dms, meet_last_enemy_dms, chase_enemy_dms, turning_time, chase_time = last_uav_move_strategy(
        data['minimum_speed'], data['maximum_speed'], uav_deceleration_speed, data['speed'],
        max_distance, data['detect_distance'],last_point, data['basepoint'], data['enemy_approx'], acceleration)


    # ===========================处理第1波次无人机===============================
    # 对第一批次无人机进行排序，距离敌群由近到远，并计算相遇时间（包含安全距离）
    first_uav_sorted = first_sorted_y_points(data['first_uavs'], converter)
    first_uav_move_strategy(data['minimum_speed'], uav_deceleration_speed, data['maximum_speed'], data['speed'], first_uav_sorted,
                            data['enemy_approx'], 1000, acceleration)



    plot_positions(data['first_uavs'], data['second_uavs'], data['enemy_approx'], last_point,
                                meet_last_enemy_dms, last_begin_turn_dms,
                                after_turning_last_uav_dms, chase_uav_dms, chase_enemy_dms)




