import geopy
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
import subprocess, sys
import sympy as sp
from numpy.testing.print_coercion_tables import print_new_cast_table
from sympy.physics.units import acceleration
from scipy.optimize import linear_sum_assignment
from geopy.distance import geodesic

from geopy.distance import distance
from geopy import Point
import plotly.graph_objs as go
import plotly.graph_objects as go


from numpy.ma.core import remainder

import GeodeticConverter
import outdata
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


# 通过纵队情况判断应该提前起飞的无人机数
def column_judge_last_uav(column, num_of_column, uav_num):
    last_uav_num = 0
    judge = uav_num % num_of_column
    if judge == 0:
        last_uav_num = column
    else:
        last_uav_num = column - 1
    return last_uav_num


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


#对追击时间信息表进行整理
def chase_time_disposal(new_chase_info):

    t_total, t_acc, t_cruise, t_dec, intercept_dms = (
        new_chase_info.get(k) for k in ("t_total", "t_acc", "t_cruise", "t_dec", "intercept_dms"))
    chase_time_info = [t_total, t_acc, t_cruise, t_dec, intercept_dms]

    return chase_time_info


# 纵队最后一架无人机的追击策略，调用追击函数，整理追击时间
def last_uav_chase_strategy (after_turning_last_uav_dms, after_turn_enemy_dms, bearing_enemy, enemy_speed, detection, after_turn_time, enemy_center, uav_center, num, uav_max_speed):

    target_lat, target_lon, target_alt = GeodeticConverter.decimal_dms_to_degrees(after_turn_enemy_dms)

    target_start_dms = enemy_timed_position(after_turn_enemy_dms, enemy_center, enemy_speed, uav_center, after_turn_time)
    # print("在转弯后敌群的位置：",target_start_dms)
    # print("在转弯后无人机的位置:", after_turning_last_uav_dms)

    # 得到转弯后追击敌方时所需要花费的时间距离等
    new_chase_info = pursue_moving_point(after_turning_last_uav_dms, target_start_dms, bearing_enemy, enemy_speed,

                                         100, uav_max_speed, 80, 80, 30, 1, 0.05, 0.6, 20, 100000, num)

    # print("得到转弯后追击敌方时所需要花费的时间距离等：", new_chase_info["traj_dms"])


    # 追击时间信息，包括总时间，加速时间，匀速时间，减速时间,追击到的点位
    chase_time_info_all = chase_time_disposal(new_chase_info)


    # 追击时，我方和敌方移动距离
    d_uav, d_enemy = (new_chase_info.get(k) for k in ("S_uav", "S_enemy"))
    chase_distancce_info = [d_uav, d_enemy]


    return chase_time_info_all, chase_distancce_info, chase_time_info_all[4]



# 纵队最后一架无人机的飞行策略
def last_uav_move_strategy(uav_speed, uav_max_speed, uav_deceleration_speed, enemy_speed, max_distances, detect_distances, uavpoint, min_enemy_dms, acceleration, enemy_center):

    last_time_info = []
    last_state_dms_info = []
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
    last_uniform_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(lat, lon, alt)) #最后一架无人机加后dms
    last_state_dms_info.append(None)
    last_state_dms_info.append(last_uniform_dms)


    # 敌群最前方位置
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(min_enemy_dms)
    enemy_center_lat, enemy_center_lon, enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)

    # 求飞行方向
    bearing =  converter.calculate_flight_bearing(lat, lon, enemy_lat, enemy_lon)
    # print(f"飞行方向 = {bearing:.6f}")
    bearing_enemy = converter.calculate_flight_bearing(enemy_center_lat, enemy_center_lon, lat, lon)

    # 球面预测加速后飞行点
    new_lat, new_lon, new_alt = converter.calculate_destination_point(
        lat, lon, alt, bearing, total_distances, 0)

    # 敌群位置更新（反方向飞行）
    enemy_movedis = enemy_speed * (time + time_acc)
    new_enemy_lat, new_enemy_lon, new_enemy_alt = converter.calculate_destination_point(
        enemy_lat, enemy_lon, enemy_alt, bearing_enemy, enemy_movedis, 0)
    new_center_lat, new_center_lon, new_center_alt = converter.calculate_destination_point(
        enemy_center_lat, enemy_center_lon, enemy_center_alt, bearing_enemy, enemy_movedis, 0
    ) #敌群中心位置更新

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
        new_center_lat, new_center_lon, new_center_alt, bearing_enemy, distance_enemy, 0
    )

    # 在减速到达敌方前就爬升至敌方高度上方
    angle = converter.calculate_climb_angle(new_alt, new_enemy_alt + safety_distance, distance_uav)
    angle_rad = math.radians(angle)

    #如果本来敌群就低，那就不爬升了
    if new_enemy_alt < new_alt:
        angle = 0

    # 开始转弯的点位（相遇位置）
    acc_uni_lat,acc_uni_lon,acc_uni_alt, distance_uav_val = converter.calculate_destination_with_climb_angle(new_lat, new_lon, new_alt, bearing, distance_uav, angle)#加速再匀速后
    last_acc_uni_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(acc_uni_lat, acc_uni_lon, acc_uni_alt))  # 最后一架无人机加后dms
    last_state_dms_info.append(last_acc_uni_dms)

    uav_meet_lat, uav_meet_lon, uav_meet_alt, distance_uav_val = converter.calculate_destination_with_climb_angle(acc_uni_lat,acc_uni_lon,acc_uni_alt, bearing, uav_deceleration_distence, 0)
    last_dec_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(uav_meet_lat, uav_meet_lon, uav_meet_alt))  # 最后一架无人机减速后dms
    last_state_dms_info.append(last_dec_dms)

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
    last_begin_turn_dms = last_dec_dms
    meet_last_enemy_dms = converter.local_to_geodetic_dms(
        converter.geodetic_to_local(meet_enemy_lat, meet_enemy_lon, meet_enemy_alt))


    bearing_rel =  converter.calculate_flight_bearing(uav_meet_lat, uav_meet_lon, enemy_lat, enemy_lon)
    # print("最开始敌群最近位置:",min_enemy_dms)

    # 得到转弯angle_deg后的无人机dms位置
    after_turning_last_uav_dms = after_turn_position(turning_radius, last_begin_turn_dms, angle_deg, bearing_rel, direction = 'right')
    last_state_dms_info.append(after_turning_last_uav_dms) #最后一架无人机转弯后dms

    #转弯后敌群中心此时的位置
    after_turn_enemy_lat, after_turn_enemy_lon, after_turn_enemy_alt = converter.calculate_destination_point(
        meet_enemy_lat, meet_enemy_lon, meet_enemy_alt, bearing_enemy, enemy_speed * turning_time, 0
    )
    after_turn_enemy_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(after_turn_enemy_lat, after_turn_enemy_lon, after_turn_enemy_alt))



    return last_state_dms_info, last_begin_turn_dms, after_turning_last_uav_dms, bearing_enemy, meet_last_enemy_dms, after_turn_enemy_dms, last_time_info


# 计算第一批无人机与敌群相遇的时间
def first_meet_enemy_time(uav_sorted, max_enemy_dms, uav_speed, uav_deceleration_speed, enemy_speed, acceleration):
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
    firstsecond_state_dms_info = []

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
        # 排序后无人机的dms坐标由近到远
        first_uav = uav[3]
        # dms转度数
        uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(first_uav)


        new_uav_lat, new_uav_lon, new_uav_alt = converter.calculate_destination_point(
                    uav_lat, uav_lon, uav_alt, bearing, distance_uav, 0)

        start_first_uav = converter.local_to_geodetic_dms(
                    converter.geodetic_to_local(new_uav_lat, new_uav_lon, new_uav_alt))

        start_first_uav_dms.append((uav[0] - distance_uav - distance_enemy, uav[1], uav[2], start_first_uav))

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
    meet_time_info = first_meet_enemy_time(start_first_uav_dms, start_enemy, uav_speed, uav_dec_speed, enemy_speed, acceleration)
    # print("排序后第1波或第2波次无人机相遇时间:", meet_time_info)

    #从时间信息表中获取平均匀速时间
    time2_values = [entry[2] for entry in meet_time_info]
    avg_time2 = sum(time2_values) / len(time2_values)
    time_uniform = avg_time2 + time_detect

    first_uav_time_info.append(time_uniform) #获得匀速时间点
    first_uav_time_info.append(first_uav_time_info[0]) #没有加速，加速时间点=匀速时间点
    first_uav_time_info.append(first_uav_time_info[1]) #没有加速后匀速，加速后匀速时间点=匀速时间点

    #第1/2波无人机中心匀速后得到的dms
    unifrom_lat, unifrom_lon, unifrom_alt = converter.calculate_destination_point(start_center_lat, start_center_lon, start_center_alt, bearing, time_uniform * uav_speed , 0)
    firstsecond_uni_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(unifrom_lat, unifrom_lon, unifrom_alt))
    firstsecond_state_dms_info.append(None)
    firstsecond_state_dms_info.append(None)
    firstsecond_state_dms_info.append(firstsecond_uni_dms) #第1/2波匀速后dms

    # 从信息表中获取平均匀速再减速总时间
    time1_values = [entry[1] for entry in meet_time_info]
    avg_time1 = sum(time1_values) / len(time1_values)
    time_del = avg_time1 - avg_time2 #减速了几秒的平均时间
    dec_distance = (uav_max_speed ** 2 - uav_dec_speed ** 2) / 2 * acceleration

    #第1/2波无人机中心减速后得到的dms
    dec_lat, dec_lon, dec_alt = converter.calculate_destination_point(unifrom_lat, unifrom_lon, unifrom_alt, bearing, dec_distance, 0)
    firstsecond_dec_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(dec_lat, dec_lon, dec_alt))
    firstsecond_state_dms_info.append(firstsecond_dec_dms) #第1/2波减速后dms

    first_uav_time_info.append(time_del + first_uav_time_info[2])

    # 飞行方向
    bearing_next = converter.calculate_flight_bearing(start_center_lat, start_center_lon, start_enemy_lat, start_enemy_lon)
    bearing_enemy_next = converter.calculate_flight_bearing(start_enemy_lat, start_enemy_lon, start_center_lat, start_center_lon)

    # print("我方移动方向:",bearing_next)

    for i,uav in enumerate(start_first_uav_dms):

        # 排序后无人机的dms坐标
        first_uav = uav[3]

        # dms转度数
        uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(first_uav)

        # 求无人机转弯开始时的点位（输出）
        meet_uav_lat, meet_uav_lon, meet_uav_alt = converter.calculate_destination_point(
            uav_lat, uav_lon, uav_alt, bearing_next, meet_time_info[i][3], 0)
        meet_uav_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(meet_uav_lat, meet_uav_lon, meet_uav_alt))
        meet_first_uav_dms.append([meet_uav_dms, uav[1]])
        # print("转弯前的无人机坐标：", meet_uav_dms)

        # 当前假设转弯180度
        angle_deg = 180
        angle_deg_rad = math.radians(angle_deg)

        # 计算转弯半径（目前转弯速度和转弯半径均相同）(输出)
        turning_radius = (uav_dec_speed ** 2) / (GRAVITY_EARTH * math.tan(math.radians(45)))  # (uav_deceleration_speed**2) * (math.cos(angle_rad)**2))
        turning_time = (angle_deg_rad * turning_radius) / uav_dec_speed


        # 由于每架无人机遇到敌机的时机可能都不一样，所以认为飞行航向是当前无人机的飞行航向，而不再以整体飞行航向作为标准
        bearing_rel = converter.calculate_flight_bearing(meet_uav_lat, meet_uav_lon, enemy_lat, enemy_lon)

        # 得到转弯angle_deg后的无人机dms位置
        after_turn_uav_dms = after_turn_position(turning_radius, meet_uav_dms, angle_deg, bearing_rel, direction='right')
        after_turn_first_uav_dms.append(after_turn_uav_dms)
        # print("转弯后的无人机坐标", after_turn_first_uav_dms)

    first_uav_time_info.append(turning_time + first_uav_time_info[3])  # 转弯完成时间

    firstsecond_turn_dms = after_turn_position(turning_radius, firstsecond_dec_dms, angle_deg, bearing, direction='right')
    firstsecond_state_dms_info.append(firstsecond_turn_dms)  # 第1/2波转弯后dms
    firstsecond_state_dms_info.append(None) # 保证与第三波次一致数目一致
    # 每一架无人机从进入视场开始直到转弯结束的时间，仅时间列表，time_detect是开始到进入视场，turning_time是转弯时间
    meet_single_time = [(time_detect + entry[1] + turning_time) for entry in meet_time_info]
    result = [(meet_time_info[i][0], meet_single_time[i], after_turn_first_uav_dms[i] ) for i in range(len(meet_single_time))]
    sorted_result = sorted(result, key=lambda x: x[1])

    # 在相遇时间中增加探测时间
    meet_time = [(item[1] + time_detect ) for item in meet_time_info]
    time_item = [item[0] for item in meet_time_info]
    continue_meet_time_info = list(zip(time_item, meet_time))


    # print("每架无人机从开始到转弯结束的编号、时间、位置", result)
    # print("转弯后的位置：", after_turn_first_uav_dms)


    return  firstsecond_state_dms_info, meet_first_uav_dms, after_turn_first_uav_dms, first_uav_time_info, bearing_enemy_next, meet_single_time, continue_meet_time_info, sorted_result, turning_time


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
        accel_act, cruise_act, decel_act, turn_act, chase_act, after_chase = state_end_times
        if time <= accel_act:
            return "匀速前进"
        elif accel_act < time <= cruise_act:
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
        if time <= accel_act:
            return "匀速前进"
        elif accel_act < time <= cruise_act and accel_act != cruise_act:
            return "加速前进"
        elif accel_act < time <= cruise_act and accel_act == cruise_act:
            return "匀速前进"
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


# 计算坐标点投影值
def dot_projection(p, vx, vy):
    return p[0] * vx + p[1] * vy


# 计算敌群前进方向的前沿边和后沿边，并进行无人机占位
def generate_placements_with_bearing(
    enemy_center_dms, enemy_latrange_dms, enemy_lonrange_dms,
    distance_m,  bearing_deg, exclusion_radius_m, preplaced_dms,
    start_side = 'left', max_rows = 999, first_uav_num = 45, second_uav_num = 30
):
    print("所有信息都输出一遍我先看看:",enemy_center_dms,"weidu",enemy_lonrange_dms,"jingdu",enemy_latrange_dms,"juli",distance_m,
          "fangxaing",bearing_deg,"banjin",exclusion_radius_m,"yijinzhanwei",preplaced_dms)
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
    print("前后沿找到了吗：",inters_front)

    # 前沿坐标点转dms
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
    for i,uav in enumerate(preplaced_dms):
        pre_lat, pre_lon, pre_alt = GeodeticConverter.decimal_dms_to_degrees(uav)
        local = converter_enu.geodetic_to_local(pre_lat, pre_lon, pre_alt)
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
    # print("最小值:", Smin)

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
    converter_point = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)

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
def chase_strategy(occupy_dms, uav_after_turn_dms, first_uav_num, second_uav_num, enemy_bearing_deg, enemy_speed,
                   uav_start_speed=100.0,  # 我方起始空速 m/s
                   uav_max_speed=500.0,  # 我方最大空速 m/s
                   a_acc=80.0,  # 加速度
                   a_dec=80.0,  # 减速度
                   pos_tol=3.0,  # 空间位置收敛阈值m
                   vel_tol=1.0,  # 速度收敛阈值m/s（接近敌速）
                   dt=1,  # 仿真步长s
                   k_lead=0.6,  # 前视系数（0.4~0.8 常用）
                   distance_margin=20.0,  # 刹车安全距离 m
                   max_steps=200000):
    uav_tag = "None"
    if len(uav_after_turn_dms) == first_uav_num:
        uav_tag = "first"
    elif len(uav_after_turn_dms) == second_uav_num:
        uav_tag = "second"

    chase_info = []
    print("占位点数量:",occupy_dms)
    print("追击数量:",uav_after_turn_dms)

    for i,uav in enumerate(uav_after_turn_dms):

        if uav_tag == "first":

            info = pursue_moving_point(uav, occupy_dms[i], enemy_bearing_deg, enemy_speed, uav_start_speed, uav_max_speed, a_acc, a_dec,
                                             pos_tol, vel_tol, dt, k_lead, distance_margin, max_steps, -1)
            info = chase_time_disposal(info)
            chase_info.append(info)

        if uav_tag == "second":
            info = pursue_moving_point(uav, occupy_dms[i], enemy_bearing_deg, enemy_speed, uav_start_speed, uav_max_speed, a_acc, a_dec,
                                             pos_tol, vel_tol, dt, k_lead, distance_margin, max_steps, -1)
            info = chase_time_disposal(info)
            chase_info.append(info)


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
        max_steps=200000,  # 最长仿真步数（防止死循环）
        num = -1
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
    last_target_local = target_local.copy() #记录最后一次目标位置

    for _ in range(max_steps):
        # 敌点位置
        target_t_local = target_local + enemy_speed * t * f_target
        last_target_local = target_t_local  #记录

        # 到目标的相对向量
        dir_vector_r = target_t_local - uav_local
        dist = float(np.linalg.norm(dir_vector_r))

        # 收敛判据：位置逼近 & 速度逼近敌速
        if dist < pos_tol and abs(v - target_speed) < vel_tol:
            break

        # 前视引导
        # r0、r_par、r_perp
        r0 = target_t_local - uav_local
        r_par = float(np.dot(r0, f_target))
        r_perp_vec = r0 - r_par * f_target
        r_perp = float(np.linalg.norm(r_perp_vec))

        # 选用用于解方程的速度：用 v_cmd 或 uav_max_speed 都行（更激进用 uav_max_speed）
        v_for_solve = max(v, 1e-3)  # 或者 uav_max_speed

        # 若 v_for_solve <= target_speed，解析解不可用，退回原来的前视法
        if v_for_solve > target_speed:
            A = (v_for_solve ** 2 - target_speed ** 2)
            B = r_par * target_speed
            C = r_perp ** 2 + r_par ** 2
            t_star = (B + math.sqrt(B * B + A * C)) / A
            P_star = target_local + target_speed * (t + t_star) * f_target
            dvec = P_star - uav_local
        else:
            # 退回原前视
            t_lead = float(np.clip(k_lead * dist / max(v, 1e-3), 0.1, 3.0))
            P_lead = target_local + target_speed * (t + t_lead) * f_target
            dvec = P_lead - uav_local

        dnorm = float(np.linalg.norm(dvec))
        if dnorm < 1e-6: break
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
        uav_local = uav_local + v_new * dt * dir_hat
        traj_uav.append(uav_local.copy())

        # 时间/路程累计
        S_uav += v_new * dt
        v = v_new
        t += dt

    t_total = t
    S_e = enemy_speed * t_total

    # 可达性粗检查：若跑满步数仍未收敛，多半是几何+速度不可达或容差太严
    reached = (t < max_steps * dt)

    intercept_dms = None
    intercept_local = None
    try:
        if reached:
            intercept_local = uav_local  # 收敛时无人机位置
            # 如果有 local_to_geodetic_dms，直接用
            intercept_dms = converter_point.local_to_geodetic_dms(intercept_local)
            intercept_dms[2] = target_dms[2]

    except Exception as e:
        pass

    traj_dms = []

    def get_traj_point_dms(num: int):
        if num < 0 or num >= len(traj_uav):
            traj_dms = None
            return traj_dms

        point_local = traj_uav[num]
        traj_dms = converter_point.local_to_geodetic_dms(point_local)
        traj_dms[2] = target_dms[2]
        return traj_dms

    traj_dms = get_traj_point_dms(num)


    return {
        "reached": reached,
        "t_total": t_total,
        "t_acc": t_acc,
        "t_cruise": t_cruise,
        "t_dec": t_dec,
        "S_uav": S_uav,
        "S_enemy": S_e,
        "triangular": tri_sign,
        "traj_u_local": np.array(traj_uav),  # ENU轨迹，需可视化时使用
        "intercept_local": intercept_local,
        "intercept_dms": intercept_dms,
        "traj_dms": traj_dms
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
def uav_timed_position(first_center_pos, uav_speed, uav_max_speed, uav_deceleration_speed, enemy_center, acceleration, state_end_times, state_dms, enemy_speed, time):

    #转换波次中心、敌群中心、敌群被追击的位置、敌群朝向
    timed_position = []
    first_lat, first_lon, first_alt = GeodeticConverter.decimal_dms_to_degrees(first_center_pos)
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)

    #初始航向
    bearing = converter.calculate_flight_bearing(first_lat, first_lon, enemy_lat, enemy_lon)

    #追击结束跟随航向(敌群航向)
    after_chase_bearing = converter.calculate_flight_bearing(enemy_lat, enemy_lon, first_lat, first_lon)

    # 第一二波次匀速结束/第三波次加匀速结束的位置
    uniform_over_lat, uniform_over_lon, uniform_over_alt = GeodeticConverter.decimal_dms_to_degrees(state_dms[2]) #改为匀速或加匀速结束的点[2]

    #减速结束的位置
    dec_over_lat, dec_over_lon, dec_over_alt = GeodeticConverter.decimal_dms_to_degrees(state_dms[3])  # 改成减速后的点[3]

    # 转弯结束的位置，总转弯角度/弧度，半径及航向角
    angle_deg = 180
    angle_deg_rad = math.radians(angle_deg)
    turning_radius = (uav_deceleration_speed ** 2) / (GRAVITY_EARTH * math.tan(math.radians(45)))
    bearing_deg = converter.calculate_flight_bearing(dec_over_lat, dec_over_lon, enemy_lat, enemy_lon)
    turn_over_lat, turn_over_lon, turn_over_alt  = GeodeticConverter.decimal_dms_to_degrees(state_dms[4])  # 改点turn_over_uav是dms，后面追击函数需要用

    # 遍历每个标准时间点，判断三波次无人机的状态
    if time <= state_end_times[2]: #匀速阶段 分12和3两种情况
        if state_end_times [0] != state_end_times[1]:
            pass
        else:
            uniform_process_distance = uav_speed * time
            #匀速阶段time处的位置
            uav_chase_lat, uav_chase_lon, uav_chase_alt = converter.calculate_destination_point(first_lat, first_lon, first_alt, bearing, uniform_process_distance, 0) #用初始点
    elif state_end_times[2] < time <= state_end_times[3]:#减速阶段
        dec_time = time - state_end_times[2]
        dec_distance = uav_speed * dec_time - (1/2) * acceleration * dec_time ** 2

        #减速阶段time处的位置
        uav_chase_lat, uav_chase_lon, uav_chase_alt = converter.calculate_destination_point(uniform_over_lat, uniform_over_lon, uniform_over_alt,
                                                                                      bearing, dec_distance, 0)
    elif state_end_times[3] < time <= state_end_times[4]: #转弯阶段 不接收数据不传出数据(开始转弯到开始追击)

        #当前时间以及移动的弧度
        turning_time = (angle_deg_rad * turning_radius) / uav_deceleration_speed
        elapsed_turn_time = time - state_end_times[3]
        angle_turned_rad = (elapsed_turn_time / turning_time) * angle_deg_rad #用时间比例计算当前位置

        # 转弯阶段time处的位置 bearing_deg是相遇点和初始敌群位置计算得到的
        uav_chase_lat, uav_chase_lon, uav_chase_alt = calculate_turning_position_with_bearing(
            dec_over_lat, dec_over_lon, dec_over_alt, turning_radius, angle_turned_rad, bearing_deg, direction='right')
    elif time > state_end_times[4]:  # 追击阶段 不接收数据不传出数据 traj_u_local
        enemy_after_turn_pos = enemy_timed_position(enemy_center, enemy_center, enemy_speed, first_center_pos, state_end_times[4])
        if state_end_times[4] < time <= state_end_times[5]:
            num = round(20 * (time  - state_end_times[4]))
            # print("步长时间：", time, state_end_times[4], num)
            chase_info = pursue_moving_point(state_dms[4], enemy_after_turn_pos, after_chase_bearing, enemy_speed,
                                                                                    100, 500, 80, 80, 30, 1, 0.05, 0.6, 20, 100000, num)
            uav_chase_dms = chase_info["traj_dms"]
            uav_chase_lat, uav_chase_lon, uav_chase_alt = GeodeticConverter.decimal_dms_to_degrees(uav_chase_dms)
        elif time > state_end_times[5]:
            #得到追击最终的位置
            uav_chase_over_lat, uav_chase_over_lon, uav_chase_over_alt = GeodeticConverter.decimal_dms_to_degrees(state_dms[5])
            after_chase_time = time - state_end_times[5]
            after_chase_dis = enemy_speed * after_chase_time
            uav_chase_lat, uav_chase_lon, uav_chase_alt = converter.calculate_destination_point(uav_chase_over_lat, uav_chase_over_lon, uav_chase_over_alt,
                                                                                                after_chase_bearing, after_chase_dis,
                                                                                                0)
    timed_position = converter.local_to_geodetic_dms(converter.geodetic_to_local(uav_chase_lat, uav_chase_lon, uav_chase_alt))
    # print(f"某时刻无人机位置", timed_position)
    return timed_position


# 敌群定时输出位置
def enemy_timed_position(enemy_pos, enemy_center, enemy_speed, uav_center, time):
    #转换敌群和无人机的位置形式
    output_enemy_timed_position = []
    uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(uav_center)
    enemy_center_lat, enemy_center_lon, enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_pos)
    enemy_bearing = converter.calculate_flight_bearing(enemy_center_lat, enemy_center_lon, uav_lat, uav_lon)

    enemy_action_lat, enemy_action_lon, enemy_action_alt = converter.calculate_destination_point(enemy_lat, enemy_lon,
                                                                                                 enemy_alt,
                                                                                                 enemy_bearing,
                                                                                                 time * enemy_speed,
                                                                                                 0)

    output_enemy_timed_position = converter.local_to_geodetic_dms(converter.geodetic_to_local(enemy_action_lat, enemy_action_lon, enemy_action_alt))
    return output_enemy_timed_position


#每一架无人机根据其转弯结束的时间确定追击后占位——具体位置（占位整体结构以确定，即与敌群的相对位置确定）
'''由于placements_dms为敌群初始占位
    所以在这个函数中需要根据初始占位来确定每一架无人机转弯结束后需要追击的位置
    以placements_dms[0]，time_info[0]为起始位置/时间，第n个点随第n个时间移动
    移动方式需要敌群速度和行动方向'''
def uav_turned_specific_position(enemy_center, enemy_speed, uav_center, time_info, placements_dms, first_uav_num, second_uav_num, tag_full_second):
    #转换敌群和无人机的位置形式
    output_enemy_timed_position = []
    # uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(uav_center)
    # enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    # enemy_bearing = converter.calculate_flight_bearing(enemy_lat, enemy_lon, uav_lat, uav_lon)

    #改变数值，由小到大排序，默认False
    time_info = sorted(time_info)
    uav_tag = "None"
    print("时间数据长度",len(time_info))
    print("第二波次无人机数",second_uav_num)

    #通过时间点的个数判断第几波次
    if len(time_info) == first_uav_num:
        uav_tag = "first"
    elif len(time_info) == second_uav_num:
        uav_tag = "second"
    elif len(time_info) == second_uav_num - 1 and tag_full_second == True:
        uav_tag = "second"

    print("uav_tag", uav_tag)
    #如果是第n波次，占位按照第n波次的执行
    if uav_tag == "first":
        position_section = placements_dms[:first_uav_num]
    elif uav_tag == "second":
        position_section = placements_dms[first_uav_num+1:]

    # 遍历每个时间，计算每一架无人机应到过顶占位的位置
    for position_sec, time in zip(position_section, time_info):
        output_enemy_position = enemy_timed_position(position_sec, enemy_center, enemy_speed, uav_center, time)
        output_enemy_timed_position.append(output_enemy_position)
    # print("输出位置：", output_enemy_timed_position)

    return output_enemy_timed_position


#根据输入的实时信息表得到相对位置
def real_relative_position(real_time, last_uav_strat_turn_time, second_uav_after_turn_time, relative_info):

    if real_time < last_uav_strat_turn_time:
        print("当前时刻未有无人机进行转弯")
        return 0

    elif real_time > last_uav_strat_turn_time and real_time < second_uav_after_turn_time:
        real_relative_info = relative_info[real_time]
        return real_relative_info

    elif real_time > second_uav_after_turn_time:
        print("当前时刻所有无人机均转弯完毕")
        return 0



# 判断每个时间点下三波次无人机的状态,某一波次转弯过程中接收待转弯无人机中心（或完成追击的无人机中心）和敌群中心位置
def drone_state(uav1_state_end_times, uav2_state_end_times, uav3_state_end_times, uav_speed, uav_max_speed, enemy_speed, basepoint, enemy_center,
                first_center_pos, second_center_pos, third_center_pos, uav_deceleration_speed, acceleration, state_dms, chase_time_val=None,):#max_distances, detect_distances, uavpoint,
    result = []
    text = ""
    message = "第1波，第2波，前置转弯无人机状态"
    standard_time = list(range(round(uav3_state_end_times[3]), round(max(uav2_state_end_times))+1))
    # 遍历每个标准时间点，判断三架无人机的状态
    for time in standard_time:
        information = []
        # 记录状态
        uav3_status = get_uav_state(time, uav3_state_end_times, uav_number=3)
        uav1_status = get_uav_state(time, uav1_state_end_times, uav_number=1)
        uav2_status = get_uav_state(time, uav2_state_end_times, uav_number=2)

        #第三波次转弯且第一波次和第二波次未转弯时，需要接收1，2,e位置
        if uav3_status == "转弯" and uav1_status != "转弯" and uav2_status != "转弯":
            uav1_position = uav_timed_position(first_center_pos, uav_speed, uav_max_speed, uav_deceleration_speed, enemy_center, acceleration, uav1_state_end_times, state_dms[1], enemy_speed, time)
            uav2_position = uav_timed_position(second_center_pos, uav_speed, uav_max_speed, uav_deceleration_speed, enemy_center, acceleration, uav2_state_end_times, state_dms[2], enemy_speed, time)
            output_enemy_position = enemy_timed_position(enemy_center,enemy_center, enemy_speed, basepoint, time)
            #print(f"At time {time} UAV3 is turning, UAV1's position is {uav1_position}, UAV2's position is {uav2_position}, enemy's position is {output_enemy_position}")
            information.append(uav1_position)
            information.append(uav2_position)
            information.append(output_enemy_position)
            text = "前置转弯无人机，第1波第2波未转弯，接收第1波第2波和敌机位置（第1波、第2波、敌机）"

        # 第一波次转弯且第二波次未转弯时，需要接收2,e位置
        elif uav1_status == "转弯" and uav2_status != "转弯": #需要接收2，e位置
            #第三波次已经追上了，接收第二，第三波次位置和敌群位置
            uav2_position = uav_timed_position(second_center_pos, uav_speed, uav_max_speed,uav_deceleration_speed, enemy_center, acceleration,uav2_state_end_times, state_dms[2], enemy_speed, time)
            uav3_position = uav_timed_position(third_center_pos, uav_speed, uav_max_speed,uav_deceleration_speed, enemy_center, acceleration,uav3_state_end_times, state_dms[0],  enemy_speed, time)
            output_enemy_position = enemy_timed_position(enemy_center, enemy_center, enemy_speed, basepoint, time)
            #print( f"At time {time} UAV1 is turning, UAV2's position is {uav2_position}, UAV3's position is {uav3_position}, enemy's position is {output_enemy_position}")
            information.append(uav2_position)
            information.append(uav3_position)
            information.append(output_enemy_position)
            text = "第1波转弯，第2波未转弯，接收第2波和敌机位置（第2波、前置转弯无人机、敌机）"

        # 第一波次和第二波次都转弯时，需要接收3,e位置
        elif uav1_status == "转弯" and uav2_status == "转弯":  # 需要接收3，e位置
            # 第一波次和第二波次转弯之前，第三波次已经追上了
            uav3_position = uav_timed_position(third_center_pos, uav_speed, uav_max_speed, uav_deceleration_speed,enemy_center, acceleration, uav3_state_end_times, state_dms[0],enemy_speed, time)
            output_enemy_position = enemy_timed_position(enemy_center, enemy_center, enemy_speed, basepoint, time)
            #print(f"At time {time} UAV1 and UVA2 are turning, UAV3's position is {uav3_position}, enemy's position is {output_enemy_position}")
            information.append(uav3_position)
            information.append(output_enemy_position)
            text = "第1波和第2波均转弯，前置转弯无人机已转完，接收前置转弯无人机和敌机位置（前置转弯无人机、敌机）"

        # 第二波次转弯,第一波次开始追击时，需要接收3,e位置
        elif uav2_status == "转弯" and uav1_status != "转弯" : #需要接收3，e位置
            #第三波次追上了
            uav3_position = uav_timed_position(third_center_pos, uav_speed, uav_max_speed,uav_deceleration_speed, enemy_center, acceleration, uav3_state_end_times, state_dms[0], enemy_speed, time)
            output_enemy_position = enemy_timed_position(enemy_center, enemy_center, enemy_speed, basepoint, time)
            information.append(uav3_position)
            information.append(output_enemy_position)
            #print( f"At time {time} UAV2 is turning, UAV3's position is {uav3_position}, enemy's position is {output_enemy_position}")
            text = "第2波转弯，第1波可能转完，接收前置转弯无人机和敌机位置（前置转弯无人机、敌机）"
        else:
            information = []

        result.append((time, message, uav1_status, uav2_status, uav3_status, text, information))#列表后面加位置
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
    # print("uav_chase_new_pos:", uav_chase_new_pos)

    return uav_bearing, uav_speed, uav_chase_new_pos


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

#计算飞行方向
def bearing (dms1, dms2):
    lat1, lon1, alt1 = GeodeticConverter.decimal_dms_to_degrees(dms1)
    lat2, lon2, alt2 = GeodeticConverter.decimal_dms_to_degrees(dms2)
    bearing = converter.calculate_flight_bearing(lat1, lon1, lat2, lon2)

    return bearing

#对若干last_time_info按列求均值，但如果只有1组数据，直接返回，如果2组以上，求平均值，如果没有有效数据，抛出异常
def elementwise_mean_last(last_time_info_all, limit=None):

    if last_time_info_all is None:
        raise ValueError("last_time_info_all 不能为空")
    if limit is None:
        limit = len(last_time_info_all)

    # 取前 limit 组，并过滤掉 None/空列表
    use = [x for x in last_time_info_all[:limit] if x]
    if not use:
        raise ValueError("没有可用于求平均的数据组")
    if len(use) == 1:
        return list(use[0])  # 直接返回唯一一组的拷贝

    # 对齐到最短长度，避免 zip 截断隐式行为带来的混淆
    L = min(len(x) for x in use)
    if L == 0:
        return []  # 或者 raise ValueError("数据长度为0")

    # 逐列平均
    return [sum(row[i] for row in use) / len(use) for i in range(L)]




def plot_positions(uav_first_geo_init, uav_second_geo_init,
                    enemy_center_init, meet_last_enemy_center, chase_enemy_center,
                    last_uav, meet_last_uav_point, after_turn_last_uav, chase_last_point,
                    meet_first_uav_point, after_turn_first_uav, meet_second_uav_point, after_turn_second_uav, output_each_enemy_pos1,output_each_enemy_pos2,corners, placements_dms, uav_base):


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
    # print("中心相遇点位", meet_last_lat, meet_last_lon, meet_last_alt)

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
    # ==================================追击位置============================================
    chase_first_uav_lat, chase_first_uav_lon, chase_first_alt = geo_to_degrees(output_each_enemy_pos1)
    chase_second_uav_lat, chase_second_uav_lon, chase_second_alt = geo_to_degrees(output_each_enemy_pos2)
    chase_first_uav_positions = {
        f"chase_first_UAV{i + 1}": [lat, lon, alt]  # 键名为 "chase_first_UAVi"，值为 [纬度, 经度, 海拔]
        for i, (lat, lon, alt) in enumerate(zip(chase_first_uav_lat, chase_first_uav_lon, chase_first_alt ))
    }

    # 使用 zip 函数将三个列表打包
    chase_second_uav_positions = {
        f"chase_second_UAV{i + 1}": [lat, lon, alt]  # 键名为 "chase_second_UAVi"，值为 [纬度, 经度, 海拔]
        for i, (lat, lon, alt) in enumerate(zip(chase_second_uav_lat, chase_second_uav_lon, chase_second_alt))
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
    scatter_points_0 = create_uav_scatter(points, color='magenta')#points里面的点，包括东南西北等
    scatter_points_1 = create_uav_scatter(first_uav_positions, color='red')#第一波次无人机初始位置
    scatter_points_2 = create_uav_scatter(second_uav_positions, color='orange')#第二波次无人机初始位置
    scatter_points_3 = create_uav_scatter(meet_first_uav_positions, color='pink')#第一波次无人机与敌群相遇
    scatter_points_4 = create_uav_scatter(turn_first_uav_positions, color='yellow')  # 第一波次无人机转弯之后
    scatter_points_5 = create_uav_scatter(meet_second_uav_positions, color='gray')  # 第二波次无人机与敌群相遇
    scatter_points_6 = create_uav_scatter(turn_second_uav_positions, color='brown')  # 第二波次无人机转弯之后
    scatter_points_7 = create_uav_scatter(chase_first_uav_positions, color='purple')  # 第一波次无人机追击点位
    scatter_points_8 = create_uav_scatter(chase_second_uav_positions, color='black')  # 第二波次无人机追击点位


    scatter_points = scatter_points_0 + scatter_points_1 + scatter_points_2 + scatter_points_3 + scatter_points_4 + scatter_points_5 + scatter_points_6+ scatter_points_7 + scatter_points_8#合并字典

    #存储所有点，为了找坐标端点
    merged_dict = {**points, **first_uav_positions, **second_uav_positions, **chase_first_uav_positions, **chase_second_uav_positions}#解压
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
    # print(all_lons)
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

def dataset_realtime(config: dict | None = None) -> dict:
    cfg = config or {}
    auto_import = cfg.get("auto_import_afsim", True)
    afsim_args = cfg.get("afsim_args", [])

    if auto_import:
        project_root = Path(__file__).resolve().parent  # 手动运行时就在这个目录
        script_rel = Path("data") / "import_afsim_data.py"  # 和手动一样的相对路径
        try:
            res = subprocess.run(
                [sys.executable, str(script_rel), *afsim_args],
                cwd=str(project_root),  # 关键：保持与手动运行相同的 CWD
                check=True,
                capture_output=True,
                text=True,
            )
            if res.stdout.strip():
                print("[import_afsim_data.py][stdout]\n", res.stdout)
            if res.stderr.strip():
                print("[import_afsim_data.py][stderr]\n", res.stderr)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"AFSIM 数据导入失败：{e.stderr or e}")


#接口函数
def run_port(config: dict | None = None) -> dict:


    # ============= 1) 载入数据 & 覆盖配置 =============
    data = dataset()
    if config:
        for k, v in config.items():
            data[k] = v

    # 增加转弯时速度为100
    uav_deceleration_speed = data.get('uav_deceleration_speed', 100)  # 默认 100
    tag_full_second = False

    # ============= 2) 构建全局坐标轴converter =============
    global converter
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(data['enemy_approx'])
    base_lat, base_lon, base_alt = GeodeticConverter.decimal_dms_to_degrees(data['basepoint'])
    converter = GeodeticConverter.GeodeticToLocalConverter(base_lat, base_lon, base_alt,
                                                           enemy_lat, enemy_lon, enemy_alt)

    # ============= 3) 处理第2波次纵队最后一架无人机 =============
    # 对第2波次无人机纵队情况进行分析
    last_uav_num = column_judge_last_uav(data['column'], data['num_of_column'], data['second_num'])
    # 初始敌群飞行方向
    bearing_enemy = bearing(data['enemy_approx'], data['basepoint'])

    # 占位策略，得到无人机上的占位信息，便于追击
    front_dms, corners, placements_dms = generate_placements_with_bearing(
        data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange'],
        distance_m=data['near_detect_distance'],
        bearing_deg=bearing_enemy,  # 敌方从东向西
        exclusion_radius_m=None,
        preplaced_dms=[],
        start_side='left',  # 先左
        max_rows=100, first_uav_num=data['first_num'], second_uav_num=data['second_num']
    )

    # 求第2波次无人机中心
    second_uav_center = calculate_center_dms(data['second_uavs'])

    # 已知敌机中心和经纬度范围求得敌机距离base的最大距离，即最远边界值
    max_dist, max_enemy_dms, min_dist, min_enemy_dms = max_distance(
        data['basepoint'], data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange'])

    # 计算当第1波次无人机能将敌方全纳入探测范围时的时间
    time_detect = function_last_detection_time(
        data['minimum_speed'], data['speed'], max_dist, data['detect_distance'])
    print("新数据的相遇时间:",time_detect)

    second_uavs = [data[f"second_uavs_{i}"] for i in range(1, data['column'] + 1)]


    last_time_info_all = []
    last_uav_chase_time_all = []
    chase_last_uav_dms = []
    new_last_uav_time_info = []
    third_uav_dms = []
    last_begin_turn_dms_all = []
    after_turn_last_dms_all = []
    last_state_dms_info = []
    regression_remian_second_uavs = []


    for i in range(last_uav_num):
        # 对第2波次无人机按距离由远到近进行排序，得到sorted_second[0]就是末尾那架无人机(循环)
        last_uavs_sorted, last_uavs_dms = uav_sorted_distances_points(
            second_uavs[i], second_uav_center, data['enemy_approx'], reverse=True)
        third_uav_dms.append(last_uavs_dms[0])

        remain_second_dms = list(zip([id[1] for id in last_uavs_sorted], [dms[3] for dms in last_uavs_sorted]))
        remain_second_dms = remain_second_dms[1:]  #去掉最后一架无人机时编号+dms
        regressioin_remain_single_second = [row[1] for row in sorted(remain_second_dms, key=lambda x: x[0])]

        for dms in regressioin_remain_single_second:
            regression_remian_second_uavs.append(dms)


        # 计算最后一架无人机转弯起点位置、转弯180度后位置、追赶位置以及敌方中心在我方无人机开始转弯时位置、被追赶上位置；转弯时间，追逐时间
        last_state_info, last_begin_turn_dms, after_turning_last_uav_dms, new_bearing_enemy, \
            meet_last_enemy_dms, after_turn_enemy_dms, last_time_info = last_uav_move_strategy(
                data['minimum_speed'], data['maximum_speed'], uav_deceleration_speed, data['speed'],
                max_dist, data['detect_distance'], last_uavs_dms[0], min_enemy_dms,
                data['acceleration'], data['enemy_approx']
            )

        last_time_info_all.append(last_time_info)
        last_begin_turn_dms_all.append(last_begin_turn_dms)
        after_turn_last_dms_all.append(after_turning_last_uav_dms)

        # 得到转弯后追击敌方时所需要花费的时间距离等
        if i == 0:
            last_state_dms_info.append(last_state_info)
            tgt_center = after_turn_enemy_dms
            tgt_center[2] = str(float(after_turn_enemy_dms[2]) + data['near_detect_distance'][1])
            chase_time_info, _, chase_uav_dms = last_uav_chase_strategy(
                after_turning_last_uav_dms, tgt_center, new_bearing_enemy,
                data['speed'], data['near_detect_distance'][1], 0,
                data['enemy_approx'], data['basepoint'], -1, data['maximum_speed']
            )
            last_uav_chase_time_all.append(chase_time_info)
            chase_last_uav_dms.append(chase_uav_dms)
            last_state_dms_info[0].append(chase_uav_dms)
            last_state_dms_info = last_state_dms_info[0]

        else:
            chase_time_info, _, chase_uav_dms = last_uav_chase_strategy(
                after_turning_last_uav_dms, placements_dms[i - 1 + data['first_num']],
                new_bearing_enemy, data['speed'],
                data['near_detect_distance'][1], last_time_info_all[i][4],
                data['enemy_approx'], data['basepoint'], 30, data['maximum_speed']
            )
            last_uav_chase_time_all.append(chase_time_info)
            chase_last_uav_dms.append(chase_uav_dms)

    #说明此时纵队无人机处于刚好排满情况
    if last_uav_num == data['column']:
        tag_full_second = True
    else:
        regression_remian_second_uavs += data[f"second_uavs_{last_uav_num + 1}"]

    # 再次计算占位策略，除掉重复点
    front_dms, corners, placements_dms = generate_placements_with_bearing(
        data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange'],
        distance_m=data['near_detect_distance'],
        bearing_deg=bearing_enemy,
        exclusion_radius_m=None,
        preplaced_dms=third_uav_dms,
        start_side='left',
        max_rows=100, first_uav_num=data['first_num'], second_uav_num=data['second_num']
    )

    print("当前的占位信息：",placements_dms)

    new_last_uav_time_info = elementwise_mean_last(last_time_info_all, 2)
    print("得到的最后一架无人机信息表:",new_last_uav_time_info)

    # =================================处理第1波次无人机===========================================

    # 求第1波次无人机中心
    first_uav_center = calculate_center_dms(data['first_uavs'])

    # 对第一批次无人机进行排序，距离敌群由近到远，并计算相遇时间（包含安全距离）
    first_sorted, first_sorted_dms = uav_sorted_distances_points(
        data['first_uavs'], first_uav_center, max_enemy_dms, reverse=False)

    # result为转弯后的点位
    first_state_dms_info, meet_first_uav_dms, after_turn_first_uav_dms, first_uav_time_info, \
        bearing_enemy1, meet_single_time1, first_continue_meet_time, result1, turning_time1 = \
        first_uav_move_strategy(
            data['minimum_speed'], uav_deceleration_speed, data['maximum_speed'], data['speed'],
            first_sorted, first_uav_center, max_enemy_dms, 1000, data['acceleration'], time_detect
        )
    result_turn_uav_pos1 = [i[2] for i in result1]

    # =================================处理第2波次无人机===========================================

    remain_second_uav_sorted, remain_second_sorted_dms = uav_sorted_distances_points(
        regression_remian_second_uavs, second_uav_center, data['enemy_approx'], reverse=False)
    # print("剩下的第二波次无人机数:",len(remain_second_uav_sorted))
    second_state_dms_info, meet_second_uav_dms, after_turn_second_uav_dms, second_uav_time_info, \
        bearing_enemy2, meet_single_time2, second_continue_meet_time, result2, turning_time2 = \
        first_uav_move_strategy(
            data['minimum_speed'], uav_deceleration_speed, data['maximum_speed'], data['speed'],
            remain_second_uav_sorted, second_uav_center, max_enemy_dms, 1000, data['acceleration'], time_detect
        )
    result_turn_uav_pos2 = [i[2] for i in result2]

    # ===============计算两波次无人机从敌群进入视场到转弯结束的时间====================
    enter_to_meet_time = meet_single_time1 + meet_single_time2
    #第n波次每一架无人机转弯后应追击的占位
    output_each_enemy_pos1 = uav_turned_specific_position(
        data['enemy_approx'], data['speed'], data['basepoint'],
        meet_single_time1, placements_dms, data['first_num'], data['second_num'] - last_uav_num, tag_full_second
    )
    output_each_enemy_pos2 = uav_turned_specific_position(
        data['enemy_approx'], data['speed'], data['basepoint'],
        meet_single_time2, placements_dms, data['first_num'], data['second_num'] - last_uav_num, tag_full_second
    )


    # 追击时间
    first_chase_info = chase_strategy(
        output_each_enemy_pos1, result_turn_uav_pos1,
        data['first_num'], data['second_num'] - last_uav_num,
        bearing_enemy1, data['speed'],
        uav_start_speed=100, uav_max_speed=data['maximum_speed'],
        a_acc=data['acceleration'], a_dec=data['acceleration'],
        pos_tol=30, vel_tol=1, dt=0.05, k_lead=0.6,
        distance_margin=20, max_steps=100000
    )
    second_chase_info = chase_strategy(
        output_each_enemy_pos2, result_turn_uav_pos2,
        data['first_num'], data['second_num'] - last_uav_num,
        bearing_enemy2, data['speed'],
        uav_start_speed=100, uav_max_speed=data['maximum_speed'],
        a_acc=data['acceleration'], a_dec=data['acceleration'],
        pos_tol=30, vel_tol=1, dt=0.05, k_lead=0.6,
        distance_margin=20, max_steps=100000
    )

    result_first = [item[0] for item in result1]
    result_second = [item[0] for item in result2]

    first_chase_dms = [item[4] for item in first_chase_info]
    second_chase_dms = [item[4] for item in second_chase_info]

    # 合并
    first_uav_chase_all = list(zip(result_first, first_chase_dms))
    second_uav_chase_all = list(zip(result_second, second_chase_dms))

    first_uav_chase_time_list = list(zip(result_first, [item[0] for item in first_chase_info]))
    second_uav_chase_time_list = list(zip(result_second, [item[0] for item in second_chase_info]))

    time_uav_info = generate_uav_time_info(new_last_uav_time_info, first_uav_time_info, second_uav_time_info)
    state_uav_info = generate_uav_time_info(last_state_dms_info, first_state_dms_info, second_state_dms_info)
    # print("总时间信息表:", time_uav_info) #包括时间，纬度/经度/海拔

    # 纵队最后一架无人机信息整理
    last_uav_meet_time = [row[3] for row in last_time_info_all]  # 无人机与敌群相遇时间
    last_uav_turn_time = [row[4] - row[3] for row in last_time_info_all]  # 无人机转弯需要花费时间
    last_uav_chase_time = [row[0] for row in last_uav_chase_time_all]  # 无人机追击占位需要花费时间

    # 第1波次无人机信息整理
    first_uav_turn_point = [row[0] for row in sorted(meet_first_uav_dms, key=lambda x: x[1])]
    first_uav_after_turn_point = [row[2] for row in sorted(result1, key=lambda x: x[0])]
    first_uav_chase_point = [row[1] for row in sorted(first_uav_chase_all, key=lambda x: x[0])]
    first_uav_meet_time = [row[1] for row in sorted(first_continue_meet_time, key=lambda x: x[0])]
    first_uav_chase_time = [row[1] for row in sorted(first_uav_chase_time_list, key=lambda x: x[0])]

    # 第2波次无人机信息整理
    second_uav_turn_point = [row[0] for row in sorted(meet_second_uav_dms, key=lambda x: x[1])]
    second_uav_after_turn_point = [row[2] for row in sorted(result2, key=lambda x: x[0])]
    second_uav_chase_point = [row[1] for row in sorted(second_uav_chase_all, key=lambda x: x[0])]
    second_uav_meet_time = [row[1] for row in sorted(second_continue_meet_time, key=lambda x: x[0])]
    second_uav_chase_time = [row[1] for row in sorted(second_uav_chase_time_list, key=lambda x: x[0])]

    # 获取三波次追击时间
    last_time_list, first_uav_time_list, second_uav_time_list = time_uav_info[0], time_uav_info[1], time_uav_info[2]
    last_time_list.append(last_time_list[4] + last_uav_chase_time[0])
    first_uav_time_list.append(first_uav_time_list[4] + np.mean(first_uav_chase_time))
    second_uav_time_list.append(
        second_uav_time_list[4] + np.mean(second_uav_chase_time))  # last_time_list[4] + last_uav_chase_time_all[0][0]
    # print("last_time_list, first_uav_time_list, second_uav_time_list:", second_uav_time_list)

    # 相对位置
    state_result = drone_state(first_uav_time_list, second_uav_time_list, last_time_list, data['minimum_speed'],
                               data['maximum_speed'], data['speed'], data['basepoint'], data['enemy_approx'],
                               first_uav_center, second_uav_center, third_uav_dms[0], uav_deceleration_speed,
                               data['acceleration'], state_uav_info, chase_time_val=None)

    # real_relative_position_info = real_relative_position(real_time - 1, time_uav_info[0][3], time_uav_info[2][4],
    #                                                 state_result)
    # print("实时位置:", real_relative_position_info)

    print("第2波次的长度:",len(regression_remian_second_uavs),"2222222", len(second_uav_after_turn_point),"3333333",len(second_uav_chase_point),"44444444",len(second_uav_meet_time),
          "55555555555",len(second_uav_chase_time))


    out = {
        "uavs_speed": {
            "current": data['minimum_speed'],
            "acceleration": data['acceleration'],
            "max_speed": data['maximum_speed'],
            "turn_speed": uav_deceleration_speed,
        },
        "enemy_speed": data['speed'],

        #==================初始信息===================
        "first_init_point":  data['first_uavs'],
        "second_init_point": regression_remian_second_uavs,
        "last_init_point": third_uav_dms,
        "enemy_init_point": data['enemy_approx'],


        # ==============每个纵队最后一架无人机移动点位============
        "last_uav_turn_point": last_begin_turn_dms_all,
        "last_uav_after_turn_point": after_turn_last_dms_all,
        "last_uav_chase_point": chase_last_uav_dms,

        #==============每个纵队最后一架无人机移动时间============
        "last_uav_meet_time": last_uav_meet_time,
        "last_uav_turn_time": last_uav_turn_time,
        "last_uav_chase_time": last_uav_chase_time,

        #==============第1波次无人机移动点位============
        "first_uav_turn_point": first_uav_turn_point,
        "first_uav_after_turn_point": first_uav_after_turn_point,
        "first_uav_chase_point": first_uav_chase_point,

        #==============第1波次无人机移动时间============
        "first_uav_meet_time": first_uav_meet_time,
        "first_uav_turn_time": turning_time1,
        "first_uav_chase_time": first_uav_chase_time,

        #==============第2波次无人机移动点位============
        "second_uav_turn_point": second_uav_turn_point,
        "second_uav_after_turn_point": second_uav_after_turn_point,
        "second_uav_chase_point": second_uav_chase_point,

        #==============第2波次无人机移动时间============
        "second_uav_meet_time": second_uav_meet_time,
        "second_uav_turn_time": turning_time2,
        "second_uav_chase_time": second_uav_chase_time,

        #================转弯波次相对位置===============
        "relative_position":  state_result,
    }
    print("数据导出完成")
    return out


def main(config: dict | None = None):
    """
    与旧代码风格兼容的入口：POST 进来的 JSON 会作为 config 覆盖默认 dataset。
    """
    return run_port(config or {})



if __name__ == "__main__":
    # dataset_realtime()
    result = run_port()  # 可传 config 覆盖默认数据
    # 你若还想显示可视化，可在这里读取 result 后，调用原有 plot_* 方法
    # e.g. plot_positions(...使用 result 里的字段组织参数...)






