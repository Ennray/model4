import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
from sympy.physics.units import acceleration

import velocity_recong
from sklearn.cluster import DBSCAN
import plotly.graph_objs as go

from numpy.ma.core import remainder

import GeodeticConverter
import outdata
from FollowPositition import safety_distance
from GeodeticConverter import dms_to_decimal
from data.dataset import dataset
from sympy import symbols, solve, Eq

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
#对第一批uav进行排序，由敌群的近到远
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
#计算第一批无人机与敌群相遇的时间
def first_meet_enemy_time(first_sorted, enemy_pos,uav_speed, enemy_speed, safe_distence):
    meet_time = []
    safe_distence = 1000
    e_lat, e_lon ,e_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_pos)
    print("ssssss", e_lat,e_lon,e_alt)
    for i,uav in enumerate(first_sorted):
        uav_pos = uav[0].tolist()
        # print("uav_pos", uav_pos)
        uav_number = uav [1]
        # print("number:",uav_number)
        dms_uav_pos = converter.local_to_geodetic_dms(uav_pos)
        lat, lon ,alt = GeodeticConverter.decimal_dms_to_degrees(dms_uav_pos)
        # 计算无人机与敌机之间的初始距离
        # print(type(uav_pos))  # 打印 uav_pos 的类型
        # print(uav_pos)         # 打印 uav_pos 的内容
        distance_between_uav_enemy = converter.calculate_spherical_distance(lat, lon, alt, e_lat, e_lon, e_alt)
        # print("distance_between_uav_enemy:", distance_between_uav_enemy)
        # 计算相遇所需时间
        meet_time_value = (distance_between_uav_enemy - safe_distence) / (uav_speed + enemy_speed)
        # print("meet:",meet_time_value)
        # 添加到meet_time列表中
        meet_time.append((uav_number, meet_time_value))
    print("meet_time_all:",meet_time)
    return meet_time
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


#纵队最后一架无人机的飞行策略
def last_uav_move_strategy(uav_speed, uav_max_speed, enemy_speed, max_distances, detect_distances, uavpoint, basepoint, enemy_center, acceleration):

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
    bearing_enemy = converter.calculate_flight_bearing(enemy_lat, enemy_lon, lat, lon)

    #球面预测飞行点
    new_lat, new_lon, new_alt = converter.calculate_destination_point(
        lat, lon, alt, bearing, total_distances, 0
    )

    #敌群位置更新（反方向飞行）
    enemy_movedis = enemy_speed * (time + time_acc)
    new_enemy_lat, new_enemy_lon, new_enemy_alt = converter.calculate_destination_point(
        enemy_lat, enemy_lon, enemy_alt, bearing_enemy, enemy_movedis, 0
    )

    #计算此时相对距离
    distance_move = converter.calculate_spherical_distance(new_lat, new_lon, new_alt, new_enemy_lat, new_enemy_lon, new_enemy_alt)
    print("*********distance:",distance_move)
    safety_distance = 1000
    uav_deceleration_speed = 100
    deceleration_time = (uav_max_speed - uav_deceleration_speed) / acceleration
    uav_deceleration_distence = uav_max_speed * deceleration_time - 0.5 * acceleration *deceleration_time **2
    time_move = (distance_move - safety_distance + uav_max_speed * deceleration_time - uav_deceleration_distence) / (uav_max_speed + enemy_speed)
    print("匀速减速再相遇所需时间：", time_move)
    # time_move_1 = (distance_move - safety_distance) / (uav_max_speed + enemy_speed)
    # print("匀速相遇所需时间：", time_move_1)
    distance_uav = uav_max_speed * (time_move - deceleration_time)  #我方第一波次移动的距离
    print("distance:", distance_uav)
    distance_enemy = enemy_speed * time_move
    angle = converter.calculate_climb_angle(new_alt, new_enemy_alt + safety_distance, distance_uav)
    angle_rad = math.radians(angle)
    print("angle", angle)
    #开始转弯的点位（相遇位置）
    uav_meet_lat, uav_meet_lon, uav_meet_alt, distance_uav_val= converter.calculate_destination_with_climb_angle(new_lat, new_lon, new_alt, bearing, distance_uav, angle)

    # 计算转弯半径以及追赶时间和距离
    GRAVITY_EARTH = 9.80665  # 地球表面重力加速度
    turning_radius = ((uav_deceleration_speed**2) * (math.cos(angle_rad)**2)) / (GRAVITY_EARTH * math.tan(math.radians(45)))
    print("半径", turning_radius)
    turning_time = (math.pi * turning_radius) / uav_max_speed
    print("turning_time,chase_time:",turning_time)
# 使用符号计算求解追赶时间和距离
    from sympy import symbols, Eq, solve, sqrt
    chase_time, chase_distence, distence_enemy_chase = symbols('chase_time chase_distence distence_enemy_chase')
    # 建立方程组
    equations = [
        Eq(chase_distence / uav_max_speed, chase_time),  # 追击距离与时间关系
        Eq((turning_time + chase_time) * enemy_speed, distence_enemy_chase),  # 敌人在追击期间移动距离
        Eq(sqrt(distence_enemy_chase**2 + (2 * turning_radius)**2), chase_distence)  # 追击距离几何关系
    ]
    # 解方程组
    solutions = solve(equations, (chase_time, chase_distence, distence_enemy_chase))
    for sol in solutions:
        numeric_sol = [float(val.evalf()) for val in sol]
        chase_time_val, chase_distence_val, distence_enemy_chase_val = numeric_sol
    print(f"转弯后追赶时间，转弯后追赶距离，转弯及追赶过程中敌群移动距离：[{chase_time_val:.6f}, {chase_distence_val:.6f}, {distence_enemy_chase_val:.1f}]")
    # 计算转弯后无人机的位置
    after_turning_uav_lat, after_turning_uav_lon, after_turning_uav_alt = calculate_turning_position(uav_meet_lat, uav_meet_lon, uav_meet_alt, turning_radius, math.pi)
    print(f"转弯后无人机的位置: 纬度 = {after_turning_uav_lat}, 经度 = {after_turning_uav_lon}, 高度 = {after_turning_uav_alt}")
    new_chase_enemy_lat, new_chase_enemy_lon, new_chase_enemy_alt = converter.calculate_destination_point(
        new_enemy_lat, new_enemy_lon, new_enemy_alt, bearing_enemy, distence_enemy_chase_val, 0)
    new_chase_uav_lat, new_chase_uav_lon, new_chase_uav_alt = converter.calculate_destination_point(
        new_enemy_lat, new_enemy_lon, new_enemy_alt + safety_distance, bearing_enemy, distence_enemy_chase_val, 0)
    # 10. 输出
    print(f"无人机原位置: {uavpoint}")
    print(f"敌群原位置: {enemy_center}")
    print(f"飞行方向: {bearing:.2f}°，飞行距离: {total_distances / 1000:.2f} km")
    print(f"无人机相遇位置: [{uav_meet_lon:.6f}, {uav_meet_lat:.6f}, {uav_meet_alt:.1f}]")
    print(f"无人机转弯后位置: [{after_turning_uav_lon:.6f}, {after_turning_uav_lat:.6f}, {after_turning_uav_alt:.1f}]")
    print(f"敌群新位置: [{new_enemy_lon:.6f}, {new_enemy_lat:.6f}, {enemy_alt:.1f}]")
    print(f"敌群被拐弯追赶后的位置: [{new_chase_enemy_lon:.6f}, {new_chase_enemy_lat:.6f}, {new_chase_enemy_alt:.1f}]")
    print(f"无人机追赶后的位置: [{new_chase_uav_lon:.6f}, {new_chase_uav_lat:.6f}, {new_chase_uav_alt:.1f}]")
    # 转为 DMS 格式
    uav_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(new_lat, new_lon, new_alt))
    enemy_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(new_enemy_lat, new_enemy_lon, new_enemy_alt))
    after_turning_uav_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(after_turning_uav_lat, after_turning_uav_lon, after_turning_uav_alt))
    before_turning_uav_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(uav_meet_lat, uav_meet_lon, uav_meet_alt))
    uav_chase_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(new_chase_uav_lat, new_chase_uav_lon, new_chase_uav_alt))
    enemy_chase_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(new_chase_enemy_lat, new_chase_enemy_lon, new_chase_enemy_alt))

    return uav_dms, enemy_dms,uav_chase_dms,enemy_chase_dms,before_turning_uav_dms,after_turning_uav_dms
def calculate_turning_position(lat, lon, alt, turning_radius, angle):
    """
    计算转弯后的无人机位置
    lat, lon, alt: 当前无人机位置的纬度、经度和高度
    turning_radius: 转弯半径
    angle: 转弯的角度（单位：弧度）
    """
    # 地球半径（米）
    R = 6371000  # 地球半径，单位：米

    # 将经纬度转为弧度
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # 计算转弯后的纬度变化（沿纬度方向移动）
    delta_lat = turning_radius * math.cos(angle) / R
    # 计算转弯后的经度变化（沿经度方向移动）
    delta_lon = turning_radius * math.sin(angle) / (R * math.cos(lat_rad))

    # 计算新的纬度和经度
    new_lat = lat + math.degrees(delta_lat)  # 纬度变化，转换回度
    new_lon = lon + math.degrees(delta_lon)  # 经度变化，转换回度

    # 返回新位置
    return new_lat, new_lon, alt

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
                                enemy_center_init, last_uav, last_uav_point, last_enemy_center, 
                                last_uav_chase_point, last_enemy_chase_center,before_turning_uav_point,after_turning_uav_point,converter):

    #==================================点位转坐标=============================================
    # 初始第1波点位 uav_dms, enemy_dms,
    first_uav_init_lats, first_uav_init_lons, first_uav_init_alts = geo_to_degrees(uav_first_geo_init)
    # 初始第2波点位
    second_uav_init_lats, second_uav_inti_lons, second_uav_init_alts = geo_to_degrees(uav_second_geo_init)



    #===================================中心转坐标============================================
    # 初始中心（已知）
    enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center_init)#开始敌群中心
    last_enemy_center_lat, last_enemy_center_lon, last_enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(last_enemy_center)#开始转弯敌群中心
    last_uav_lat, last_uav_lon, last_uav_alt = GeodeticConverter.decimal_dms_to_degrees(last_uav_point) #开始转弯
    last_lat, last_lon, last_alt = GeodeticConverter.decimal_dms_to_degrees(last_uav)#开始
    new_chase_enemy_lat, new_chase_enemy_lon, new_chase_enemy_alt = GeodeticConverter.decimal_dms_to_degrees(last_enemy_chase_center)
    new_chase_uav_lat, new_chase_uav_lon, new_chase_uav_alt = GeodeticConverter.decimal_dms_to_degrees(last_uav_chase_point)
    after_turning_uav_lat, after_turning_uav_lon, after_turning_uav_alt = GeodeticConverter.decimal_dms_to_degrees(after_turning_uav_point)
    before_turning_uav_lat, before_turning_uav_lon, before_turning_uav_alt = GeodeticConverter.decimal_dms_to_degrees(before_turning_uav_point)

    #=====================================绘图==============================================
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 初始第1波
    ax.scatter(first_uav_init_lats, first_uav_init_lons, first_uav_init_alts, c='cyan', marker='x', label='First UAV Init', s=50)

    # 初始第2波
    ax.scatter(second_uav_init_lats, second_uav_inti_lons, second_uav_init_alts, c='blue', marker='o', label='Second UAV Init', s=50)

    # 初始敌群
    ax.scatter(enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt, c='red', marker='*', label='Enemy Init', s=50)
    ax.scatter(last_enemy_center_lat, last_enemy_center_lon, last_enemy_center_alt, c='green', marker='*', label='Turning Enemy Init', s=50)

    ax.scatter(last_lat, last_lon, last_alt, c='pink', marker='x', label='Second last UAV', s=50)
    ax.scatter(last_uav_lat, last_uav_lon, last_uav_alt, c='green', marker='x', label='Turning UAV Init', s=50)
    ax.scatter(before_turning_uav_lat, before_turning_uav_lon, before_turning_uav_alt, c='orange', marker='x', label='Before turning UAV Init', s=50)
    ax.scatter(after_turning_uav_lat, after_turning_uav_lon, after_turning_uav_alt, c='yellow', marker='x', label='After turning UAV Init', s=50)
    ax.scatter(new_chase_enemy_lat, new_chase_enemy_lon, new_chase_enemy_alt, c='black', marker='*', label='Chase enemy center', s=50)
    ax.scatter(new_chase_uav_lat, new_chase_uav_lon, new_chase_uav_alt, c='black', marker='x', label='Chase UAV point', s=50)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('3D Positions: Initial & New with Centers')
    ax.legend()
    plt.tight_layout()
    plt.show()
#无人机路径记录

if __name__ == "__main__":
    #=============================初始化数据==================================
    data = dataset()
    acceleration = 80 #加速度减速度均设为80


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

    #===========================处理第2波次无人机===============================
    #对第2波次无人机进行聚类，得到纵队情况
    cluster_second_uav = cluster_uavs_by_latitude(data['second_uavs'], converter) #聚类
    num_columns = len(cluster_second_uav) #得到类别数

    #对第2波次无人机按y从小到大进行排序，得到sorted_second[0]就是末尾那架无人机
    sorted_second = sorted_y_points(data['second_uavs'], converter)
    # 对第一批次无人机进行排序，距离敌群由近到远，并计算相遇时间（包含安全距离）
    first_uav_sorted = first_sorted_y_points(data['first_uavs'], converter)
    print("第一批排序", first_uav_sorted)
    first_meet_time = first_meet_enemy_time(first_uav_sorted,data['enemy_approx'],data['minimum_speed'],data['speed'], 1000)
    print("第一批相遇时间", first_meet_time)
    #已知敌机中心和经纬度范围求得敌机距离base的最大距离，即最远边界值
    max_distance = max_distance(data['basepoint'], data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange'])
    # print(f"max_distance, distances2 = {max_distance} ")

    #计算当第1波次无人机能将敌方全纳入探测范围时的时间
    time_detect = function_last_detection_time(data['minimum_speed'], data['speed'], max_distance, data['detect_distance'])
    # print("time_detect = ", time_detect)

    # last_uav_distance = last_uav_distance(data['basepoint'], sorted_second[0][0][1], data['minimum_speed'], data['speed'], max_distance, data['detect_distance'], converter) #求得第2波次无人机纵队最后的无人机需要提前飞出的距离
    # print("last_uav_distance", last_uav_distance)
    last_point = converter.local_to_geodetic_dms(sorted_second[0][0])
    print("*******", last_point)


    #计算最后一架无人机刚开始加速到最大速度时的位置以及敌机位置
    last_uav_point, last_enemy_center,last_uav_chase_point, last_enemy_chase_center,before_turning_uav_point,after_turning_uav_point= last_uav_move_strategy(data['minimum_speed'], data['maximum_speed'], data['speed'], max_distance, data['detect_distance'],
                                                              last_point, data['basepoint'], data['enemy_approx'], acceleration)
    print("last_uav_point", last_uav_point)
    print("enemy_point", last_enemy_center)
    print("last_uav_chase_point", last_uav_chase_point)
    print("last_enemy_chase_center", last_enemy_chase_center)
    print("before_turning_uav_point", before_turning_uav_point)
    print("after_turning_uav_point", after_turning_uav_point)
    plot_positions_with_centers(data['first_uavs'], data['second_uavs'], data['enemy_approx'], last_point, last_uav_point, last_enemy_center,last_uav_chase_point, last_enemy_chase_center,before_turning_uav_point,after_turning_uav_point,converter)




