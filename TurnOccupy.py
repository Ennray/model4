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

    time_move = (distance_move - safety_distance) / (uav_speed + enemy_speed)
    distance_uav = uav_speed * time_move
    distance_enemy = enemy_speed * time_move
    angle = converter.calculate_climb_angle(new_alt, new_enemy_alt, distance_uav)
    uav_meet_lat, uav_meet_lon, uav_meet_alt, distance_uav_val= converter.calculate_destination_with_climb_angle(new_lat, new_lon, new_alt, bearing, distance_uav, angle)







    # 10. 输出
    print(f"无人机原位置: {uavpoint}")
    print(f"敌群原位置: {enemy_center}")
    print(f"飞行方向: {bearing:.2f}°，飞行距离: {total_distances / 1000:.2f} km")
    print(f"无人机新位置: [{new_lon:.6f}, {new_lat:.6f}, {new_alt:.1f}]")
    print(f"敌群新位置: [{new_enemy_lon:.6f}, {new_enemy_lat:.6f}, {enemy_alt:.1f}]")

    # 转为 DMS 格式
    uav_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(uav_meet_lat, uav_meet_lon, uav_meet_alt))
    enemy_dms = converter.local_to_geodetic_dms(converter.geodetic_to_local(new_enemy_lat, new_enemy_lon, enemy_alt))

    return uav_dms, enemy_dms



#纬经高转化为坐标轴分量
def geo_to_degrees(geo):
    lats, lons, alts = [], [], [] #纬经高
    for i in geo:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(i)
        lats.append(lat)
        lons.append(lon)
        alts.append(alt)
    return lats, lons, alts


#3D绘图便于观察
def plot_positions_with_centers(uav_first_geo_init, uav_second_geo_init,
                                enemy_center_init, last_uav, last_uav_point, last_enemy_center,
                                converter):

    #==================================点位转坐标=============================================
    # 初始第1波点位
    first_uav_init_lats, first_uav_init_lons, first_uav_init_alts = geo_to_degrees(uav_first_geo_init)
    # 初始第2波点位
    second_uav_init_lats, second_uav_inti_lons, second_uav_init_alts = geo_to_degrees(uav_second_geo_init)



    #===================================中心转坐标============================================
    # 初始中心（已知）
    enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center_init)
    last_enemy_center_lat, last_enemy_center_lon, last_enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(last_enemy_center)
    last_uav_lat, last_uav_lon, last_uav_alt = GeodeticConverter.decimal_dms_to_degrees(last_uav_point)
    last_lat, last_lon, last_alt = GeodeticConverter.decimal_dms_to_degrees(last_uav)

    #our_center_init_lat, our_center_init_lon, our_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(our_center_geo_init)


    #=====================================绘图==============================================
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 初始第1波
    ax.scatter(first_uav_init_lats, first_uav_init_lons, first_uav_init_alts, c='cyan', marker='x', label='First UAV Init', s=50)

    # 初始第2波
    ax.scatter(second_uav_init_lats, second_uav_inti_lons, second_uav_init_alts, c='blue', marker='o', label='Second UAV Init', s=50)

    # 初始敌群
    ax.scatter(enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt, c='red', marker='*', label='Enemy Init', s=50)
    ax.scatter(last_enemy_center_lat, last_enemy_center_lon, last_enemy_center_alt, c='green', marker='*', label='Last UAV Init', s=50)

    ax.scatter(last_lat, last_lon, last_alt, c='pink', marker='x', label='Last Init', s=50)
    ax.scatter(last_uav_lat, last_uav_lon, last_uav_alt, c='green', marker='x', label='Last UAV Init', s=50)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('3D Positions: Initial & New with Centers')
    ax.legend()
    plt.tight_layout()
    plt.show()


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
    last_uav_point, last_enemy_center= last_uav_move_strategy(data['minimum_speed'], data['maximum_speed'], data['speed'], max_distance, data['detect_distance'],
                                                              last_point, data['basepoint'], data['enemy_approx'], acceleration)
    print("last_uav_point", last_uav_point)

    plot_positions_with_centers(data['first_uavs'], data['second_uavs'], data['enemy_approx'], last_point, last_uav_point, last_enemy_center, converter)




