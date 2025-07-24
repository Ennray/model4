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
def max_distance(basepoint, enemy_center, lat_range, lon_range, converter):
    ne, se, nw, sw = get_enemy_edges(enemy_center, lat_range, lon_range)
    max_distance = 0

    pos_ne = to_local(ne, converter)
    pos_se = to_local(se, converter)
    pos_nw = to_local(nw, converter)
    pos_sw = to_local(sw, converter)
    pos_base = to_local(basepoint, converter)


    #计算与base的y轴距离
    def euclidean_y(p1,p2):
        return p2[1] - p1[1]

    distances = [
        euclidean_y(pos_base, pos_ne),
        euclidean_y(pos_base, pos_se),
        euclidean_y(pos_base, pos_nw),
        euclidean_y(pos_base, pos_sw)
    ]
    distances = max(distances)
    return distances


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
    delta_enemy = enemy_start * step_time

    # 当前地理位置
    uav_lat, uav_lon, uav_alt = GeodeticConverter.decimal_dms_to_degrees(uav_start)
    enemy_lat, enemy_lon, enemy_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_start)
    base_lat, base_lon, base_alt = GeodeticConverter.decimal_dms_to_degrees(basepoint)

    uav_positions = []
    enemy_positions = []
    converters = []

    for i in range(int(steps)):
        # 创建当前局部坐标系
        converter = GeodeticToLocalConverter(base_lat, base_lon, base_alt, enemy_lat, enemy_lon, enemy_alt)

        # 当前位置 → local
        uav_local = converter.geodetic_to_local(uav_lat, uav_lon, uav_alt)
        enemy_local = converter.geodetic_to_local(enemy_lat, enemy_lon, enemy_alt)

        # 按 y 轴推进（默认 y 轴方向是 base → enemy_center）
        uav_local[1] += delta_uav
        enemy_local[1] += delta_enemy

        # local → 地理坐标
        uav_lat, uav_lon, uav_alt = converter.local_to_geodetic(uav_local)
        enemy_lat, enemy_lon, enemy_alt = converter.local_to_geodetic(enemy_local)

        # 记录每一步坐标与 converter
        uav_positions.append(degrees_to_dms(uav_lat, uav_lon, uav_alt))
        enemy_positions.append(degrees_to_dms(enemy_lat, enemy_lon, enemy_alt))
        converters.append(converter)

    return converters, uav_positions, enemy_positions


#纵队最后一架无人机的飞行策略
def last_uav_move_strategy(uav_speed, uav_max_speed, enemy_speed, max_distances, detect_distances, uavpoint, basepoint, enemy_center, acceleration, converter):
    time = function_last_detection_time(uav_speed, enemy_speed, max_distances, detect_distances)

    lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(uavpoint)
    uav_local = converter.geodetic_to_local(lat, lon, alt)

    time_acc = (uav_max_speed - uav_speed) / acceleration  # 得到加速时间
    distance_acc = (uav_max_speed ** 2 - uav_speed ** 2) / (2 * acceleration)  # 得到加速期间前进的距离

    uav_local[1] = uav_local[1] + uav_speed * time + distance_acc
    new_uav_local = uav_local
    new_geo = converter.local_to_geodetic_dms(new_uav_local)
    print("new_geo", new_geo)

    enemy_center_lat, enemy_center_lon, enemy_center_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    pos_enemy_center = converter.geodetic_to_local(enemy_center_lat, enemy_center_lon, enemy_center_alt)

    pos_enemy_center[1] = pos_enemy_center[1] - enemy_speed * (time +time_acc)
    new_geo_enemy_center = converter.local_to_geodetic_dms(pos_enemy_center)




    return new_geo, new_geo_enemy_center







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
                                enemy_center_init, last_uav_point, last_enemy_center,
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
    # ne_lat, ne_lon, ne_alt = GeodeticConverter.decimal_dms_to_degrees(ne)
    # se_lat, se_lon, se_alt = GeodeticConverter.decimal_dms_to_degrees(se)
    # nw_lat, nw_lon, nw_alt = GeodeticConverter.decimal_dms_to_degrees(nw)
    # sw_lat, sw_lon, sw_alt = GeodeticConverter.decimal_dms_to_degrees(sw)

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


    ax.scatter(last_uav_lat, last_uav_lon, last_uav_alt, c='green', marker='x', label='Last UAV Init', s=50)
    # ax.scatter(ne_lat, ne_lon, ne_alt, c='green', marker='x', label='NE', s=50)
    # ax.scatter(se_lat, se_lon, se_alt, c='yellow', marker='x', label='SE', s=50)
    # ax.scatter(nw_lat, nw_lon, nw_alt, c='magenta', marker='x', label='NW', s=50)
    # ax.scatter(sw_lat, sw_lon, sw_alt, c='cyan', marker='x', label='SW', s=50)

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
    A_lat, A_lon, A_alt = GeodeticConverter.decimal_dms_to_degrees(data['basepoint'])
    B_lat, B_lon, B_alt = GeodeticConverter.decimal_dms_to_degrees(data['enemy_approx'])
    converter = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)


    #===========================处理第2波次无人机===============================

    cluster_second_uav = cluster_uavs_by_latitude(data['second_uavs'], converter) #聚类
    num_columns = len(cluster_second_uav) #得到类别数

    sorted_second = sorted_y_points(data['second_uavs'], converter) #对第2波次无人机按y从小到大进行排序，得到sorted_second[0]就是末尾那架无人机

    ne, se, nw, sw = get_enemy_edges(data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange']) #得到敌机边界四个角
    #print("ne,se,nw,sw,enemy_approx", ne, se, nw, sw, data['enemy_approx'])

    max_distance = max_distance(data['basepoint'], data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange'], converter) #求得最远y轴距离
    # print(f"max_distance, distances2 = {max_distance} ")

    time_detect = function_last_detection_time(data['minimum_speed'], data['speed'], max_distance, data['detect_distance'])
    # print("time_detect = ", time_detect, "*******", data['detect_distance'])

    last_uav_distance = last_uav_distance(data['basepoint'], sorted_second[0][0][1], data['minimum_speed'], data['speed'], max_distance, data['detect_distance'], converter) #求得第2波次无人机纵队最后的无人机需要提前飞出的距离
    # print("last_uav_distance", last_uav_distance)
    last_point = converter.local_to_geodetic_dms(sorted_second[0][0])
    print("*******", last_point)


    last_uav_point, last_enemy_center= last_uav_move_strategy(data['minimum_speed'], data['maximum_speed'], data['speed'], max_distance, data['detect_distance'], last_point, data['basepoint'], data['enemy_approx'], acceleration, converter)
    print("last_uav_point", last_uav_point)

    plot_positions_with_centers(data['first_uavs'], data['second_uavs'], data['enemy_approx'], last_uav_point, last_enemy_center, converter)




