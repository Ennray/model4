import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
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


#计算第1波无人机距离敌群的最远距离
def max_distance(basepoint, enemy_center, lat_range, lon_range, convernter):
    ne, se, nw, sw = get_enemy_edges(enemy_center, lat_range, lon_range)
    max_distance = 0

    # 将五个点转换为坐标
    def to_local(pos_dms):
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(pos_dms)
        return converter.geodetic_to_local(lat, lon, alt)

    pos_ne = to_local(ne)
    pos_se = to_local(se)
    pos_nw = to_local(nw)
    pos_sw = to_local(sw)
    pos_base = to_local(basepoint)

    # 计算与 base 的欧氏距离
    def euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    distances = [
        euclidean(pos_base, pos_ne),
        euclidean(pos_base, pos_se),
        euclidean(pos_base, pos_nw),
        euclidean(pos_base, pos_sw)
    ]
    distances = max(distances)
    return distances


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
                                enemy_center_init, ne, se, nw, sw,
                                converter):

    #==================================点位转坐标=============================================
    # 初始第1波点位
    first_uav_init_lats, first_uav_init_lons, first_uav_init_alts = geo_to_degrees(uav_first_geo_init)
    # 初始第2波点位
    second_uav_init_lats, second_uav_inti_lons, second_uav_init_alts = geo_to_degrees(uav_second_geo_init)



    #===================================中心转坐标============================================
    # 初始中心（已知）
    enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center_init)
    ne_lat, ne_lon, ne_alt = GeodeticConverter.decimal_dms_to_degrees(ne)
    se_lat, se_lon, se_alt = GeodeticConverter.decimal_dms_to_degrees(se)
    nw_lat, nw_lon, nw_alt = GeodeticConverter.decimal_dms_to_degrees(nw)
    sw_lat, sw_lon, sw_alt = GeodeticConverter.decimal_dms_to_degrees(sw)

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
    ax.scatter(ne_lat, ne_lon, ne_alt, c='green', marker='x', label='NE', s=50)
    ax.scatter(se_lat, se_lon, se_alt, c='yellow', marker='x', label='SE', s=50)
    ax.scatter(nw_lat, nw_lon, nw_alt, c='magenta', marker='x', label='NW', s=50)
    ax.scatter(sw_lat, sw_lon, sw_alt, c='cyan', marker='x', label='SW', s=50)

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
    print(f"敌机 {data['enemy_approx']}")



    #============================创建局部坐标系================================
    A_lat, A_lon, A_alt = GeodeticConverter.decimal_dms_to_degrees(data['basepoint'])
    B_lat, B_lon, B_alt = GeodeticConverter.decimal_dms_to_degrees(data['enemy_approx'])
    converter = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)


    #===========================处理第2波次无人机===============================

    cluster_second_uav = cluster_uavs_by_latitude(data['second_uavs'], converter) #聚类
    num_columns = len(cluster_second_uav) #得到类别数

    sorted_second = sorted_y_points(data['second_uavs'], converter) #对第2波次无人机按y从小到大进行排序，得到sorted_second[0]就是末尾那架无人机

    ne, se, nw, sw = get_enemy_edges(data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange']) #得到敌机边界四个角
    print("ne,se,nw,sw,enemy_approx", ne, se, nw, sw, data['enemy_approx'])

    max_distance = max_distance(data['basepoint'], data['enemy_approx'], data['enemy_latrange'], data['enemy_lonrange'], converter)
    print(f"max_distance = {max_distance}")


    plot_positions_with_centers(data['first_uavs'], data['second_uavs'], data['enemy_approx'], ne, se, nw, sw,  converter)




