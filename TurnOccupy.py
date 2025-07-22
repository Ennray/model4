import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
import velocity_recong
from sklearn.cluster import DBSCAN

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
def cluster_uavs_by_latitude(second_points, converter, eps = 0.00005):
    #先将second_point转为坐标系
    local_coords = []
    for dms in second_points:
        lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(dms)
        local = converter.geodetic_to_local(lat, lon, alt)
        local_coords.append(local)

    #使用局部坐标系中的y轴进行聚类
    local_y = np.array([p[1] for p in local_coords]).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(local_y) #目前按照5米进行聚类
    labels = clustering.labels_  #分别打上标签

    #
    column_map = {}
    for idx, label in enumerate(labels):
        if label not in column_map: #如果这个标签（纵队）还没有在字典中，就为它新建一个空列表
            column_map[label] = []
        column_map[label].append(second_points[idx])  #将对应的原始的DMS的无人机数据加进去

    return [[col_id, uavs] for col_id, uavs in column_map.items()] #返回一个二元数组




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
                                enemy_center_geo_init, our_center_geo_init,
                                converter):

    #==================================点位转坐标=============================================
    # 初始第1波点位
    first_uav_init_lats, first_uav_init_lons, first_uav_init_alts = geo_to_degrees(uav_first_geo_init)
    # 初始第2波点位
    second_uav_init_lats, second_uav_inti_lons, second_uav_init_alts = geo_to_degrees(uav_second_geo_init)


    #===================================中心转坐标============================================
    # 初始中心（已知）
    enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center_geo_init)
    our_center_init_lat, our_center_init_lon, our_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(our_center_geo_init)


    #=====================================绘图==============================================
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 初始第1波
    ax.scatter(first_uav_init_lats, first_uav_init_lons, first_uav_init_alts, c='red', marker='x', label='First UAV Init', s=50)

    # 初始第2波
    ax.scatter(second_uav_init_lats, second_uav_inti_lons, second_uav_init_alts, c='blue', marker='o', label='Second UAV Init', s=50)

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
    print(f"Our Center Init: lat={our_center_init_lat:.6f}, lon={our_center_init_lon:.6f}, alt={our_center_init_alt:.1f}")






if __name__ == "__main__":
    #=============================初始化数据==================================
    data = dataset()
    print(f"测试数据调用:{data['basepoint']}和敌机 {data['enemy_approx']}")



    #============================创建局部坐标系================================
    A_lat, A_lon, A_alt = GeodeticConverter.decimal_dms_to_degrees(data['basepoint'])
    B_lat, B_lon, B_alt = GeodeticConverter.decimal_dms_to_degrees(data['enemy_approx'])
    converter = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)


    #===========================处理第2波次无人机===============================
    plot_positions_with_centers(data['first_num'],data['second_num'],converter)
   # sorted_second = sorted_y_points(second_geo, converter) #对第2波次无人机按y从小到大进行排序，得到sorted_second[0]就是末尾那架无人机



