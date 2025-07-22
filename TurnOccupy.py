import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
import velocity_recong

from numpy.ma.core import remainder

import GeodeticConverter
import outdata
import data.dataset as dataset

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
def plot_positions_with_centers(enemy_positions_geo_init, uav_positions_geo_init,
                                enemy_center_geo_init, our_center_geo_init,
                                converter):


    # 初始敌机点
    enemy_init_lats, enemy_init_lons, enemy_init_alts = geo_to_degrees(enemy_positions_geo_init)

    # 初始我方点
    uav_init_lats, uav_init_lons, uav_init_alts = geo_to_degrees(uav_positions_geo_init)

    # 初始中心（已知）
    enemy_center_init_lat, enemy_center_init_lon, enemy_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center_geo_init)
    our_center_init_lat, our_center_init_lon, our_center_init_alt = GeodeticConverter.decimal_dms_to_degrees(our_center_geo_init)


    # ----------------------------------
    # 绘图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 初始敌机
    ax.scatter(enemy_init_lons, enemy_init_lats, enemy_init_alts, c='red', marker='x', label='Enemy Init', s=50)

    # 初始我方
    ax.scatter(uav_init_lons, uav_init_lats, uav_init_alts, c='blue', marker='o', label='UAV Init', s=50)

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
    # enemy_geo, second_geo = dataset.dataset()
    second_uavs = [
        [
            "29:47:28.25N",
            "120:37:56.47E",
            "5095.53"
        ],
        [
            "29:47:28.49N",
            "120:37:37.90E",
            "5124.76"
        ],
        [
            "29:47:28.73N",
            "120:37:19.34E",
            "5154.04"
        ],
        [
            "29:47:28.96N",
            "120:37:0.77E",
            "5183.36"
        ],
        [
            "29:47:29.20N",
            "120:36:42.20E",
            "5212.71"
        ],
        [
            "29:47:29.43N",
            "120:36:23.63E",
            "5242.11"
        ],
        [
            "29:47:29.67N",
            "120:36:5.07E",
            "5271.54"
        ],
        [
            "29:47:29.90N",
            "120:35:46.50E",
            "5301.01"
        ],
        [
            "29:47:30.13N",
            "120:35:27.93E",
            "5330.52"
        ],
        [
            "29:47:30.37N",
            "120:35:9.37E",
            "5360.07"
        ],
        [
            "29:47:30.60N",
            "120:34:50.80E",
            "5389.66"
        ],
        [
            "29:47:30.83N",
            "120:34:32.23E",
            "5419.29"
        ],
        [
            "29:47:31.06N",
            "120:34:13.67E",
            "5448.96"
        ],
        [
            "29:47:31.29N",
            "120:33:55.10E",
            "5478.66"
        ],
        [
            "29:47:31.52N",
            "120:33:36.54E",
            "5508.41"
        ],
        [
            "29:47:31.74N",
            "120:33:17.97E",
            "5538.19"
        ],
        [
            "29:47:31.97N",
            "120:32:59.40E",
            "5568.01"
        ],
        [
            "29:47:32.20N",
            "120:32:40.84E",
            "5597.88"
        ],
        [
            "29:47:32.42N",
            "120:32:22.27E",
            "5627.78"
        ],
        [
            "29:47:32.65N",
            "120:32:3.71E",
            "5657.72"
        ],
        [
            "29:47:32.87N",
            "120:31:45.14E",
            "5687.69"
        ],
        [
            "29:47:33.09N",
            "120:31:26.58E",
            "5717.71"
        ],
        [
            "29:47:33.32N",
            "120:31:8.01E",
            "5747.77"
        ],
        [
            "29:47:33.54N",
            "120:30:49.45E",
            "5777.87"
        ],
        [
            "29:47:33.76N",
            "120:30:30.88E",
            "5808.00"
        ],
        [
            "29:47:33.98N",
            "120:30:12.32E",
            "5838.17"
        ],
        [
            "29:47:34.20N",
            "120:29:53.75E",
            "5868.39"
        ],
        [
            "29:47:34.42N",
            "120:29:35.19E",
            "5898.64"
        ],
        [
            "29:47:34.64N",
            "120:29:16.63E",
            "5928.93"
        ],
        [
            "29:47:34.86N",
            "120:28:58.06E",
            "5959.26"
        ]
    ]
    base = ["29:47:22.84N", "120:38:14.82E", "5005.0"]
    enemy_center = ["29:28:00.11N", "128:19:36.77E","5000.0"]


    #============================创建局部坐标系================================
    A_lat, A_lon, A_alt = GeodeticConverter.decimal_dms_to_degrees(base)
    B_lat, B_lon, B_alt = GeodeticConverter.decimal_dms_to_degrees(enemy_center)
    converter = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)


    #===========================处理第2波次无人机===============================
    sorted_second = sorted_y_points(second_geo, converter) #对第2波次无人机按y从小到大进行排序，得到sorted_second[0]就是末尾那架无人机



