import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, Point
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from geopy.point import Point as geopyPoint
import math
import GeodeticConverter


# 探测的不规则阵面，r为近似圆半径
def generate_irregular_polygon(radii):
    radii.append(radii[0])
    directions_deg = np.linspace(0, 360, 9)
    directions_rad = np.deg2rad(directions_deg)
    cs = CubicSpline(directions_rad, radii, bc_type='periodic')

    theta_dense = np.linspace(0, 2 * np.pi, 500)
    radius_dense = cs(theta_dense)
    x = radius_dense * np.cos(theta_dense)  # 将极坐标转为直角坐标
    y = radius_dense * np.sin(theta_dense)
    polygon = Polygon(zip(x, y))
    # return x, y, polygon, directions_deg[:-1], cs
    return polygon


# 单个侦察无人机探测面积：f光圈，distance距离，image_height、image_width目标在图像中的尺寸。 敌我距离10km，det尺寸800* 1200 m，f=300mm = 0.03m  image
# def signal_detection_size(f, distance, image_height, image_width):
#     det_height = distance * image_height / f
#     det_width = distance * image_width / f
#     return det_height, det_width

# 设置第一批次侦察无人机的位置：radii探测误差，enemy_approx_center敌机中心点，dete_dist,detection探测大小
# 返回值 points第一波次阵位, len(rects)第二波次无人机数量, rects第一波次探测范围, center第一波次中心无人机
def first_set(radii, detection):
    polygon = generate_irregular_polygon(radii)
    minx, miny, maxx, maxy = polygon.bounds
    polygon_center = polygon.centroid
    rects = []
    for y in np.arange(miny, maxy, detection[1]):
        x = minx - detection[0]
        while x < maxx:
            tmp_rect = box(x, y, x + detection[0], y + detection[1])
            intersection = polygon.intersection(tmp_rect)
            min_x, min_y, max_x, max_y = intersection.bounds
            if not intersection.is_empty:
                # 找到左边第一个探测点
                if x < min_x:
                    x = min_x
                    tmp_rect = box(x, y, x + detection[0], y + detection[1])
                rects.append(tmp_rect)
                if tmp_rect.contains(polygon_center):
                    center = Point(tmp_rect.centroid.x, tmp_rect.centroid.y)
            x += detection[0]
    first_points = []  # 侦察无人机位置
    for rect in rects:
        x_coords, y_coords = rect.exterior.coords.xy
        first_points.append((np.mean(x_coords), 0, np.mean(y_coords)))  # 默认y轴为0
    return first_points, len(rects), rects, center


# 计算尾后探测无人机数量：enemy_number敌群数量, detection尾后探测大小
def total_num(enemy_number, detection):
    enemy_interval = 100
    total = math.ceil(enemy_number * (enemy_interval ** 2) / (detection[0] ** 2))
    return total


# 第二波次阵位设置：first_center第一波次中心无人机，first_uavs第一波次无人机位置，first_num第一波次无人机数量，interval无人机的间距，total尾后探测无人机总数， t转弯时间（结合机动模型）, speed敌机速度
# 返回值：second_points第二波次阵位, second_uavs_num第二波次无人机数量。
def second_set(first_center, first_uavs, first_num, total, interval, t, speed):
    single_row = math.ceil(speed * t / interval)  # singel_row = speed * time/ interval
    second_uavs_num = total - first_num
    first_center_x = first_center.x
    first_center_z = first_center.y
    second_points = []  #侦察无人机位置，三维坐标x、y、z分别是第二波次无人机左右（机翼方向）的距离、前后距离、高
    i = 1

    center = np.array([first_center_x, 0, first_center_z])
    distances = np.linalg.norm(first_uavs - center, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_first_uavs = np.array(first_uavs)[sorted_indices]
    turn = int(np.ceil(second_uavs_num / single_row))

    for j in range(turn):
        for m in range(single_row):
            if i > second_uavs_num:
                break
            p = sorted_first_uavs[j]
            second_points.append([p[0], p[1] - (m + 1) * interval, p[2]])
            i += 1
    return second_points, second_uavs_num


# 阵位调整：basepoint我方位置，enemy_approx_center敌方位置，first_points第一波次阵位，second_points第二波次阵位，dist敌机位置变动矢量
def position_adjust(basepoint, enemy_approx_center, first_points, second_points, type, dist=[500, 0, 200]):
    if type == 1:
        print("Case 1 中心点重合，无需调整")
        return first_points, second_points
    if type == 2 or type == 3:
        dist = Converter(enemy_approx_center, basepoint, dist)
        dist_lat, dist_lon, dist_alt = GeodeticConverter.decimal_dms_to_degrees(dist)
        new_first_points = []
        for point in first_points:
            lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(point)
            lon -= dist_lon
            lat -= dist_lat
            alt -= dist_alt
            lat_deg, lat_min, lat_sec = GeodeticConverter.GeodeticToLocalConverter.decimal_degrees_to_dms(lat)
            lon_deg, lon_min, lon_sec = GeodeticConverter.GeodeticToLocalConverter.decimal_degrees_to_dms(lon)
            lat_dir = 'N' if lat >= 0 else 'S'
            lon_dir = 'W' if lon >= 0 else 'E'
            formatted_str = [f"{lat_deg}:{lat_min}:{lat_sec:.2f}{lat_dir}",
                             f"{lon_deg}:{lon_min}:{lon_sec:.2f}{lon_dir}", f"{alt:.2f}"]
            new_first_points.append(formatted_str)
        new_second_points = []
        for point in second_points:
            lat, lon, alt = GeodeticConverter.decimal_dms_to_degrees(point)
            lon -= dist_lon
            lat -= dist_lat
            alt -= dist_alt
            lat_deg, lat_min, lat_sec = GeodeticConverter.GeodeticToLocalConverter.decimal_degrees_to_dms(lat)
            lon_deg, lon_min, lon_sec = GeodeticConverter.GeodeticToLocalConverter.decimal_degrees_to_dms(lon)
            lat_dir = 'N' if lat >= 0 else 'S'
            lon_dir = 'W' if lon >= 0 else 'E'
            formatted_str = [f"{lat_deg}:{lat_min}:{lat_sec:.2f}{lat_dir}",
                             f"{lon_deg}:{lon_min}:{lon_sec:.2f}{lon_dir}", f"{alt:.2f}"]
            new_second_points.append(formatted_str)
        return new_first_points, new_second_points


def Converter(enemy, basepoint, uavs):
    A_lat, A_lon, A_alt = GeodeticConverter.decimal_dms_to_degrees(basepoint)
    B_lat, B_lon, B_alt = GeodeticConverter.decimal_dms_to_degrees(enemy)

    # 创建坐标系转换器
    converter = GeodeticConverter.GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)

    UavsStr = []
    for uav in uavs:
        result = converter.local_to_geodetic_dms(uav)
        UavsStr.append(result)

    return UavsStr


# 初始阵位设置：radii探测误差，min_detection远距离探测尺寸，clear_detection敌机清晰可见时探测尺寸，enemy_approx_center敌机中心点，enemy_number敌机数量，interval第一波次与第二波次的间距，singel_row为单列无人机数量
# first_num第一波次无人机数量,  first_uavs第一波次阵位, second_num第二波次无人机数量, second_uavs第二波次阵位, rects小视场的覆盖矩阵, first_center第一波次无人机中心
def init_position_set(basepoint, enemy_approx_center, radii, min_detection, clear_detection, enemy_number=350,
                      interval=500, t=5, speed=240):
    first_uavs, first_num, rects, first_center = first_set(radii, min_detection)
    total = total_num(enemy_number, clear_detection)
    second_uavs, second_num = second_set(first_center, first_uavs, first_num, total, interval, t, speed)
    print(first_num, total, second_num)
    first_str = Converter(enemy_approx_center, basepoint, first_uavs)
    second_str = Converter(enemy_approx_center, basepoint, second_uavs)
    return first_num, first_str, second_num, second_str, rects, first_center


# 发现敌群后的中心推算和我方光电探测侦察无人机集群阵位调整：center第一波次侦察无人机中心，first_rects第一波次探测范围，first_uavs_points第一波次阵位, second_uavs_points第二波次阵位，old_has_enemy（原）无人机是否出现敌机，new_has_enemy（新）无人机是否出现敌机，type中心点
# 返回值：first_uavs_points第一波次阵位, second_uavs_points第二波次阵位
# def adjust_enemy_centre(center, first_rects, first_uavs_points, second_uavs_points, old_has_enemy, new_has_enemy):
#     # old_rects 原出现敌机的范围，new_rects新出现敌机的范围
#     old_rects = []
#     new_rects = []
#     for i in range(len(first_rects)):
#         if old_has_enemy[i]:
#             old_rects.append(first_rects[i])
#         if new_has_enemy[i]:
#             new_rects.append(first_rects[i])
#     # 计算原出现敌机的中心点和新出现敌机的中心点,计算每个矩形框的中心点,将中心点转换为 NumPy 数组,计算中心点的平均值
#     old_enemy_center = Point(*np.mean(np.array([(rect.centroid.x, rect.centroid.y) for rect in old_rects]), axis=0))
#     new_enemy_center = Point(*np.mean(np.array([(rect.centroid.x, rect.centroid.y) for rect in new_rects]), axis=0))
#
#     #侦察无人机探测范围的中心点与敌机中心点的距离
#     dist1 = center.distance(old_enemy_center)
#     dist2 = center.distance(new_enemy_center)
#     diff = new_enemy_center - old_enemy_center
#     adjust_dist = np.array([diff.x, 0, diff.y])
#
#     if old_enemy_center.equals(new_enemy_center)  and len(old_has_enemy)==len(new_has_enemy) :
#         print("Case 1 中心点重合，无需调整")
#     if dist1 > dist2 and len(old_has_enemy) < len(new_has_enemy):
#         print ("Case2 敌我不重合，且中心点接近")
#         first_uavs_points, second_uavs_points = position_adjust(first_uavs_points, second_uavs_points, adjust_dist)
#     if dist1 < dist2 and len(old_has_enemy) > len(new_has_enemy):
#         print("Case3 敌我不重合，且中心点远离")
#         first_uavs_points, second_uavs_points = position_adjust(first_uavs_points, second_uavs_points, adjust_dist)
#     return first_uavs_points, second_uavs_points
#

if __name__ == "__main__":
    r = 2500   # 不规则近似圆阵面的半径
    radi = [2500, 2502, 2504, 2501, 2502, 2501, 2500, 2503]
    enemy_approx = '28:11:34.95W 00:30:35.24N 55.0'
    basepoint = '28:13:34.95W 00:30:31.24N 50.0'
    min_detect = [900, 600]  # 远距离探测尺寸，根据距离不同会有不同,单位 m
    clear_detect = [240, 160]  # 敌机清晰可见时探测尺寸
    enemy_number = 350  # 敌机数量
    interval = 500  #侦察无人机间距
    t = 5  # 侦察无人机转弯时间5s
    speed = 240  # 敌机飞行速度
    type = 2  #敌群中心改变情况

    first_num, first_uavs, second_num, second_uavs, optimal_rects, first_center = init_position_set(basepoint,
                                                                                                    radi, min_detect,
                                                                                                    clear_detect,
                                                                                                    enemy_approx,
                                                                                                    enemy_number,
                                                                                                    interval, t, speed)
    # second_uavs = np.array(second_uavs)
    # new_first_uavs, new_second_uavs = position_adjust(first_uavs, second_uavs, type)

    # old_has_enemy_rects = np.array([True] * 10 + [True] * 60)  # 原探测到敌机的无人机序列
    # new_has_enemy_rects = np.array([True] * 2 + [False] * 50)  # 新探测到敌机的无人机序列
    # new_first_uavs, new_second_uavs = adjust_enemy_centre(first_center, optimal_rects, first_uavs, second_uavs, old_has_enemy_rects, new_has_enemy_rects)
    #
    # # 阵位展示
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for rect in optimal_rects:
    #     x_r, y_r = rect.exterior.xy
    #     vertices = list(zip(x_r, [enemy_approx[1]]*len(x_r) , y_r ))  # 添加 z 坐标
    #     ax.add_collection3d(Poly3DCollection([vertices], facecolors='grey', edgecolors='black', alpha=0.5))
    #
    # # 第一波次
    # for p in first_uavs:
    #     ax.scatter(p[0],p[1], p[2], color='skyblue', s=10)
    # # 第二波次
    # for p in second_uavs:
    #     ax.scatter(p[0],p[1], p[2], color='pink', s=10)
    # # 新第一波次
    # for p in new_first_uavs:
    #     ax.scatter(p[0], p[1], p[2], color='blue', s=20)
    # # 新第二波次
    # for p in new_second_uavs:
    #     ax.scatter(p[0],p[1], p[2], color='red', s=20)
    # # 设置坐标轴标签
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # # 设置标题
    # ax.set_title('Position Setting')
    # plt.show()