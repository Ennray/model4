import numpy as np
from pyproj import CRS, Transformer
from geopy.distance import great_circle
import math


class GeodeticToLocalConverter:
    """经纬度与局部坐标相互转换 - 已改进支持球面距离计算"""
    def __init__(self, A_lat, A_lon, A_alt, B_lat, B_lon, B_alt):
        """
        初始化坐标系转换器
        :param A_lat: A点纬度(度)
        :param A_lon: A点经度(度)
        :param A_alt: A点高度(m)
        :param B_lat: B点纬度(度)
        :param B_lon: B点经度(度)
        :param B_alt: B点高度(m)
        """
        self.A_lat, self.A_lon, self.A_alt = A_lat, A_lon, A_alt
        self.B_lat, self.B_lon, self.B_alt = B_lat, B_lon, B_alt

        # 定义WGS84坐标系
        self.wgs84 = CRS.from_epsg(4979)  # WGS84 3D
        self.enu = CRS.from_epsg(4979)  # 临时ENU坐标系

        # 计算ECEF坐标
        self.A_ecef = self.latlonalt_to_ecef(A_lat, A_lon, A_alt)
        self.B_ecef = self.latlonalt_to_ecef(B_lat, B_lon, B_alt)

        # 构建ENU到局部坐标系的转换矩阵
        self.rotation_matrix = self._build_rotation_matrix()

    def latlonalt_to_ecef(self, lat, lon, alt):
        """将经纬度高程转换为ECEF坐标"""
        transformer = Transformer.from_crs(CRS.from_epsg(4979), CRS.from_epsg(4978))
        return np.array(transformer.transform(lat, lon, alt))

    def ecef_to_latlonalt(self, x, y, z):
        """将ECEF坐标转换为经纬度高程"""
        transformer = Transformer.from_crs(CRS.from_epsg(4978), CRS.from_epsg(4979))
        return transformer.transform(x, y, z)

    def _build_rotation_matrix(self):
        """构建ENU到局部坐标系的旋转矩阵(AB为y轴)"""
        # 计算B在ENU坐标系中的坐标
        delta = self.B_ecef - self.A_ecef
        enu = self._ecef_to_enu(delta)

        x_axis = enu / np.linalg.norm(enu)

        # z 轴：Up（竖直）
        z_axis = np.array([0, 0, 1], dtype=float)

        # 退化处理：若 AB 近乎竖直，选一个水平 x 轴兜底
        if abs(np.dot(x_axis, z_axis)) > 1 - 1e-10:
            x_axis = np.array([1, 0, 0], dtype=float)

        # y 轴：z × x，保证右手系且 y 在水平面
        y_axis = np.cross(z_axis, x_axis);
        y_axis /= np.linalg.norm(y_axis)

        # 重新正交化 x（可选，抹去极小数值误差）
        x_axis = np.cross(y_axis, z_axis)

        # 旋转矩阵(ENU -> 局部坐标系)
        return np.vstack([x_axis, y_axis, z_axis]).T

    def _ecef_to_enu(self, delta_ecef):
        """ECEF坐标差转ENU坐标"""
        lat, lon = np.radians(self.A_lat), np.radians(self.A_lon)

        # ENU旋转矩阵
        R = np.array([
            [-np.sin(lon), np.cos(lon), 0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        ])

        return R @ delta_ecef

    def _enu_to_ecef(self, enu):
        """ENU坐标转ECEF坐标差"""
        lat, lon = np.radians(self.A_lat), np.radians(self.A_lon)

        # ENU旋转矩阵的逆
        R_inv = np.array([
            [-np.sin(lon), -np.sin(lat) * np.cos(lon), np.cos(lat) * np.cos(lon)],
            [np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat) * np.sin(lon)],
            [0, np.cos(lat), np.sin(lat)]
        ])

        return R_inv @ enu

    def local_to_geodetic(self, local_coords):
        """
        将局部坐标转换为经纬度高度
        :param local_coords: 局部坐标系中的坐标[x,y,z](m)
        :return: (lat, lon, alt) 纬度(度), 经度(度), 高度(m)
        """
        # 局部坐标转ENU
        enu = self.rotation_matrix @ local_coords

        # ENU转ECEF坐标差
        delta_ecef = self._enu_to_ecef(enu)

        # 计算C点的ECEF坐标
        C_ecef = self.A_ecef + delta_ecef

        # ECEF转经纬度高程
        return self.ecef_to_latlonalt(*C_ecef)

    def geodetic_to_local(self, lat, lon, alt):
        """
        将经纬度高程转换为局部坐标
        :param lat: 纬度(度)
        :param lon: 经度(度)
        :param alt: 高度(m)
        :return: 局部坐标系中的坐标[x,y,z](m)
        """
        # 转换为ECEF坐标
        P_ecef = self.latlonalt_to_ecef(lat, lon, alt)

        # 计算相对于A的ECEF坐标差
        delta_ecef = P_ecef - self.A_ecef

        # ECEF转ENU
        enu = self._ecef_to_enu(delta_ecef)

        # ENU转局部坐标
        return self.rotation_matrix.T @ enu

    def decimal_degrees_to_dms(self, decimal_degrees):
        """
        将十进制度数转换为度分秒表示
        :param decimal_degrees: 十进制度数值（如40.001798）
        :return: 元组（度, 分, 秒）和格式化字符串
        """
        degrees = int(decimal_degrees)
        remainder = abs(decimal_degrees - degrees) * 60
        minutes = int(remainder)
        seconds = (remainder - minutes) * 60

        return degrees, minutes, seconds

    def local_to_geodetic_dms(self, local_coords):
        """
        将局部坐标转换为度分秒表示的经纬度高程
        :param local_coords: 局部坐标系中的坐标[x,y,z](m)
        :return: 字典 {
            'lat_dms': 纬度度分秒,
            'lon_dms': 经度度分秒,
            'alt': 高度,
            'formatted': 格式化字符串
        }
        """
        lat, lon, alt = self.local_to_geodetic(local_coords)

        # 转换纬度
        lat_deg, lat_min, lat_sec = self.decimal_degrees_to_dms(lat)

        # 转换经度
        lon_deg, lon_min, lon_sec = self.decimal_degrees_to_dms(lon)

        # 确定南北纬/东西经
        # lat_dir = 'N' if lat >= 0 else 'S'
        if lat >= 0:
            lat_dir = 'N'
        else:
            lat_deg = abs(lat_deg)
            lat_dir = 'S'
        if lon >= 0:
            lon_dir = 'E'
        else:
            lon_deg = abs(lon_deg)
            lon_dir = 'W'
        formatted_str = [f"{lon_deg}:{lon_min}:{lon_sec:.2f}{lon_dir}", f"{lat_deg}:{lat_min}:{lat_sec:.2f}{lat_dir}",f"{alt:.2f}"]

        return formatted_str

    def calculate_spherical_distance(self, lat1, lon1, alt1, lat2, lon2, alt2):
        """
        计算两点间的球面距离（大圆距离）
        这是飞机实际应该飞行的距离，考虑地球曲率

        :param lat1, lon1, alt1: 点1的纬度、经度、高度
        :param lat2, lon2, alt2: 点2的纬度、经度、高度
        :return: 球面距离(米)
        """
        # 地表大圆距离
        ground_distance = great_circle((lat1, lon1), (lat2, lon2)).meters

        # 高度差
        altitude_diff = alt2 - alt1

        # 3D距离 (考虑高度的实际飞行距离)
        actual_distance = math.sqrt(ground_distance**2 + altitude_diff**2)

        return actual_distance

    def calculate_climb_angle(self, alt1, alt2, ground_distance_m):
        """
        根据两点高度差与地面距离，计算爬升/下降角（仰角）

        :param alt1: 起点高度（米）
        :param alt2: 终点高度（米）
        :param ground_distance_m: 水平地面距离（米）
        :return: 爬升角/下降角（度），正值为上升，负值为下降
        """
        if ground_distance_m == 0:
            raise ValueError("地面距离不能为 0，无法计算仰角")

        delta_alt = alt2 - alt1  # 高度差
        angle_rad = math.atan2(delta_alt, ground_distance_m)  # 注意使用 atan2，考虑正负方向
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def calculate_flight_bearing(self, lat1, lon1, lat2, lon2):
        """
        计算真航向角（从点1到点2的方位角）

        :return: 航向角(度，0-360°，正北为0°，顺时针)
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)

        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)

        bearing_rad = math.atan2(y, x)
        bearing_deg = (math.degrees(bearing_rad) + 360) % 360

        return bearing_deg

    def generate_flight_waypoints(self, start_lat, start_lon, start_alt,
                                 end_lat, end_lon, end_alt, num_points=10):
        """
        生成大圆航线上的航路点
        这些点构成了飞机应该实际飞行的路径

        :param num_points: 航路点数量
        :return: 航路点列表 [(lat, lon, alt), ...]
        """
        waypoints = []

        for i in range(num_points + 1):
            fraction = i / num_points

            if fraction == 0:
                lat, lon, alt = start_lat, start_lon, start_alt
            elif fraction == 1:
                lat, lon, alt = end_lat, end_lon, end_alt
            else:
                # 大圆插值计算中间点
                lat, lon = self._interpolate_great_circle(
                    start_lat, start_lon, end_lat, end_lon, fraction
                )
                # 线性插值高度
                alt = start_alt + (end_alt - start_alt) * fraction

            waypoints.append((lat, lon, alt))

        return waypoints

    def _interpolate_great_circle(self, lat1, lon1, lat2, lon2, fraction):
        """
        在大圆上进行插值计算中间点
        """
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # 计算两点间的角距离
        d = math.acos(
            math.sin(lat1_rad) * math.sin(lat2_rad) +
            math.cos(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)
        )

        if d == 0:  # 同一点
            return lat1, lon1

        a = math.sin((1 - fraction) * d) / math.sin(d)
        b = math.sin(fraction * d) / math.sin(d)

        x = a * math.cos(lat1_rad) * math.cos(lon1_rad) + \
            b * math.cos(lat2_rad) * math.cos(lon2_rad)
        y = a * math.cos(lat1_rad) * math.sin(lon1_rad) + \
            b * math.cos(lat2_rad) * math.sin(lon2_rad)
        z = a * math.sin(lat1_rad) + b * math.sin(lat2_rad)

        lat_rad = math.atan2(z, math.sqrt(x**2 + y**2))
        lon_rad = math.atan2(y, x)

        return math.degrees(lat_rad), math.degrees(lon_rad)

    def compare_distances(self, lat1, lon1, alt1, lat2, lon2, alt2):
        """
        比较直线距离和球面距离的差异

        :return: 包含两种距离计算结果的字典
        """
        # 使用原来的局部坐标系方法计算直线距离
        local_coords_1 = self.geodetic_to_local(lat1, lon1, alt1)
        local_coords_2 = self.geodetic_to_local(lat2, lon2, alt2)
        linear_distance = np.linalg.norm(local_coords_2 - local_coords_1)

        # 使用球面方法计算实际飞行距离
        spherical_distance = self.calculate_spherical_distance(
            lat1, lon1, alt1, lat2, lon2, alt2
        )

        # 计算差异
        difference = spherical_distance - linear_distance
        difference_percent = (difference / spherical_distance) * 100

        return {
            'linear_distance_m': linear_distance,
            'spherical_distance_m': spherical_distance,
            'difference_m': difference,
            'difference_percent': difference_percent,
            'flight_bearing_deg': self.calculate_flight_bearing(lat1, lon1, lat2, lon2)
        }

    def calculate_destination_point(self, start_lat, start_lon, start_alt, 
                                   bearing_deg, distance_m, altitude_change_m=0):
        """
        根据起点、移动方向和距离计算终点的经纬度高度
        使用 geopy.distance.great_circle 的 destination 方法
        
        :param start_lat: 起点纬度(度)
        :param start_lon: 起点经度(度) 
        :param start_alt: 起点高度(米)
        :param bearing_deg: 移动方向(度，0-360°，正北为0°，顺时针)
        :param distance_m: 移动距离(米，地表距离)
        :param altitude_change_m: 高度变化(米，正值为上升，负值为下降)
        :return: (终点纬度, 终点经度, 终点高度)
        """
        # 使用 geopy 的 great_circle 计算目标点
        # 创建距离对象（以米为单位）
        distance_obj = great_circle(meters=distance_m)
        
        # 计算目标点坐标
        destination_point = distance_obj.destination(
            point=(start_lat, start_lon), 
            bearing=bearing_deg
        )
        
        # 提取经纬度
        end_lat = destination_point.latitude
        end_lon = destination_point.longitude
        
        # 计算终点高度
        end_alt = start_alt + altitude_change_m
        
        return end_lat, end_lon, end_alt

    def calculate_destination_with_climb_angle(self, start_lat, start_lon, start_alt,
                                             bearing_deg, ground_distance_m, climb_angle_deg=0):
        """
        根据起点、方向、地面距离和爬升角计算终点坐标
        考虑飞机的爬升/下降轨迹
        
        :param start_lat: 起点纬度(度)
        :param start_lon: 起点经度(度)
        :param start_alt: 起点高度(米)
        :param bearing_deg: 移动方向(度，0-360°，正北为0°，顺时针)
        :param ground_distance_m: 地面投影距离(米)
        :param climb_angle_deg: 爬升角(度，正值为爬升，负值为下降)
        :return: (终点纬度, 终点经度, 终点高度, 实际飞行距离)
        """
        # 计算高度变化
        climb_angle_rad = math.radians(climb_angle_deg)
        altitude_change = ground_distance_m * math.tan(climb_angle_rad)
        
        # 计算实际飞行距离(3D距离)
        actual_flight_distance = ground_distance_m / math.cos(climb_angle_rad)
        
        # 使用地面距离计算终点的经纬度
        end_lat, end_lon, _ = self.calculate_destination_point(
            start_lat, start_lon, start_alt, bearing_deg, ground_distance_m, 0
        )
        
        # 计算终点高度
        end_alt = start_alt + altitude_change
        
        return end_lat, end_lon, end_alt, actual_flight_distance

    # ...existing code...
def dms_to_decimal(dms_str):
    """
    将度分秒字符串转换为十进制度数
    格式示例: "28:13:34.95W" 或 "00:30:31.24N"
    规则: W为负，E为正；N为正，S为负
    """
    # 分离方向字符
    direction = dms_str[-1]
    value_str = dms_str[:-1]

    # 分割度分秒
    parts = value_str.split(':')
    degrees = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])

    # 计算十进制度数
    decimal = degrees + minutes / 60 + seconds / 3600

    # 根据方向确定正负
    if direction in ['E', 'N']:
        return decimal
    elif direction in ['W', 'S']:
        return -decimal
    else:
        raise ValueError(f"无效的方向标识: {direction}")

def decimal_dms_to_degrees(coord_str):
    """
    解析完整坐标字符串 '["28:13:34.95W", "00:30:31.24N", "50.0"]'
    返回 (纬度, 经度, 高度)
    """
    lat_dms = coord_str[1] #纬度
    lon_dms = coord_str[0] #经度
    alt = coord_str[2]
    latitude = dms_to_decimal(lat_dms)  # N为正
    longitude = dms_to_decimal(lon_dms)  # W为正
    altitude = float(alt)

    return latitude, longitude,  altitude


# 使用示例
if __name__ == "__main__":
    print("=== 坐标系转换和飞行距离计算示例 ===")
    print()
    
    # 定义A点和B点(经纬度高程)
    A_lat, A_lon, A_alt = 40.0, 116.0, 50.0  # 北京附近, 海拔50米
    B_lat, B_lon, B_alt = 40.01, 116.0, 55.0  # A点正北约1.1公里, 海拔55米

    # 创建坐标系转换器
    converter = GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)

    print("1. 局部坐标转换为经纬度高程 (度分秒格式)")
    C_local = np.array([100, 200, 10])
    result = converter.local_to_geodetic_dms(C_local)
    print(f"C点局部坐标: {C_local} 米")
    print(f"C点经纬度高程: {result}")
    print()

    print("2. 比较直线距离与球面距离")
    # 北京到上海的示例
    beijing_lat, beijing_lon, beijing_alt = 39.9042, 116.4074, 100
    shanghai_lat, shanghai_lon, shanghai_alt = 31.2304, 121.4737, 50
    
    distance_comparison = converter.compare_distances(
        beijing_lat, beijing_lon, beijing_alt,
        shanghai_lat, shanghai_lon, shanghai_alt
    )
    
    print(f"起点: 北京 ({beijing_lat}°N, {beijing_lon}°E, {beijing_alt}m)")
    print(f"终点: 上海 ({shanghai_lat}°N, {shanghai_lon}°E, {shanghai_alt}m)")
    print(f"")
    print(f"直线距离 (局部坐标系): {distance_comparison['linear_distance_m']/1000:.2f} km")
    print(f"球面距离 (实际飞行): {distance_comparison['spherical_distance_m']/1000:.2f} km")
    print(f"距离差异: {distance_comparison['difference_m']/1000:.2f} km ({distance_comparison['difference_percent']:.2f}%)")
    print(f"飞行航向: {distance_comparison['flight_bearing_deg']:.1f}°")
    print()
    
    print("3. 生成飞行航路点")
    waypoints = converter.generate_flight_waypoints(
        beijing_lat, beijing_lon, beijing_alt,
        shanghai_lat, shanghai_lon, shanghai_alt,
        num_points=5
    )
    
    print("北京到上海的航路点:")
    for i, (lat, lon, alt) in enumerate(waypoints):
        if i == 0:
            print(f"  起点: {lat:.4f}°N, {lon:.4f}°E, {alt:.1f}m")
        elif i == len(waypoints)-1:
            print(f"  终点: {lat:.4f}°N, {lon:.4f}°E, {alt:.1f}m")
        else:
            print(f"  航路点{i}: {lat:.4f}°N, {lon:.4f}°E, {alt:.1f}m")
    
    print()
    print("4. 根据方向和距离计算目标点")
    start_lat, start_lon, start_alt = 39.9042, 116.4074, 1000  # 北京，海拔1000米
    
    # 示例1: 向正北飞行100公里，高度不变
    bearing = 0  # 正北
    distance = 100000  # 100公里
    end_lat, end_lon, end_alt = converter.calculate_destination_point(
        start_lat, start_lon, start_alt, bearing, distance
    )
    print(f"起点: {start_lat}°N, {start_lon}°E, {start_alt}m")
    print(f"向北飞行100km后:")
    print(f"终点: {end_lat:.4f}°N, {end_lon:.4f}°E, {end_alt}m")
    
    # 验证距离
    actual_distance = converter.calculate_spherical_distance(
        start_lat, start_lon, start_alt, end_lat, end_lon, end_alt
    )
    print(f"验证距离: {actual_distance/1000:.2f} km")
    print()
    
    # 示例2: 向东南方向飞行，带爬升角
    bearing = 135  # 东南方向
    ground_distance = 50000  # 地面距离50公里
    climb_angle = 5  # 爬升角5度
    
    end_lat2, end_lon2, end_alt2, flight_distance = converter.calculate_destination_with_climb_angle(
        start_lat, start_lon, start_alt, bearing, ground_distance, climb_angle
    )
    print(f"向东南方向飞行，带5°爬升角:")
    print(f"地面距离: {ground_distance/1000:.1f} km")
    print(f"实际飞行距离: {flight_distance/1000:.2f} km")
    print(f"终点: {end_lat2:.4f}°N, {end_lon2:.4f}°E, {end_alt2:.1f}m")
    print(f"高度变化: +{end_alt2-start_alt:.1f}m")
    
    print()
    print("=== 重要说明 ===")
    print("1. 直线距离: 假设地球是平面，计算两点间的直线距离")
    print("2. 球面距离: 考虑地球曲率，计算大圆距离，这是飞机实际应该飞行的路径")
    print("3. 对于长距离飞行，两者差异会很大，应该使用球面距离进行导航计算")
    print("4. calculate_destination_point: 根据方向和距离计算目标点坐标")
    print("5. calculate_destination_with_climb_angle: 考虑爬升/下降角度的飞行轨迹计算")