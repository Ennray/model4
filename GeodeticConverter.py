import numpy as np
from pyproj import CRS, Transformer
from geopy.point import Point as geopyPoint


class GeodeticToLocalConverter:
    """经纬度与局部坐标相互转换"""
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

        # y轴方向(AB方向)
        y_axis = enu / np.linalg.norm(enu)

        # z轴(垂直于y轴和天向)
        up = np.array([0, 0, 1])  # ENU的天向
        z_axis = np.cross(y_axis, up)
        if np.linalg.norm(z_axis) < 1e-10:
            # 处理AB与天向平行的情况
            z_axis = np.array([1, 0, 0])
        else:
            z_axis = z_axis / np.linalg.norm(z_axis)

        # x轴(y × z)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

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
            lon_dir = 'W'
        else:
            lon_deg = abs(lon_deg)
            lon_dir = 'E'
        formatted_str = [f"{lon_deg}:{lon_min}:{lon_sec:.2f}{lon_dir}", f"{lat_deg}:{lat_min}:{lat_sec:.2f}{lat_dir}",f"{alt:.2f}"]

        return formatted_str


def dms_to_decimal(dms_str):
    """
    将度分秒字符串转换为十进制度数
    格式示例: "28:13:34.95W" 或 "00:30:31.24N"
    规则: W为正，E为负；N为正，S为负
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
    if direction in ['W', 'N']:
        return decimal
    elif direction in ['E', 'S']:
        return -decimal
    else:
        raise ValueError(f"无效的方向标识: {direction}")

def decimal_dms_to_degrees(coord_str):
    """
    解析完整坐标字符串 '["28:13:34.95W", "00:30:31.24N", "50.0"]'
    返回 (纬度, 经度, 高度)
    """
    lat_dms = coord_str[1]
    lon_dms = coord_str[0]
    alt = coord_str[2]
    latitude = dms_to_decimal(lat_dms)  # N为正
    longitude = dms_to_decimal(lon_dms)  # W为正
    altitude = float(alt)

    return latitude, longitude,  altitude


# 使用示例
if __name__ == "__main__":
    # 定义A点和B点(经纬度高程)
    A_lat, A_lon, A_alt = 40.0, 116.0, 50.0  # 北京附近, 海拔50米
    B_lat, B_lon, B_alt = 40.01, 116.0, 55.0  # A点正北约1.1公里, 海拔55米

    # lat_deg, lat_min, lat_sec = .decimal_degrees_to_dms(40.01)

    # 创建坐标系转换器
    converter = GeodeticToLocalConverter(A_lat, A_lon, A_alt, B_lat, B_lon, B_alt)

    # 示例1: 将C点从局部坐标转换为经纬度高程
    C_local = np.array([100, 200, 10])  # 局部坐标系中的坐标(x=100m东, y=200m北, z=10m上)
    result = converter.local_to_geodetic_dms(C_local)
    result += result
    print(result)
    # print(result)
    # print(f"C点局部坐标: {C_local} 米")
    # print(f"C点经纬度高程: ({C_lat:.6f}°, {C_lon:.6f}°, {C_alt:.2f}米)")
    #
    # # 示例2: 将经纬度高程转换为局部坐标
    # P_lat, P_lon, P_alt = 40.005, 116.003, 52.0
    # P_local = converter.geodetic_to_local(P_lat, P_lon, P_alt)
    # print(f"\nP点经纬度高程: ({P_lat}°, {P_lon}°, {P_alt}米)")
    # print(f"P点局部坐标: {P_local} 米")