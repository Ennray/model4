#
# Formation.py  -- 梯度网格敌机阵型（层间恒定高度差，每层等量）
import math
from typing import List
from GeodeticConverter import GeodeticToLocalConverter

def dms_to_decimal(dms_str):
    """
    将度分秒字符串转换为十进制度数
    格式示例: "116:00:00.00E" 或 "40:00:00.00N"
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

def decimal_to_dms(decimal_degrees, is_latitude=True):
    """
    将十进制度数转换为度分秒字符串
    :param decimal_degrees: 十进制度数
    :param is_latitude: 是否为纬度（True为纬度，False为经度）
    :return: 度分秒字符串
    """
    # 确定方向
    if is_latitude:
        direction = 'N' if decimal_degrees >= 0 else 'S'
    else:
        direction = 'E' if decimal_degrees >= 0 else 'W'
    
    # 取绝对值计算
    decimal_degrees = abs(decimal_degrees)
    degrees = int(decimal_degrees)
    minutes_float = (decimal_degrees - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    
    return f"{degrees}:{minutes:02d}:{seconds:05.2f}{direction}"

def generate_gradient_grid_enemy(
        center_dms: List[str],
        total_planes: int,
        layers: int,
        rows_per_layer: int,
        lateral_spacing_m: float,    # 左右间距（横向）
        longitudinal_spacing_m: float, # 前后间距（纵向）
        layer_height_delta: float = 30.0   # 层间恒定高度差（米）
) -> List[List[str]]:
    """
    每层行数固定，列数按剩余飞机动态计算 -> 前面每层数量相同，仅末层可能不足
    层间高度差恒定（前低后高）
    返回: 敌机坐标列表，每个坐标为 [经度DMS, 纬度DMS, 高度] 的字符串类型
    
    参数说明:
    - lateral_spacing_m: 左右间距（米）- 控制横向（X轴）飞机间隔
    - longitudinal_spacing_m: 前后间距（米）- 控制纵向（Y轴）飞机间隔
    """
    # 将中心点从度分秒转换为十进制度
    lon_center = dms_to_decimal(center_dms[0])
    lat_center = dms_to_decimal(center_dms[1])
    alt_center = float(center_dms[2])
    
    converter = GeodeticToLocalConverter(lat_center, lon_center, alt_center,
                                         lat_center + 1e-4, lon_center, alt_center)

    out_coords = []  # 存储字符串类型的坐标 [lon_dms, lat_dms, alt]
    remaining = total_planes

    # 如果是分层阵型（layers > 1），确保层间在垂直投影上没有重叠
    if layers > 1:
        # 计算每层应该分配的飞机数量（平均分配）
        planes_per_layer = total_planes // layers
        extra_planes = total_planes % layers

        for lyr in range(layers):
            if remaining <= 0:
                break
                
            # 本层实际飞机数量
            layer_planes = planes_per_layer
            if lyr < extra_planes:
                layer_planes += 1
                
            if layer_planes <= 0:
                continue

            # 恒定高度差：每层 +layer_height_delta
            z_layer = lyr * layer_height_delta
            
            # 关键修改：每一层都比前一层更靠后，确保垂直投影无重叠
            # 第0层从Y=0开始，第1层从rows_per_layer * longitudinal_spacing_m开始，以此类推
            layer_y_base = lyr * rows_per_layer * longitudinal_spacing_m

            # 动态列数 = 本层飞机数量 / 行数
            cols = math.ceil(layer_planes / rows_per_layer)
            cols = max(cols, 1)

            for row in range(rows_per_layer):
                if layer_planes <= 0:
                    break
                    
                # 计算本行在前后方向的位置（相对于本层基准）
                y_row = layer_y_base + row * longitudinal_spacing_m
                
                for col in range(cols):
                    if layer_planes <= 0:
                        break
                        
                    # 计算本列在左右方向的位置 - 使用 lateral_spacing_m
                    x_col = - (cols - 1) / 2 * lateral_spacing_m + col * lateral_spacing_m

                    # 使用 local_to_geodetic 获取经纬度，高度直接计算
                    lat, lon, _ = converter.local_to_geodetic([x_col, y_row, 0])
                    alt_final = alt_center + z_layer  # 直接计算高度，避免精度损失
                    
                    # 转换为度分秒格式
                    lon_dms = decimal_to_dms(lon, is_latitude=False)
                    lat_dms = decimal_to_dms(lat, is_latitude=True)
                    
                    # 存储字符串类型的坐标
                    out_coords.append([
                        lon_dms,        # 经度 DMS
                        lat_dms,        # 纬度 DMS  
                        f"{alt_final:.2f}"  # 高度字符串
                    ])
                    layer_planes -= 1
                    remaining -= 1

            if remaining <= 0:
                break
    else:
        # 单层阵型保持原有逻辑，但使用不同的间距参数
        for lyr in range(layers):
            if remaining <= 0:
                break
            # 恒定高度差：每层 +layer_height_delta
            z_layer = lyr * layer_height_delta
            # 水平中心：前→后 - 使用 longitudinal_spacing_m
            y_center = (layers - 1) / 2 * longitudinal_spacing_m - lyr * longitudinal_spacing_m

            # 动态列数 = 剩余飞机 / 本层剩余行块
            cols = math.ceil(remaining / (rows_per_layer * (layers - lyr)))
            cols = max(cols, 1)

            for row in range(rows_per_layer):
                if remaining <= 0:
                    break
                y_row = y_center - (row - (rows_per_layer - 1) / 2) * longitudinal_spacing_m
                for col in range(cols):
                    if remaining <= 0:
                        break
                    # 使用 lateral_spacing_m 计算左右位置
                    x_col = - (cols - 1) / 2 * lateral_spacing_m + col * lateral_spacing_m

                    # 使用 local_to_geodetic 获取经纬度，高度直接计算
                    lat, lon, _ = converter.local_to_geodetic([x_col, y_row, 0])
                    alt_final = alt_center + z_layer  # 直接计算高度，避免精度损失
                    
                    # 转换为度分秒格式
                    lon_dms = decimal_to_dms(lon, is_latitude=False)
                    lat_dms = decimal_to_dms(lat, is_latitude=True)
                    
                    # 存储字符串类型的坐标
                    out_coords.append([
                        lon_dms,        # 经度 DMS
                        lat_dms,        # 纬度 DMS  
                        f"{alt_final:.2f}"  # 高度字符串
                    ])
                    remaining -= 1

            if remaining <= 0:
                break

    return out_coords


def api_main(config: dict) -> dict:
    """
    统一接口入口函数，用于 Flask API 调用

    参数:
        config: 配置字典，包含以下字段：
            - center_dms: 中心点坐标 [经度DMS, 纬度DMS, 高度]
            - total_planes: 飞机总数
            - layers: 层数
            - rows_per_layer: 每层行数
            - lateral_spacing_m: 左右间距（米）
            - longitudinal_spacing_m: 前后间距（米）
            - layer_height_delta: 层高差（米）

    返回:
        包含生成结果的字典
    """
    # 从 config 中提取参数
    center_dms = config.get("center_dms")
    total_planes = config.get("total_planes")
    layers = config.get("layers")
    rows_per_layer = config.get("rows_per_layer")
    lateral_spacing_m = config.get("lateral_spacing_m")
    longitudinal_spacing_m = config.get("longitudinal_spacing_m")
    layer_height_delta = config.get("layer_height_delta", 30.0)  # 默认值30米

    # 参数验证
    if not center_dms or not total_planes or not layers or not rows_per_layer:
        raise ValueError("缺少必要参数: center_dms, total_planes, layers, rows_per_layer")

    if not lateral_spacing_m or not longitudinal_spacing_m:
        raise ValueError("缺少必要参数: lateral_spacing_m, longitudinal_spacing_m")

    # 调用核心生成函数
    formation_data = generate_gradient_grid_enemy(
        center_dms=center_dms,
        total_planes=total_planes,
        layers=layers,
        rows_per_layer=rows_per_layer,
        lateral_spacing_m=lateral_spacing_m,
        longitudinal_spacing_m=longitudinal_spacing_m,
        layer_height_delta=layer_height_delta
    )

    # 返回结果
    return {
        "code": 200,
        "Form": formation_data,
        "total_generated": len(formation_data),
        "config_used": {
            "center_dms": center_dms,
            "total_planes": total_planes,
            "layers": layers,
            "rows_per_layer": rows_per_layer,
            "lateral_spacing_m": lateral_spacing_m,
            "longitudinal_spacing_m": longitudinal_spacing_m,
            "layer_height_delta": layer_height_delta
        }
    }


# 测试函数
if __name__ == "__main__":
    # 测试生成敌机阵型 - 使用不同的左右和前后间距
    print("测试敌机阵型生成（不同间距）...")
    
    # 测试单平面6x6阵型，左右间距500米，前后间距800米
    enemy_planes = generate_gradient_grid_enemy(
        center_dms=["116:00:00.00E", "40:00:00.00N", "1000.0"],
        total_planes=36,
        layers=1,
        rows_per_layer=6,
        lateral_spacing_m=500,      # 左右间距500米
        longitudinal_spacing_m=800, # 前后间距800米
        layer_height_delta=600.0
    )
    
    print(f"成功生成 {len(enemy_planes)} 架飞机")
    print("前5架飞机坐标:")
    for i, coord in enumerate(enemy_planes[:5], 1):
        lon, lat, alt = coord
        print(f"飞机{i}: 经度={lon}, 纬度={lat}, 高度={alt}")