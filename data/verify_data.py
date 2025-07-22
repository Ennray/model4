from dataset import dataset

def verify_data():
    """验证导入的数据格式"""
    data = dataset()
    
    print("=== 完整数据验证报告 ===")
    print(f"基准点坐标: {data['basepoint']}")
    print(f"最小检测范围: {data['min_detect']}")
    print(f"清晰检测范围: {data['clear_detect']}")
    print(f"最大速度: {data['maximum_speed']} m/s")
    print(f"最小速度: {data['minimum_speed']} m/s")
    print(f"当前速度: {data['speed']} m/s")
    print(f"敌方近似位置: {data['enemy_approx']}")
    print(f"半径数组: {data['radii']}")
    print(f"敌方数量: {data['enemy_number']}")
    print(f"敌方经度范围: {data['enemy_lonrange']}")
    print(f"敌方纬度范围: {data['enemy_latrange']}")
    
    print(f"\n=== 无人机数据验证 ===")
    print(f"第一批无人机数量: {data['first_num']} (实际: {len(data['first_uavs'])})")
    print(f"第二批无人机数量: {data['second_num']} (实际: {len(data['second_uavs'])})")
    
    # 检查数据格式
    print(f"\n=== 坐标格式验证 ===")
    
    # 验证第一批数据
    if data['first_uavs']:
        sample = data['first_uavs'][0]
        print(f"第一批无人机样本: {sample}")
        print(f"纬度格式: {sample[0]} (度:分:秒.小数N/S)")
        print(f"经度格式: {sample[1]} (度:分:秒.小数E/W)")
        print(f"高度格式: {sample[2]} (米)")
    
    # 验证第二批数据  
    if data['second_uavs']:
        sample = data['second_uavs'][0]
        print(f"\n第二批无人机样本: {sample}")
        print(f"纬度格式: {sample[0]} (度:分:秒.小数N/S)")
        print(f"经度格式: {sample[1]} (度:分:秒.小数E/W)")
        print(f"高度格式: {sample[2]} (米)")
    
    # 验证坐标范围
    print(f"\n=== 坐标范围验证 ===")
    if data['first_uavs']:
        lats = [float(coord[0].split(':')[0]) for coord in data['first_uavs'] if 'N' in coord[0]]
        lons = [float(coord[1].split(':')[0]) for coord in data['first_uavs'] if 'E' in coord[1]]
        alts = [float(coord[2]) for coord in data['first_uavs']]
        
        print(f"第一批 - 纬度范围: {min(lats):.1f}° ~ {max(lats):.1f}°")
        print(f"第一批 - 经度范围: {min(lons):.1f}° ~ {max(lons):.1f}°")
        print(f"第一批 - 高度范围: {min(alts):.1f}m ~ {max(alts):.1f}m")
    
    if data['second_uavs']:
        lats = [float(coord[0].split(':')[0]) for coord in data['second_uavs'] if 'N' in coord[0]]
        lons = [float(coord[1].split(':')[0]) for coord in data['second_uavs'] if 'E' in coord[1]]
        alts = [float(coord[2]) for coord in data['second_uavs']]
        
        print(f"第二批 - 纬度范围: {min(lats):.1f}° ~ {max(lats):.1f}°")
        print(f"第二批 - 经度范围: {min(lons):.1f}° ~ {max(lons):.1f}°")
        print(f"第二批 - 高度范围: {min(alts):.1f}m ~ {max(alts):.1f}m")
    
    print(f"\n✅ 完整数据导入验证完成！")
    print(f"数据字典包含 {len(data)} 个字段")

if __name__ == "__main__":
    verify_data()
