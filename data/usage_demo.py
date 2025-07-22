from dataset import dataset

def demo_usage():
    """展示如何使用dataset函数获取完整数据"""
    
    # 获取所有数据
    data = dataset()
    
    print("=== Dataset 使用示例 ===\n")
    
    # 访问基础配置
    print("🔧 基础配置:")
    print(f"  基准点: {data['basepoint']}")
    print(f"  速度范围: {data['minimum_speed']}-{data['maximum_speed']} m/s")
    print(f"  当前速度: {data['speed']} m/s")
    print(f"  检测范围: 最小{data['min_detect']}, 清晰{data['clear_detect']}")
    
    # 访问敌方信息
    print(f"\n🎯 敌方信息:")
    print(f"  敌方位置: {data['enemy_approx']}")
    print(f"  敌方数量: {data['enemy_number']}")
    print(f"  经度范围: {data['enemy_lonrange']}")
    print(f"  纬度范围: {data['enemy_latrange']}")
    print(f"  半径数组: {data['radii']}")
    
    # 访问无人机数据
    print(f"\n🚁 无人机数据:")
    print(f"  第一批无人机: {data['first_num']}架")
    print(f"  第二批无人机: {data['second_num']}架")
    
    # 示例：获取特定无人机的坐标
    print(f"\n📍 坐标示例:")
    if data['first_uavs']:
        uav1 = data['first_uavs'][0]
        print(f"  第1架无人机位置: 经度{uav1[0]}, 纬度{uav1[1]}, 高度{uav1[2]}m")
    
    if data['second_uavs']:
        uav2 = data['second_uavs'][0]  
        print(f"  第2批第1架位置: 经度{uav2[0]}, 纬度{uav2[1]}, 高度{uav2[2]}m")
    
    # 示例：批量处理坐标
    print(f"\n📊 数据统计:")
    all_first_alts = [float(uav[2]) for uav in data['first_uavs']]
    all_second_alts = [float(uav[2]) for uav in data['second_uavs']]
    
    print(f"  第一批平均高度: {sum(all_first_alts)/len(all_first_alts):.1f}m")
    print(f"  第二批平均高度: {sum(all_second_alts)/len(all_second_alts):.1f}m")
    
    print(f"\n✨ 使用方法:")
    print(f"  from dataset import dataset")
    print(f"  data = dataset()")
    print(f"  # 然后通过 data['字段名'] 访问所需数据")
    print(f"  # 可用字段: {list(data.keys())}")

if __name__ == "__main__":
    demo_usage()
