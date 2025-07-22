from dataset import dataset

def verify_data():
    """验证导入的数据格式"""
    enemy_geo, second_uav = dataset()
    
    print("=== 数据验证报告 ===")
    print(f"第一批无人机(enemy_geo)数量: {len(enemy_geo)}")
    print(f"第二批无人机(second_uav)数量: {len(second_uav)}")
    
    # 检查数据格式
    print("\n=== 数据格式验证 ===")
    
    # 验证第一批数据
    if enemy_geo:
        sample = enemy_geo[0]
        print(f"第一批无人机样本数据: {sample}")
        print(f"纬度格式: {sample[0]} (应为度:分:秒.小数N/S格式)")
        print(f"经度格式: {sample[1]} (应为度:分:秒.小数E/W格式)")
        print(f"高度格式: {sample[2]} (应为米)")
    
    # 验证第二批数据  
    if second_uav:
        sample = second_uav[0]
        print(f"\n第二批无人机样本数据: {sample}")
        print(f"纬度格式: {sample[0]} (应为度:分:秒.小数N/S格式)")
        print(f"经度格式: {sample[1]} (应为度:分:秒.小数E/W格式)")
        print(f"高度格式: {sample[2]} (应为米)")
    
    # 验证坐标范围
    print(f"\n=== 坐标范围验证 ===")
    if enemy_geo:
        lats = [float(coord[0].split(':')[0]) for coord in enemy_geo if 'N' in coord[0]]
        lons = [float(coord[1].split(':')[0]) for coord in enemy_geo if 'E' in coord[1]]
        alts = [float(coord[2]) for coord in enemy_geo]
        
        print(f"第一批 - 纬度范围: {min(lats):.1f}° ~ {max(lats):.1f}°")
        print(f"第一批 - 经度范围: {min(lons):.1f}° ~ {max(lons):.1f}°")
        print(f"第一批 - 高度范围: {min(alts):.1f}m ~ {max(alts):.1f}m")
    
    if second_uav:
        lats = [float(coord[0].split(':')[0]) for coord in second_uav if 'N' in coord[0]]
        lons = [float(coord[1].split(':')[0]) for coord in second_uav if 'E' in coord[1]]
        alts = [float(coord[2]) for coord in second_uav]
        
        print(f"第二批 - 纬度范围: {min(lats):.1f}° ~ {max(lats):.1f}°")
        print(f"第二批 - 经度范围: {min(lons):.1f}° ~ {max(lons):.1f}°")
        print(f"第二批 - 高度范围: {min(alts):.1f}m ~ {max(alts):.1f}m")
    
    print("\n✅ 数据导入验证完成！")

if __name__ == "__main__":
    verify_data()
