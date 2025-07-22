import json

def load_afsim_data():
    """从afsim.txt加载数据"""
    with open('afsim.txt', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def update_dataset():
    """使用afsim.txt的数据更新dataset.py"""
    
    # 加载afsim数据
    afsim_data = load_afsim_data()
    
    # 获取first_uavs和second_uavs数据
    first_uavs = afsim_data.get('first_uavs', [])
    second_uavs = afsim_data.get('second_uavs', [])
    
    # 生成新的dataset.py内容
    dataset_content = '''def dataset():
    enemy_geo = [
'''
    
    # 添加first_uavs数据到enemy_geo
    for i, uav in enumerate(first_uavs):
        lat, lon, alt = uav[0], uav[1], uav[2]
        if i == len(first_uavs) - 1:
            dataset_content += f'        [\n            "{lat}",\n            "{lon}",\n            "{alt}"\n        ]\n'
        else:
            dataset_content += f'        [\n            "{lat}",\n            "{lon}",\n            "{alt}"\n        ],\n'
    
    dataset_content += '''
    ]

    second_uav = [
'''
    
    # 添加second_uavs数据到second_uav
    for i, uav in enumerate(second_uavs):
        lat, lon, alt = uav[0], uav[1], uav[2]
        if i == len(second_uavs) - 1:
            dataset_content += f'        [\n            "{lat}",\n            "{lon}",\n            "{alt}"\n        ]\n'
        else:
            dataset_content += f'        [\n            "{lat}",\n            "{lon}",\n            "{alt}"\n        ],\n'
    
    dataset_content += '''
    ]

    return enemy_geo, second_uav
'''
    
    # 写入新的dataset.py文件
    with open('dataset.py', 'w', encoding='utf-8') as f:
        f.write(dataset_content)
    
    print(f"数据导入完成！")
    print(f"第一批无人机数量: {len(first_uavs)}")
    print(f"第二批无人机数量: {len(second_uavs)}")

if __name__ == "__main__":
    update_dataset()
