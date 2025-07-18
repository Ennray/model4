

def save_uav_multi_positions(
    uavs_speed,
    enemy_speed,
    our_init_positions,
    enemy_init_positions,
    our_pre_turn_positions,
    our_post_turn_positions,
    filename="uav_full_path_record.txt"
):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"我方速度设置:\n当前速度：{uavs_speed[0]}   加速度：{uavs_speed[1]}   减速度：{uavs_speed[2]}  最大速度：{uavs_speed[3]}  转弯速度：{uavs_speed[4]}\n")
        f.write(f"敌方速度设置：{enemy_speed}\n\n\n")

        f.write("我方初始位置\n")
        for i in range(len(our_init_positions)):
            f.write("{:<16} {:<16} {:>8}\n".format(*our_init_positions[i]))
        f.write("\n\n")

        f.write("敌方初始位置\n")
        for i in range(len(enemy_init_positions)):
            f.write("{:<16} {:<16} {:>8}\n".format(*enemy_init_positions[i]))
        f.write("\n\n")

        f.write("我方转弯前位置\n")
        for i in range(len(our_pre_turn_positions)):
            f.write("{:<16} {:<16} {:>8}\n".format(*our_pre_turn_positions[i]))
        f.write("\n\n")

        f.write("我方转弯后位置\n")
        for i in range(len(our_post_turn_positions)):
            f.write("{:<16} {:<16} {:>8}\n".format(*our_post_turn_positions[i]))



