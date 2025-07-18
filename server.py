import json

def save_uav_multi_positions(
    uavs_speed,
    enemy_speed,
    our_init_positions,
    enemy_init_positions,
    our_pre_turn_positions,
    our_post_turn_positions,
    filename="uav_full_path_record.json"
):
    data = {
        "uavs_speed": {
            "current": uavs_speed[0],
            "acceleration": uavs_speed[1],
            "deceleration": uavs_speed[2],
            "max_speed": uavs_speed[3],
            "turn_speed": uavs_speed[4]
        },
        "enemy_speed": enemy_speed,
        "our_initial_positions": our_init_positions,
        "enemy_initial_positions": enemy_init_positions,
        "our_pre_turn_positions": our_pre_turn_positions,
        "our_post_turn_positions": our_post_turn_positions
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



