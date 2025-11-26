import json
from typing import Any, Dict

import numpy as np
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

# 导入五个算法模块
import Impact
import Interference
import Dropbombs
import Formation
import Turnning

app = Flask(__name__)

# JSON 配置：不排序 key，允许中文
app.json.sort_keys = False
app.json.ensure_ascii = False


def to_jsonable(obj: Any):
    """
    把 numpy 等不可 JSON 直接序列化的对象，转成可 JSON 的类型。
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


# 存当前配置的全局变量
CURRENT_CONFIG: Dict[str, Any] | None = None

# 算法映射字典
ALGORITHM_MODULES = {
    "impact": Impact,
    "interference": Interference,
    "dropbombs": Dropbombs,
    "formation": Formation,
    "turning": Turnning,
}


@app.route("/", methods=["GET"])
def index():
    """
    首页，显示接口使用说明
    """
    return jsonify({
        "service": "统一算法接口",
        "algorithms": list(ALGORITHM_MODULES.keys()),
        "endpoints": {
            "/set_config": "POST - 设置全局配置（基本上用不上）",
            "/execute": "POST - 执行算法（全局入口）",
            "/impact": "POST - 执行撞击算法",
            "/interference": "POST - 执行干扰算法",
            "/dropbombs": "POST - 执行投弹算法",
            "/formation": "POST - 执行阵型生成算法",
            "/turning": "POST - 执行转弯算法",
        },
        "usage": {
            "method1": "在config中指定algorithm_type",
            "method2": "使用专用路由（/impact, /interference, /dropbombs, /formation, /turning）"
        }
    }), 200


@app.route("/set_config", methods=["POST"])
def set_config():
    """
    输入接口：
      - 只负责接收并保存配置，不做计算。
      - JSON 有两种写法：
          1）{"config": {...}}
          2）直接 {...}
    """
    global CURRENT_CONFIG
    try:
        payload = request.get_json(silent=False)
        if not isinstance(payload, dict):
            raise BadRequest("JSON body 必须是对象（dict）。")

        # 兼容两种写法：带 config 包一层，或者直接就是配置
        config = payload.get("config") if "config" in payload else payload
        if not isinstance(config, dict):
            raise BadRequest("config 必须是对象（dict）。")

        CURRENT_CONFIG = config

        return jsonify({
            "status": "ok",
            "message": "config 已更新",
            "algorithm_type": config.get("algorithm_type", "未指定"),
            "config_preview": {k: config[k] for k in list(config.keys())[:5]}  # 简单预览前几个 key
        }), 200

    except BadRequest as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"服务器内部错误: {e}"}), 500


@app.route("/execute", methods=["POST"])
def execute():
    """
    统一执行接口（推荐使用）：
      - 根据 config 中的 algorithm_type 字段选择算法
      - 支持的算法类型：impact, interference, dropbombs, formation, turning
      - 如果本次请求里带了 config，就用本次的；
      - 否则尝试使用之前 /set_config 存下来的 CURRENT_CONFIG。
    """
    global CURRENT_CONFIG
    try:
        payload = request.get_json(silent=True) or {}
        if payload and not isinstance(payload, dict):
            raise BadRequest("JSON body 必须是对象（dict）。")

        config = None
        if payload:
            config = payload.get("config") if "config" in payload else payload
            if config is not None and not isinstance(config, dict):
                raise BadRequest("config 必须是对象（dict）。")

        # 如果本次没提供 config，就使用之前 set_config 存的
        if config is None:
            if CURRENT_CONFIG is None:
                raise BadRequest("未提供 config，且尚未通过 /set_config 设置全局 config。")
            config = CURRENT_CONFIG

        # 获取算法类型
        algorithm_type = config.get("algorithm_type")
        if not algorithm_type:
            raise BadRequest("config 中缺少 algorithm_type 字段。支持的类型: impact, interference, dropbombs, formation, turning")

        # 选择对应的算法模块
        if algorithm_type not in ALGORITHM_MODULES:
            raise BadRequest(
                f"不支持的 algorithm_type: '{algorithm_type}'。"
                f"支持的类型: {list(ALGORITHM_MODULES.keys())}"
            )

        module = ALGORITHM_MODULES[algorithm_type]

        # 调用算法主函数
        print(f"[INFO] 执行算法: {algorithm_type}")
        result = module.api_main(config)

        # 转成 JSON 能用的类型
        result_jsonable = to_jsonable(result)

        return jsonify({
            "status": "ok",
            "algorithm": algorithm_type,
            "data": result_jsonable
        }), 200

    except BadRequest as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": f"服务器内部错误: {e}",
            "traceback": traceback.format_exc()
        }), 500


@app.route("/impact", methods=["POST"])
def execute_impact():
    """
    撞击算法专用接口（快捷方式）
    """
    return _execute_algorithm("impact")


@app.route("/interference", methods=["POST"])
def execute_interference():
    """
    干扰算法专用接口（快捷方式）
    """
    return _execute_algorithm("interference")


@app.route("/dropbombs", methods=["POST"])
def execute_drop_bombs():
    """
    投弹算法专用接口（快捷方式）
    """
    return _execute_algorithm("dropbombs")


@app.route("/formation", methods=["POST"])
def execute_formation():
    """
    阵型生成算法专用接口（快捷方式）
    """
    return _execute_algorithm("formation")


@app.route("/turning", methods=["POST"])
def execute_turning():
    """
    转弯算法专用接口（快捷方式）
    """
    return _execute_algorithm("turning")


def _execute_algorithm(algorithm_type: str):
    """
    内部函数：执行指定算法
    """
    global CURRENT_CONFIG
    try:
        payload = request.get_json(silent=True) or {}
        if payload and not isinstance(payload, dict):
            raise BadRequest("JSON body 必须是对象（dict）。")

        config = None
        if payload:
            config = payload.get("config") if "config" in payload else payload
            if config is not None and not isinstance(config, dict):
                raise BadRequest("config 必须是对象（dict）。")

        # 如果本次没提供 config，就使用之前 set_config 存的
        if config is None:
            if CURRENT_CONFIG is None:
                raise BadRequest("未提供 config，且尚未通过 /set_config 设置全局 config。")
            config = CURRENT_CONFIG

        # 选择对应的算法模块
        module = ALGORITHM_MODULES[algorithm_type]

        # 调用算法主函数
        print(f"[INFO] 执行算法: {algorithm_type}")
        result = module.api_main(config)

        # 转成 JSON 能用的类型
        result_jsonable = to_jsonable(result)

        return jsonify({
            "status": "ok",
            "algorithm": algorithm_type,
            "data": result_jsonable
        }), 200

    except BadRequest as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": f"服务器内部错误: {e}",
            "traceback": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    # 统一端口 5000
    print("=" * 60)
    print("支持的算法: impact, interference, dropbombs, formation, turning")
    print("访问 http://localhost:5000/ 查看接口说明")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
