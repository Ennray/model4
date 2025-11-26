import json
from typing import Any, Dict

import numpy as np
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

import Impact as TO  # 你的算法主模块

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
            "config_preview": {k: config[k] for k in list(config.keys())[:5]}  # 简单预览前几个 key
        }), 200

    except BadRequest as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"服务器内部错误: {e}"}), 500


@app.route("/impact_main", methods=["GET", "POST"])
def execute_main():
    """
    输出接口：
      - 如果本次请求里带了 config，就用本次的 config 调用 TO.main；
      - 否则尝试使用之前 /set_config 存下来的 CURRENT_CONFIG；
      - 返回 TO.main 的结果（自动转成可 JSON 类型）。
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

        # 调你的主函数
        # print("[DEBUG] execute_main 收到的 config:", config)
        result = TO.api_main(config)

        # 转成 JSON 能用的类型
        result_jsonable = to_jsonable(result)

        return jsonify({
            "status": "ok",
            "data": result_jsonable
        }), 200

    except BadRequest as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"服务器内部错误: {e}"}), 500


if __name__ == "__main__":
    # 视情况改 host/port
    app.run(host="0.0.0.0", port=5000, debug=True)
