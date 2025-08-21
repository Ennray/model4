import json
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import numpy as np

import TurnOccupy as TO  # 确保与 TurnOccupy.py 在同一目录或已入 PYTHONPATH

app = Flask(__name__)

# --- 把 numpy / 元组 等不可序列化对象转成可 JSON 的 ---
def to_jsonable(obj):
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
    return obj  # 默认原样

@app.route('/execute_main', methods=['GET'])
def execute_main():
    try:
        # 允许 body 为空；如果有就拿来覆盖默认 dataset
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise BadRequest("JSON body 必须是对象（dict）。")

        config = payload.get("config") if "config" in payload else payload
        if config is not None and not isinstance(config, dict):
            raise BadRequest("config 必须是对象（dict）。")

        # 调用 TurnOccupy 的入口（与旧 main 兼容）
        result = TO.main(config)

        # 兜底：把结果转成可 JSON 的
        result_jsonable = to_jsonable(result)

        return jsonify({"status": "ok", "data": result_jsonable}), 200

    except Exception as e:
        # 直接把错误信息返回，便于联调
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    # 如需跨网访问，加 host='0.0.0.0'
    app.run(port=5000, debug=True)
