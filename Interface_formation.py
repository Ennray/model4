from flask import Flask, request, jsonify
import json
import uuid
from Formation import generate_gradient_grid_enemy

app = Flask(__name__)

# 定义20个输入配置 - 更新为使用分开的间距参数
INPUT_CONFIGS = {
    
    # 敌机阵型
    1: {
        "id": 1,
        "name": "敌机阵型1-单层25x14",
        "center_dms": ["127:12:02.17E", "26:37:06.00N", "5000.0"],
        "total_planes": 350,
        "layers": 1,
        "rows_per_layer": 1,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": 600.0
    },
    2: {
        "id": 2,
        "name": "敌机阵型2-分层前低后高5x10x7",
        "center_dms": ["116:00:00.00E", "40:00:00.00N", "1000.0"],
        "total_planes": 350,
        "layers": 7,
        "rows_per_layer": 5,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": 600.0
    },
    3: {
        "id": 3,
        "name": "敌机阵型3-分层前高后低5x10x7",
        "center_dms": ["116:00:00.00E", "40:00:00.00N", "5000.0"],
        "total_planes": 350,
        "layers": 7,
        "rows_per_layer": 5,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": -600.0
    },
    4: {
        "id": 4,
        "name": "敌机阵型4-分层前低后高1x50x7",
        "center_dms": ["116:00:00.00E", "40:00:00.00N", "1000.0"],
        "total_planes": 350,
        "layers": 7,
        "rows_per_layer": 1,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": 600.0
    },
    5: {
        "id": 5,
        "name": "敌机阵型5-分层前高后低1x50x7",
        "center_dms": ["116:00:00.00E", "40:00:00.00N", "5000.0"],
        "total_planes": 350,
        "layers": 7,
        "rows_per_layer": 1,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": -600.0
    },
    # 我机阵型 - 对应敌机阵型1
    6: {
        "id": 6,
        "name": "我机阵型1.1-撞击",
        "center_dms": ["116:00:00.00E", "39:55:00.00N", "1000.0"],
        "total_planes": 350,
        "layers": 1,
        "rows_per_layer": 25,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": 600.0
    },
    7: {
        "id": 7,
        "name": "我机阵型1.2-干扰",
        "center_dms": ["116:00:00.00E", "39:56:00.00N", "1000.0"],
        "total_planes": 117,
        "layers": 1,
        "rows_per_layer": 13,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": 600.0
    },
    8: {
        "id": 8,
        "name": "我机阵型1.3-投弹",
        "center_dms": ["116:00:00.00E", "39:57:00.00N", "1000.0"],
        "total_planes": 175,
        "layers": 1,
        "rows_per_layer": 14,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": 600.0
    },
    # 我机阵型 - 对应敌机阵型2
    9: {
        "id": 9,
        "name": "我机阵型2.1-撞击",
        "center_dms": ["116:00:00.00E", "39:55:00.00N", "1000.0"],
        "total_planes": 350,
        "layers": 7,
        "rows_per_layer": 5,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": 600.0
    },
    10: {
        "id": 10,
        "name": "我机阵型2.2-干扰",
        "center_dms": ["116:00:00.00E", "39:56:00.00N", "1000.0"],
        "total_planes": 117,
        "layers": 4,
        "rows_per_layer": 5,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": 600.0
    },
    11: {
        "id": 11,
        "name": "我机阵型2.3-投弹",
        "center_dms": ["116:00:00.00E", "39:57:00.00N", "1000.0"],
        "total_planes": 175,
        "layers": 5,
        "rows_per_layer": 5,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": 600.0
    },
    # 我机阵型 - 对应敌机阵型3
    12: {
        "id": 12,
        "name": "我机阵型3.1-撞击",
        "center_dms": ["116:00:00.00E", "39:55:00.00N", "5000.0"],
        "total_planes": 350,
        "layers": 7,
        "rows_per_layer": 5,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": -600.0
    },
    13: {
        "id": 13,
        "name": "我机阵型3.2-干扰",
        "center_dms": ["116:00:00.00E", "39:56:00.00N", "5000.0"],
        "total_planes": 117,
        "layers": 4,
        "rows_per_layer": 5,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": -600.0
    },
    14: {
        "id": 14,
        "name": "我机阵型3.3-投弹",
        "center_dms": ["116:00:00.00E", "39:57:00.00N", "5000.0"],
        "total_planes": 175,
        "layers": 5,
        "rows_per_layer": 5,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": -600.0
    },
    # 我机阵型 - 对应敌机阵型4
    15: {
        "id": 15,
        "name": "我机阵型4.1-撞击",
        "center_dms": ["116:00:00.00E", "39:55:00.00N", "1000.0"],
        "total_planes": 350,
        "layers": 7,
        "rows_per_layer": 1,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": 600.0
    },
    16: {
        "id": 16,
        "name": "我机阵型4.2-干扰",
        "center_dms": ["116:00:00.00E", "39:56:00.00N", "1000.0"],
        "total_planes": 117,
        "layers": 4,
        "rows_per_layer": 1,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": 600.0
    },
    17: {
        "id": 17,
        "name": "我机阵型4.3-投弹",
        "center_dms": ["116:00:00.00E", "39:57:00.00N", "1000.0"],
        "total_planes": 175,
        "layers": 5,
        "rows_per_layer": 1,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": 600.0
    },
    # 我机阵型 - 对应敌机阵型5
    18: {
        "id": 18,
        "name": "我机阵型5.1-撞击",
        "center_dms": ["116:00:00.00E", "39:55:00.00N", "5000.0"],
        "total_planes": 350,
        "layers": 7,
        "rows_per_layer": 1,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": -600.0
    },
    19: {
        "id": 19,
        "name": "我机阵型5.2-干扰",
        "center_dms": ["116:00:00.00E", "39:56:00.00N", "5000.0"],
        "total_planes": 117,
        "layers": 4,
        "rows_per_layer": 1,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": -600.0
    },
    20: {
        "id": 20,
        "name": "我机阵型5.3-投弹",
        "center_dms": ["116:00:00.00E", "39:57:00.00N", "5000.0"],
        "total_planes": 175,
        "layers": 5,
        "rows_per_layer": 1,
        "lateral_spacing_m": 600,      # 左右间距
        "longitudinal_spacing_m": 600,  # 前后间距
        "layer_height_delta": -600.0
    }
}

# 默认配置
DEFAULT_CONFIG_ID = 1

def validate_custom_params(params):
    """
    验证自定义参数的有效性
    """
    required_fields = ['center_dms', 'total_planes', 'layers', 'rows_per_layer', 
                      'lateral_spacing_m', 'longitudinal_spacing_m', 'layer_height_delta']
    
    # 检查必需字段
    for field in required_fields:
        if field not in params:
            return False, f"缺少必需字段: {field}"
    
    # 验证数据类型
    try:
        if not isinstance(params['center_dms'], list) or len(params['center_dms']) != 3:
            return False, "center_dms 必须是包含3个元素的列表"
        
        if not isinstance(params['total_planes'], int) or params['total_planes'] <= 0:
            return False, "total_planes 必须是正整数"
        
        if not isinstance(params['layers'], int) or params['layers'] <= 0:
            return False, "layers 必须是正整数"
        
        if not isinstance(params['rows_per_layer'], int) or params['rows_per_layer'] <= 0:
            return False, "rows_per_layer 必须是正整数"
        
        # 验证左右间距
        if not isinstance(params['lateral_spacing_m'], (int, float)) or params['lateral_spacing_m'] <= 0:
            return False, "lateral_spacing_m 必须是正数"
        
        # 验证前后间距
        if not isinstance(params['longitudinal_spacing_m'], (int, float)) or params['longitudinal_spacing_m'] <= 0:
            return False, "longitudinal_spacing_m 必须是正数"
        
        if not isinstance(params['layer_height_delta'], (int, float)):
            return False, "layer_height_delta 必须是数字"
            
    except (TypeError, ValueError) as e:
        return False, f"参数类型错误: {str(e)}"
    
    return True, "参数验证通过"

@app.route('/api/uav/formation/generate', methods=['POST', 'GET'])
def generate_formation():
    try:
        use_custom_params = False
        custom_params = {}
        
        # 如果是POST请求，检查是否包含自定义参数
        if request.method == 'POST':
            raw_data = request.get_data(as_text=True)
            print(f"收到请求数据: {raw_data}")
            
            if raw_data.strip():
                try:
                    input_data = json.loads(raw_data)
                    
                    # 检查是否包含自定义参数（而不是config_id）
                    if all(key in input_data for key in ['center_dms', 'total_planes', 'layers', 'rows_per_layer', 'lateral_spacing_m', 'longitudinal_spacing_m', 'layer_height_delta']):
                        use_custom_params = True
                        custom_params = input_data
                        print("使用自定义参数生成阵型")
                    elif 'config_id' in input_data:
                        # 原有的config_id逻辑
                        config_id = input_data['config_id']
                        print(f"使用POST请求中的config_id: {config_id}")
                except json.JSONDecodeError:
                    print("JSON解析失败，使用URL参数或默认config_id")
        
        # 获取配置ID（GET请求或POST中没有自定义参数时使用）
        config_id = request.args.get('config_id', default=DEFAULT_CONFIG_ID, type=int)
        
        if use_custom_params:
            # 使用自定义参数
            is_valid, validation_msg = validate_custom_params(custom_params)
            if not is_valid:
                return jsonify({
                    "errorCode": "40001",
                    "errorMsg": f"参数验证失败: {validation_msg}",
                    "requestId": str(uuid.uuid4())
                }), 400
            
            selected_config = {
                "name": "自定义阵型",
                "center_dms": custom_params["center_dms"],
                "total_planes": custom_params["total_planes"],
                "layers": custom_params["layers"],
                "rows_per_layer": custom_params["rows_per_layer"],
                "lateral_spacing_m": custom_params["lateral_spacing_m"],
                "longitudinal_spacing_m": custom_params["longitudinal_spacing_m"],
                "layer_height_delta": custom_params["layer_height_delta"]
            }
            
            config_source = "custom"
            
        else:
            # 使用预定义配置
            if config_id in INPUT_CONFIGS:
                selected_config = INPUT_CONFIGS[config_id]
                print(f"使用配置 {config_id}: {selected_config['name']}")
            else:
                print(f"配置ID {config_id} 不存在，使用默认配置 {DEFAULT_CONFIG_ID}")
                selected_config = INPUT_CONFIGS[DEFAULT_CONFIG_ID]
            
            config_source = f"predefined_{config_id}"
        
        # 生成阵型
        formation_data = generate_gradient_grid_enemy(
            center_dms=selected_config["center_dms"],
            total_planes=selected_config["total_planes"],
            layers=selected_config["layers"],
            rows_per_layer=selected_config["rows_per_layer"],
            lateral_spacing_m=selected_config["lateral_spacing_m"],
            longitudinal_spacing_m=selected_config["longitudinal_spacing_m"],
            layer_height_delta=selected_config["layer_height_delta"]
        )

        # 返回结果
        response = {
            "code": 200,
            "Form": formation_data,
            "config_used": {
                "config_source": config_source,
                "config_name": selected_config["name"],
                "parameters": selected_config
            }
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({
            "errorCode": "50001",
            "errorMsg": f"服务器错误: {str(e)}",
            "requestId": str(uuid.uuid4())
        }), 500

@app.route('/api/uav/formation/custom', methods=['POST'])
def generate_custom_formation():
    """
    专门用于自定义参数生成的接口
    """
    try:
        raw_data = request.get_data(as_text=True)
        print(f"收到自定义参数请求: {raw_data}")
        
        if not raw_data.strip():
            return jsonify({
                "errorCode": "40002",
                "errorMsg": "请求体不能为空",
                "requestId": str(uuid.uuid4())
            }), 400
        
        try:
            input_data = json.loads(raw_data)
        except json.JSONDecodeError as e:
            return jsonify({
                "errorCode": "40003",
                "errorMsg": f"JSON解析错误: {str(e)}",
                "requestId": str(uuid.uuid4())
            }), 400
        
        # 验证参数
        is_valid, validation_msg = validate_custom_params(input_data)
        if not is_valid:
            return jsonify({
                "errorCode": "40001",
                "errorMsg": f"参数验证失败: {validation_msg}",
                "requestId": str(uuid.uuid4())
            }), 400
        
        # 生成阵型
        formation_data = generate_gradient_grid_enemy(
            center_dms=input_data["center_dms"],
            total_planes=input_data["total_planes"],
            layers=input_data["layers"],
            rows_per_layer=input_data["rows_per_layer"],
            lateral_spacing_m=input_data["lateral_spacing_m"],
            longitudinal_spacing_m=input_data["longitudinal_spacing_m"],
            layer_height_delta=input_data["layer_height_delta"]
        )

        # 返回结果
        response = {
            "code": 200,
            "Form": formation_data,
            "config_used": {
                "config_source": "custom",
                "config_name": "自定义阵型",
                "parameters": input_data
            }
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({
            "errorCode": "50001",
            "errorMsg": f"服务器错误: {str(e)}",
            "requestId": str(uuid.uuid4())
        }), 500

@app.route('/api/uav/formation/configs', methods=['GET'])
def list_configs():
    """列出所有可用配置"""
    config_list = {}
    for config_id, config in INPUT_CONFIGS.items():
        config_list[config_id] = {
            "name": config["name"],
            "parameters": {k: v for k, v in config.items() if k != "name"}
        }
    
    return jsonify({
        "code": 200,
        "configs": config_list,
        "total": len(config_list)
    })

@app.route('/api/uav/formation/params-template', methods=['GET'])
def get_params_template():
    """
    获取自定义参数模板
    """
    template = {
        "center_dms": ["经度DMS", "纬度DMS", "高度(米)"],
        "total_planes": "飞机总数",
        "layers": "层数",
        "rows_per_layer": "每层行数", 
        "lateral_spacing_m": "左右间距(米)",
        "longitudinal_spacing_m": "前后间距(米)",
        "layer_height_delta": "层高差(米)"
    }
    
    example = {
        "center_dms": ["116:00:00.00E", "40:00:00.00N", "1000.0"],
        "total_planes": 100,
        "layers": 3,
        "rows_per_layer": 5,
        "lateral_spacing_m": 500.0,      # 左右间距500米
        "longitudinal_spacing_m": 800.0,  # 前后间距800米
        "layer_height_delta": 300.0
    }
    
    return jsonify({
        "code": 200,
        "template": template,
        "example": example
    })

@app.route('/')
def index():
    """显示使用说明"""
    config_options = "\n".join([f"<li>{config_id}: {config['name']}</li>" for config_id, config in INPUT_CONFIGS.items()])
    
    return f"""
    <h1>无人机阵型生成API</h1>
    <p>支持20种预定义阵型配置和自定义参数</p>
    
    <h2>可用预定义配置：</h2>
    <ul>
        {config_options}
    </ul>
    
    <h2>使用方法：</h2>
    <ul>
        <li>GET请求指定配置：<a href="/api/uav/formation/generate?config_id=1">/api/uav/formation/generate?config_id=1</a></li>
        <li>POST请求自定义参数：/api/uav/formation/generate 或 /api/uav/formation/custom</li>
        <li>查看所有配置：<a href="/api/uav/formation/configs">/api/uav/formation/configs</a></li>
        <li>获取参数模板：<a href="/api/uav/formation/params-template">/api/uav/formation/params-template</a></li>
    </ul>
    
    <h2>示例命令：</h2>
    <pre>
# 使用预定义配置
curl "http://localhost:5000/api/uav/formation/generate?config_id=1"

# 使用自定义参数（通用接口）
curl -X POST -H "Content-Type: application/json" -d '{{
  "center_dms": ["116:00:00.00E", "40:00:00.00N", "1000.0"],
  "total_planes": 100,
  "layers": 3,
  "rows_per_layer": 5,
  "lateral_spacing_m": 500,
  "longitudinal_spacing_m": 800,
  "layer_height_delta": 300
}}' http://localhost:5000/api/uav/formation/generate

# 使用自定义参数（专用接口）
curl -X POST -H "Content-Type: application/json" -d '{{
  "center_dms": ["116:00:00.00E", "40:00:00.00N", "1000.0"],
  "total_planes": 50,
  "layers": 2,
  "rows_per_layer": 5,
  "lateral_spacing_m": 400,
  "longitudinal_spacing_m": 600,
  "layer_height_delta": 200
}}' http://localhost:5000/api/uav/formation/custom

# 查看所有配置
curl "http://localhost:5000/api/uav/formation/configs"

# 获取参数模板
curl "http://localhost:5000/api/uav/formation/params-template"
    </pre>
    """

if __name__ == '__main__':
    print("启动无人机阵型生成API服务...")
    print(f"可用预定义配置数量: {len(INPUT_CONFIGS)}")
    print("接口地址:")
    print("  - 阵型生成: http://localhost:5000/api/uav/formation/generate")
    print("  - 自定义生成: http://localhost:5000/api/uav/formation/custom") 
    print("  - 配置列表: http://localhost:5000/api/uav/formation/configs")
    print("  - 参数模板: http://localhost:5000/api/uav/formation/params-template")
    print("  - 使用说明: http://localhost:5000/")
    app.run(port=5000, debug=True)