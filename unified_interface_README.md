# 统一算法接口使用文档

## 概述

`unified_interface.py` 是一个统一的Flask接口，整合了五个算法模块：
- **impact**（撞击算法）- Impact
- **interference**（干扰算法）- Interference
- **dropbombs**（投弹算法）- Dropbombs
- **formation**（阵型生成算法）- Formation
- **turning**（转弯算法）- Turnning

## 启动服务

```bash
python unified_interface.py
```

服务将运行在：`http://localhost:5000`

---

## 接口列表

### 1. 首页（查看接口说明）
- **URL**: `GET http://localhost:5000/`
- **功能**: 返回接口使用说明和支持的算法列表

---

### 2. 设置全局配置（可选）
- **URL**: `POST http://localhost:5000/set_config`
- **功能**: 预先设置配置，后续调用可省略config参数

**请求示例**：
```json
{
  "algorithm_type": "interference",
  "enemy_dms": [...],
  "uav_center": [lat, lon, alt],
  ...
}
```

**响应**：
```json
{
  "status": "ok",
  "message": "config 已更新",
  "algorithm_type": "interference"
}
```

---

### 3. 统一执行接口（推荐）⭐

- **URL**: `POST http://localhost:5000/execute`
- **功能**: 根据 `algorithm_type` 自动选择算法

**请求示例（方法1：在config中指定）**：
```json
{
  "algorithm_type": "interference",
  "enemy_dms": [...],
  "uav_center": [lat, lon, alt],
  "enemy_speed": 240,
  "bearing": 270,
  "interference_mode": "limit",
  "max_c_points": 117,
  "coverage_neighbors": 25,
  ...
}
```

**请求示例（方法2：包装在config中）**：
```json
{
  "config": {
    "algorithm_type": "impact",
    "enemy_dms": [...],
    ...
  }
}
```

**响应**：
```json
{
  "status": "ok",
  "algorithm": "interference",
  "data": {
    "C_points": [[lon, lat, alt], ...],
    "C_indices": [0, 5, 10, ...],
    "mode": "limit",
    "c_count": 117
  }
}
```

---

### 4. 专用算法接口（快捷方式）

#### 4.1 撞击算法
- **URL**: `POST http://localhost:5000/impact`
- **功能**: 直接调用撞击算法（无需指定algorithm_type）

#### 4.2 干扰算法
- **URL**: `POST http://localhost:5000/interference`
- **功能**: 直接调用干扰算法（无需指定algorithm_type）

#### 4.3 投弹算法
- **URL**: `POST http://localhost:5000/dropbombs`
- **功能**: 直接调用投弹算法（无需指定algorithm_type）

#### 4.4 阵型生成算法
- **URL**: `POST http://localhost:5000/formation`
- **功能**: 直接调用阵型生成算法（无需指定algorithm_type）

#### 4.5 转弯算法
- **URL**: `POST http://localhost:5000/turning`
- **功能**: 直接调用转弯算法（无需指定algorithm_type）

**请求示例**（无需algorithm_type）：
```json
{
  "enemy_dms": [...],
  "uav_center": [lat, lon, alt],
  ...
}
```

---

## Postman使用示例

### 示例1：使用统一接口（推荐）

**步骤**：
1. 创建新请求：`POST http://localhost:5000/execute`
2. Headers: `Content-Type: application/json`
3. Body (raw JSON):
```json
{
  "algorithm_type": "interference",
  "enemy_dms": [[116.123, 39.456, 1000], ...],
  "uav_center": [116.0, 39.0, 1000],
  "enemy_speed": 240,
  "bearing": 270,
  "time_to_interference": 10,
  "uav_start_speed": 200,
  "uav_max_speed": 500,
  "acc_max": 100,
  "high_distance": 50,
  "interference_mode": "limit",
  "max_c_points": 70,
  "coverage_neighbors": 30
}
```

4. **切换算法**：只需修改 `algorithm_type` 为 `"impact"`, `"interference"`, `"dropbombs"`, `"formation"`, 或 `"turning"`

---

### 示例2：使用专用接口

**撞击算法**：
```
POST http://localhost:5000/impact
Body: { 撞击算法的config参数 }
```

**干扰算法**：
```
POST http://localhost:5000/interference
Body: { 干扰算法的config参数 }
```

**投弹算法**：
```
POST http://localhost:5000/dropbombs
Body: { 投弹算法的config参数 }
```

**阵型生成算法**：
```
POST http://localhost:5000/formation
Body: { 阵型生成算法的config参数 }
```

**转弯算法**：
```
POST http://localhost:5000/turning
Body: { 转弯算法的config参数 }
```

---

### 示例3：使用全局配置

**步骤1**：设置配置
```
POST http://localhost:5000/set_config
Body: { 完整config }
```

**步骤2**：执行算法（无需再传config）
```
POST http://localhost:5000/execute
Body: {}  (空body即可)
```

---

## 完整测试用例

### 测试1：干扰算法（1:5覆盖比）

```json
POST http://localhost:5000/execute

{
  "algorithm_type": "interference",
  "enemy_dms": [
    ["116°30'45.123\"E", "39°45'30.456\"N", "1000"],
    ["116°30'46.123\"E", "39°45'31.456\"N", "1000"],
    ...  // 350架敌机
  ],
  "uav_center": ["116°30'00.000\"E", "39°45'00.000\"N", "1000"],
  "enemy_speed": 240,
  "bearing": 270,
  "time_to_interference": 10,
  "uav_start_speed": 200,
  "uav_max_speed": 500,
  "acc_max": 100,
  "high_distance": 50,
  "max_c_points": 70,
  "coverage_neighbors": 30,
  "custom_radius": 600
}
```

**预期结果**：
- C点数量：70
- 覆盖率：95-100%
- 覆盖比：1:5

---

### 测试2：撞击算法

```json
POST http://localhost:5000/impact

{
  "enemy_dms": [...],
  "uav_center": [...],
  "enemy_speed": 240,
  "bearing": 270,
  ...撞击算法特有参数
}
```

---

### 测试3：投弹算法

```json
POST http://localhost:5000/dropbombs

{
  "enemy_dms": [...],
  "uav_center": [...],
  ...投弹算法特有参数
}
```

---

### 测试4：阵型生成算法

```json
POST http://localhost:5000/formation

{
  "algorithm_type": "formation",
  "center_dms": ["116:00:00.00E", "40:00:00.00N", "1000.0"],
  "total_planes": 100,
  "layers": 3,
  "rows_per_layer": 5,
  "lateral_spacing_m": 500,
  "longitudinal_spacing_m": 800,
  "layer_height_delta": 300
}
```

---

### 测试5：转弯算法

```json
POST http://localhost:5000/turning

{
  "algorithm_type": "turning",
  ...转弯算法特有参数
}
```

---

## 错误处理

### 常见错误1：缺少algorithm_type
```json
{
  "status": "error",
  "message": "config 中缺少 algorithm_type 字段"
}
```
**解决**：在config中添加 `"algorithm_type": "impact/interference/dropbombs/formation/turning"`

### 常见错误2：不支持的算法类型
```json
{
  "status": "error",
  "message": "不支持的 algorithm_type: 'xxx'"
}
```
**解决**：使用正确的算法类型

### 常见错误3：未提供config
```json
{
  "status": "error",
  "message": "未提供 config，且尚未通过 /set_config 设置全局 config"
}
```
**解决**：在请求body中提供config，或先调用 `/set_config`

---

## 对比：统一接口 vs 分散接口

| 特性 | 统一接口 | 分散接口 |
|------|---------|---------|
| URL数量 | 1个 (`/execute`) | 5个 |
| 切换算法 | 修改 `algorithm_type` | 修改URL |
| Postman管理 | 1个请求 | 5个请求 |
| 推荐场景 | 频繁切换算法 | 固定使用某个算法 |

---

## 最佳实践

1. **推荐使用 `/execute` 接口**：一个URL完成所有算法调用
2. **在config中指定algorithm_type**：清晰明确
3. **保存Postman Collection**：快速切换不同算法
4. **使用专用接口做快捷测试**：如只测干扰算法时用 `/interference`

---

## 迁移指南（从旧接口迁移）

### 旧方式（五个不同的接口）：
```
POST http://localhost:5000/...  (Interface_impact)
POST http://localhost:5000/...  (Interface_interference)
POST http://localhost:5000/...  (Interface_dropbombs)
POST http://localhost:5000/...  (Interface_formation)
POST http://localhost:6000/...  (Interface_turning)
```

### 新方式（统一接口）：
```
POST http://localhost:5000/execute
Body: { "algorithm_type": "impact/interference/dropbombs/formation/turning", ... }
```

**或使用快捷接口**：
```
POST http://localhost:5000/impact
POST http://localhost:5000/interference
POST http://localhost:5000/dropbombs
POST http://localhost:5000/formation
POST http://localhost:5000/turning
```

---

## 技术细节

- **Flask版本**：任意
- **端口**：5000（可在代码中修改）
- **支持方法**：POST, GET（仅首页）
- **Content-Type**：application/json
- **编码**：UTF-8
- **并发**：单线程（debug模式）

---

## 常见问题 FAQ

**Q1：可以同时运行旧接口和新接口吗？**
A：不可以，它们都使用端口5000。需要修改端口或关闭旧接口。

**Q2：如何在Postman中快速切换算法？**
A：使用变量 `{{algorithm_type}}`，在Environment中设置值。

**Q3：是否支持批量调用？**
A：目前不支持，每次请求处理一个算法。可以使用Postman的Collection Runner批量测试。

**Q4：错误信息在哪里查看？**
A：返回的JSON中包含 `traceback` 字段，显示完整的错误堆栈。

---

## 总结

统一接口让你可以：
- ✅ **一个URL调用所有算法**
- ✅ **Postman管理更简单**（1个请求 vs 5个请求）
- ✅ **代码更易维护**（集中管理）
- ✅ **灵活切换算法**（修改参数即可）
- ✅ **保留快捷方式**（专用路由仍可用）

推荐优先使用 `POST /execute` 统一接口！
