# 依赖

创建虚拟环境

```
conda create -n model4 python=3.10

conda activate model4

pip install numpy matplotlib shapely scipy geopy sympy pyproj scikit-learn plotly


```
需要GeodeticConverter.py文件



# 进度

## 1.1.1.2转弯次序设置与转弯航迹推算

```
无人机纵队 (靠前、靠后)
    ↓
根据距离 & 任务需求
    ↓
确定转弯次序（先转 or 延迟转）
    ↓
为每架无人机生成逐时刻航迹
    ↓
检查是否冲突（安全距离）
    ↓
输出：无人机转弯顺序表 + 每架无人机的详细轨迹 + 推荐速度
```