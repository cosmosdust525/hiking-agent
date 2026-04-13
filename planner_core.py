import os
import folium
import osmnx as ox
import requests
import math
from langchain_community.llms import Ollama

# ========== 高德API配置 ==========
AMAP_KEY = os.getenv("AMAP_KEY", "")


# ========== 初始化 LLM ==========
llm = Ollama(model="deepseek-r1:8b", temperature=0.3)

# ========== 地点坐标缓存（放在这里！） ==========
CACHE = {
    #"杭州龙井村": (30.2285, 120.1192),    # 如果是高德返回的，这就是 GCJ-02
    #"杭州云栖竹径": (30.1905, 120.1037),  # 同上
}

# ========== 高德地理编码函数（地址→坐标）==========
def geocode_address_amap(address: str, city: str = None):
    """
    使用高德API将地址转换为经纬度坐标
    返回: (纬度, 经度) 或 None
    """
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        "key": AMAP_KEY,
        "address": address,
        "output": "JSON"
    }
    if city:
        params["city"] = city
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data["status"] == "1" and int(data["count"]) > 0:
            location = data["geocodes"][0]["location"]
            lng, lat = location.split(",")
            return (float(lat), float(lng))
        else:
            print(f"地理编码失败: {data.get('info', '未知错误')}")
            return None
    except Exception as e:
        print(f"请求高德API失败: {e}")
        return None

# ========== 坐标系转换函数 ==========
# ========== 坐标系转换函数 ==========
def gcj02_to_wgs84(lng, lat):
    """
    高德GCJ-02坐标系 转 WGS-84坐标系（加密→原始）
    """
    a = 6378245.0
    ee = 0.00669342162296594323
    
    def transform_lat(x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
        return ret
    
    def transform_lng(x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
        return ret
    
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    wgs_lat = lat - dlat
    wgs_lng = lng - dlng
    return wgs_lng, wgs_lat

def wgs84_to_gcj02(lng, lat):
    """
    WGS-84坐标系 转 高德GCJ-02坐标系（原始→加密）
    用于将OSMnx获取的WGS-84坐标转换为高德地图可用的GCJ-02坐标
    """
    a = 6378245.0
    ee = 0.00669342162296594323
    
    def transform_lat(x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
        return ret
    
    def transform_lng(x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
        return ret
    
    if out_of_china(lng, lat):
        return lng, lat
    
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    gcj_lat = lat + dlat
    gcj_lng = lng + dlng
    return gcj_lng, gcj_lat

def out_of_china(lng, lat):
    """
    判断是否在中国境外，境外不需要转换
    """
    return not (72.004 < lng < 137.8347 and 0.8293 < lat < 55.8271)

# ========== 核心函数 ==========
def get_route_data(start_point: str, end_point: str):
    try:
        # 获取高德API坐标（GCJ-02）
        if start_point in CACHE and end_point in CACHE:
            start_gcj = CACHE[start_point]
            end_gcj = CACHE[end_point]
        else:
            start_gcj = geocode_address_amap(start_point)
            end_gcj = geocode_address_amap(end_point)
        
        if not start_gcj or not end_gcj:
            return None, 0, 0, "地址解析失败"
        
        # 🔧 关键：将GCJ-02转换为WGS-84供OSMnx使用
        start_lng, start_lat = gcj02_to_wgs84(start_gcj[1], start_gcj[0])
        end_lng, end_lat = gcj02_to_wgs84(end_gcj[1], end_gcj[0])
        start = (start_lat, start_lng)
        end = (end_lat, end_lng)
        
        print(f"转换后WGS-84坐标: {start} -> {end}")
        
        # 后续代码不变...
        center_lat = (start[0] + end[0]) / 2
        center_lon = (start[1] + end[1]) / 2
        G = ox.graph_from_point((center_lat, center_lon), dist=2000, network_type='walk')
        
        orig_node = ox.distance.nearest_nodes(G, start[1], start[0])
        dest_node = ox.distance.nearest_nodes(G, end[1], end[0])
        route_nodes = ox.shortest_path(G, orig_node, dest_node, weight='length')
        
        # 注意：route_coords 是 WGS-84 坐标
        route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route_nodes]
        
        # 计算距离...
        total_dist_m = 0
        for i in range(len(route_nodes) - 1):
            u = route_nodes[i]
            v = route_nodes[i + 1]
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                first_key = list(edge_data.keys())[0]
                if 'length' in edge_data[first_key]:
                    total_dist_m += edge_data[first_key]['length']
        
        elev_gain_m = int(total_dist_m / 1000 * 50)
        
        return route_coords, total_dist_m, elev_gain_m, None
        
    except Exception as e:
        return None, 0, 0, str(e)


def create_map(route_coords, start_point, end_point, output_file="hiking_route.html"):
    """生成交互式地图并保存为HTML文件"""
    if not route_coords:
        return None
    
    # 🔧 关键：将 WGS-84 坐标转换为 GCJ-02（高德坐标系）
    from math import pi, sin, cos, sqrt, atan2  # 确保导入
    gcj_route_coords = []
    for lat, lon in route_coords:
        gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)
        gcj_route_coords.append((gcj_lat, gcj_lon))
    
    print(f"转换前WGS-84坐标(第一个点): {route_coords[0]}")
    print(f"转换后GCJ-02坐标(第一个点): {gcj_route_coords[0]}")
    
    # 地图中心点也用转换后的坐标
    mid_idx = len(gcj_route_coords) // 2
    m = folium.Map(
        location=gcj_route_coords[mid_idx],
        zoom_start=15,
        tiles='https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='高德地图'
    )
    
    # 绘制路线（使用转换后的坐标）
    folium.PolyLine(
        gcj_route_coords,
        color="red",
        weight=5,
        opacity=0.8,
        tooltip="徒步路线"
    ).add_to(m)
    
    # 标记起点和终点
    folium.Marker(
        gcj_route_coords[0],
        popup=f"起点: {start_point}",
        icon=folium.Icon(color="green", icon="play", prefix='fa')
    ).add_to(m)
    
    folium.Marker(
        gcj_route_coords[-1],
        popup=f"终点: {end_point}",
        icon=folium.Icon(color="red", icon="flag", prefix='fa')
    ).add_to(m)
    
    m.save(output_file)
    print(f"✅ 地图已保存到 {output_file}")
    return output_file

def generate_plan_description(start, end, dist_km, elev_m):
    """调用 LLM 生成自然语言计划"""
    prompt = f"""
你是一名专业户外向导。请根据以下路线数据，生成一段简洁、友好的徒步计划（200字以内）：
起点：{start}
终点：{end}
总距离：{dist_km:.1f} 公里
累计爬升：{elev_m} 米

请包含：预计时间（按3km/h + 每100米爬升10分钟计算）、难度评估（简单/中等/困难）、安全提示。
"""
    response = llm.invoke(prompt)
    return response.strip()