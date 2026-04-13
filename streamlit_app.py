import streamlit as st
from planner_core import get_route_data, create_map, generate_plan_description, wgs84_to_gcj02

st.set_page_config(page_title="AI 徒步规划助手", layout="wide")
st.title("🥾 徒步计划 - 我的AI助手")
st.markdown("输入起点和终点，我将为你规划徒步路线并生成互动地图。")

col1, col2 = st.columns(2)
with col1:
    start = st.text_input("起点", value="杭州龙井村")
with col2:
    end = st.text_input("终点", value="杭州云栖竹径")

if st.button("开始规划", type="primary"):
    if not start or not end:
        st.error("请填写起点和终点")
    else:
        with st.spinner("正在获取路线数据..."):
            route_coords, dist_m, elev_m, error = get_route_data(start, end)
        
        if error:
            st.error(f"路线规划失败: {error}")
        else:
            dist_km = dist_m / 1000
            st.success(f"路线生成成功！总距离 **{dist_km:.1f} km**，累计爬升 **{elev_m} m**")
            
         # 生成 LLM 描述
        with st.spinner("AI 正在撰写徒步计划..."):
            plan_text = generate_plan_description(start, end, dist_km, elev_m)
        st.markdown("### 📋 徒步计划")
        st.write(plan_text)
            

        # 生成地图
        with st.spinner("正在生成地图..."):
             map_path = create_map(route_coords, start, end)
        if map_path:
            st.markdown("### 🗺️ 路线地图")
             # 读取 HTML 文件并在 Streamlit 中展示
            with open(map_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=1500)
            st.caption(f"地图文件已保存至: {map_path}")
        else:
            st.warning("地图生成失败")