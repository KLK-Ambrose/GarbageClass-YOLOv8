"""
垃圾分类识别系统 - Streamlit 界面
运行方式: streamlit run garbage_app.py

"""
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ============== 配置 ==============
MODEL_PATH = "runs/classify/garbage_cls/weights/best.pt"

# 垃圾分类信息（类别名: (中文名, 分类类型, 处理建议, 图标)）
GARBAGE_INFO = {
    "battery": ("电池", "有害垃圾", "投放到有害垃圾收集点，不要与其他垃圾混合", "🔋"),
    "biological": ("生物垃圾", "厨余垃圾", "沥干水分后投放到厨余垃圾桶", "🥬"),
    "brown-glass": ("棕色玻璃", "可回收物", "清空内容物，冲洗干净后投放", "🍺"),
    "cardboard": ("纸板", "可回收物", "折叠压平后投放，避免沾染油污", "📦"),
    "clothes": ("衣物", "可回收物", "清洗干净，打包后投放到织物回收箱", "👕"),
    "green-glass": ("绿色玻璃", "可回收物", "清空内容物，冲洗干净后投放", "🍾"),
    "metal": ("金属", "可回收物", "清空内容物，压扁后投放", "🥫"),
    "paper": ("纸张", "可回收物", "保持干燥清洁，折叠整齐后投放", "📄"),
    "plastic": ("塑料", "可回收物", "清空内容物，冲洗压扁后投放", "🧴"),
    "shoes": ("鞋子", "可回收物", "清理干净，成对打包后投放", "👟"),
    "trash": ("其他垃圾", "其他垃圾", "投放到其他垃圾桶", "🗑️"),
    "white-glass": ("白色玻璃", "可回收物", "清空内容物，冲洗干净后投放", "🫙"),
}

# 分类类型对应的颜色
TYPE_COLORS = {
    "有害垃圾": "#FF4B4B",  # 红色
    "厨余垃圾": "#00CC66",  # 绿色
    "可回收物": "#3399FF",  # 蓝色
    "其他垃圾": "#808080",  # 灰色
}


@st.cache_resource
def load_model():
    """加载模型（缓存避免重复加载）"""
    return YOLO(MODEL_PATH)


def predict(model, image):
    """预测图片类别"""
    results = model.predict(image, verbose=False)
    probs = results[0].probs

    # 获取 top5 预测结果
    top5_indices = probs.top5
    top5_conf = probs.top5conf.tolist()
    names = results[0].names

    predictions = []
    for idx, conf in zip(top5_indices, top5_conf):
        class_name = names[idx]
        predictions.append((class_name, conf))

    return predictions


def main():
    # 页面配置
    st.set_page_config(
        page_title="垃圾分类识别系统",
        page_icon="♻️",
        layout="wide"
    )

    # 标题
    st.title("♻️ 智能垃圾分类识别系统")
    st.markdown("上传垃圾图片，AI 帮你识别分类")
    st.divider()

    # 加载模型
    with st.spinner("正在加载模型..."):
        model = load_model()

    # 两列布局
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 上传图片")

        # 图片上传
        uploaded_file = st.file_uploader(
            "选择一张垃圾图片",
            type=["jpg", "jpeg", "png", "bmp"],
            help="支持 JPG、PNG、BMP 格式"
        )
        # 选择图片来源
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

        # 显示上传的图片
        if image is not None:
            st.image(image, caption="待识别图片", use_container_width=True)

    with col2:
        st.subheader("🔍 识别结果")

        if image is not None:
            # 进行预测
            with st.spinner("正在识别..."):
                predictions = predict(model, image)

            if predictions:
                # 获取最佳预测
                best_class, best_conf = predictions[0]
                info = GARBAGE_INFO.get(best_class, ("未知", "未知", "请咨询相关部门", "❓"))
                cn_name, garbage_type, suggestion, icon = info
                type_color = TYPE_COLORS.get(garbage_type, "#808080")

                # 显示主要结果
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {type_color}22, {type_color}11);
                    border-left: 5px solid {type_color};
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                ">
                    <h1 style="margin:0; font-size: 3em;">{icon} {cn_name}</h1>
                    <h2 style="color: {type_color}; margin: 10px 0;">{garbage_type}</h2>
                    <p style="font-size: 1.2em; color: #666;">置信度: {best_conf:.1%}</p>
                </div>
                """, unsafe_allow_html=True)

                # 处理建议
                st.info(f"💡 **处理建议**: {suggestion}")

                # 显示 Top5 预测
                st.markdown("#### 📊 详细预测结果")
                for class_name, conf in predictions:
                    info = GARBAGE_INFO.get(class_name, ("未知", "未知", "", "❓"))
                    cn_name, g_type, _, icon = info

                    # 进度条颜色
                    bar_color = TYPE_COLORS.get(g_type, "#808080")
                    st.markdown(f"{icon} **{cn_name}** ({g_type})")
                    st.progress(conf, text=f"{conf:.1%}")
        else:
            # 未上传图片时的提示
            st.markdown("""
            <div style="
                text-align: center;
                padding: 60px 20px;
                background: #f8f9fa;
                border-radius: 10px;
                color: #666;
            ">
                <p style="font-size: 4em; margin: 0;">📷</p>
                <p style="font-size: 1.2em;">请在左侧上传图片或拍照</p>
            </div>
            """, unsafe_allow_html=True)

    # 底部分类指南
    st.divider()
    st.subheader("📚 垃圾分类指南")

    guide_cols = st.columns(4)

    with guide_cols[0]:
        st.markdown(f"""
        <div style="background: #FF4B4B22; padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="color: #FF4B4B;">🔴 有害垃圾</h3>
            <p>电池、灯管、药品、油漆等</p>
        </div>
        """, unsafe_allow_html=True)

    with guide_cols[1]:
        st.markdown(f"""
        <div style="background: #00CC6622; padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="color: #00CC66;">🟢 厨余垃圾</h3>
            <p>剩菜剩饭、果皮、茶叶渣等</p>
        </div>
        """, unsafe_allow_html=True)

    with guide_cols[2]:
        st.markdown(f"""
        <div style="background: #3399FF22; padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="color: #3399FF;">🔵 可回收物</h3>
            <p>纸张、塑料、玻璃、金属等</p>
        </div>
        """, unsafe_allow_html=True)

    with guide_cols[3]:
        st.markdown(f"""
        <div style="background: #80808022; padding: 15px; border-radius: 10px; text-align: center;">
            <h3 style="color: #808080;">⚫ 其他垃圾</h3>
            <p>烟蒂、陶瓷、一次性餐具等</p>
        </div>
        """, unsafe_allow_html=True)

    # 页脚
    st.divider()
    st.markdown(
        "<p style='text-align: center; color: #888;'>基于 YOLOv8 深度学习模型 | 支持 12 类垃圾识别</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()