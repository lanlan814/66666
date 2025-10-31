import streamlit as st

def show_project_intro():
    st.title("学生成绩分析与预测系统")
    st.markdown("***")

    a1, a2 = st.columns(2)
    with a1:
        st.subheader("🗒️ 项目概述")
        st.markdown("本项目是一个基于streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩。")
        st.subheader("✨ 主要特点")
        st.markdown("""
        - 📊 数据可视化：多维度展示学生学业数据
        - 📚 专业分析：按专业分类的详细统计分析
        - 🧠 智能预测：基于机器学习模型的成绩预测
        - 💡 学习建议：根据预测结果提供个性化反馈
        """)
    with a2:
        st.image("https://github.com/lanlan814/66666/raw/main/show.png", caption="学生数据分析示意图", width=600)

    st.markdown("***")
    st.subheader("🎯 项目目标")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🎯 目标一：分析影响因素")
        st.markdown("""
        - 识别关键学习指标
        - 探索成绩相关因素
        - 提供数据支持决策
        """)
    with col2:
        st.markdown("#### 🎯 目标二：可视化展示")
        st.markdown("""
        - 专业对比分析
        - 性别差异研究
        - 学习模式识别
        """)
    with col3:
        st.markdown("#### 🎯 目标三：成绩预测")
        st.markdown("""
        - 机器学习模型
        - 个性化预测
        - 及时干预预警
        """)

    st.markdown("***")
    st.subheader("🛠️ 技术架构")
    tech_cols = st.columns(4)
    with tech_cols[0]:
        st.markdown("**前端框架**：Streamlit")
    with tech_cols[1]:
        st.markdown("**数据处理**：Pandas、NumPy")
    with tech_cols[2]:
        st.markdown("**可视化**：Plotly、Matplotlib")
    with tech_cols[3]:
        st.markdown("**机器学习**：Scikit-learn")
