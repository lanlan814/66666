import streamlit as st
from page import project_intro, major_analysis, grade_prediction

st.set_page_config(page_title="学生成绩分析与预测系统", layout="wide")

page = st.sidebar.radio("导航菜单", ["项目介绍", "专业数据分析", "成绩预测"])

if page == "项目介绍":
    project_intro.show_project_intro()
elif page == "专业数据分析":
    major_analysis.show_major_analysis()
elif page == "成绩预测":
    grade_prediction.show_grade_prediction()
