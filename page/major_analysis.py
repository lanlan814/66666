import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def show_major_analysis():
    st.title("专业数据分析")
    
    # 加载数据集
    df = pd.read_csv('student_data_adjusted_rounded.csv')

    # （1）各专业男女性别比例
    st.subheader("1. 各专业男女性别比例")
    gender_ratio = df.groupby(['专业', '性别']).size().unstack(fill_value=0)
    gender_ratio['总人数'] = gender_ratio['男'] + gender_ratio['女']
    gender_ratio['男性比例'] = gender_ratio['男'] / gender_ratio['总人数']
    gender_ratio['女性比例'] = gender_ratio['女'] / gender_ratio['总人数']
    fig1 = px.bar(gender_ratio.reset_index(), x='专业', y=['男性比例', '女性比例'], barmode='group', labels={'value': '比例'})
    fig1.update_layout(legend_title='性别')
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.dataframe(
            gender_ratio[['男性比例', '女性比例']].reset_index()
            .rename(columns={'男性比例': '男', '女性比例': '女'})
            .set_index('专业'), 
            use_container_width=True
        )

    # （2）各专业学习指标对比
    st.subheader("2. 各专业学习指标对比")
    score_trend = df.groupby('专业').agg({
        '期中考试分数': 'mean', 
        '期末考试分数': 'mean', 
        '每周学习时长（小时）': 'mean'
    }).reset_index()
    score_trend_melt = pd.melt(score_trend, id_vars='专业', value_vars=['期中考试分数', '期末考试分数'], var_name='考试类型', value_name='分数')
    fig2 = px.line(score_trend_melt, x='专业', y='分数', color='考试类型', title='各专业期中期末成绩趋势')
    fig2.add_scatter(x=score_trend['专业'], y=score_trend['每周学习时长（小时）'], name='每周学习时长', yaxis='y2')
    fig2.update_layout(yaxis2=dict(title='每周学习时长（小时）', overlaying='y', side='right'))
    col3, col4 = st.columns([2, 1])
    with col3:
        st.plotly_chart(fig2, use_container_width=True)
    with col4:
        st.dataframe(score_trend.set_index('专业'), use_container_width=True)

    # （3）各专业出勤率分析
    st.subheader("3. 各专业出勤率分析")
    attendance = df.groupby('专业')['上课出勤率'].mean().reset_index()
    # 改用px.bar的color参数直接映射数值，避免coloraxis报错
    fig3 = px.bar(
        attendance, 
        x='专业', 
        y='上课出勤率', 
        labels={'上课出勤率': '平均上课出勤率'},
        color='上课出勤率',  # 直接用color映射数值
        color_continuous_scale='Viridis',  # 颜色刻度
        range_color=[0, 1]  # 颜色范围
    )
    col5, col6 = st.columns([2, 1])
    with col5:
        st.plotly_chart(fig3, use_container_width=True)
    with col6:
        attendance_rank = attendance.sort_values('上课出勤率', ascending=False).reset_index(drop=True)
        attendance_rank.index += 1
        st.dataframe(
            attendance_rank.rename(columns={'专业': '专业', '上课出勤率': '平均出勤率'})
            .set_index('专业'), 
            use_container_width=True
        )

    # （4）大数据管理专业专项分析
    st.subheader("4. 大数据管理专业专项分析")
    big_data = df[df['专业'] == '大数据管理']
    avg_attendance = big_data['上课出勤率'].mean()
    avg_final_score = big_data['期末考试分数'].mean()
    avg_study_hours = big_data['每周学习时长（小时）'].mean()
    pass_rate = (big_data['期末考试分数'] >= 60).mean()
    col7, col8, col9, col10 = st.columns(4)
    with col7:
        st.metric("平均出勤率", f"{avg_attendance:.1%}")
    with col8:
        st.metric("平均期末分数", f"{avg_final_score:.1f}分")
    with col9:
        st.metric("通过率", f"{pass_rate:.1%}")
    with col10:
        st.metric("平均学习时长", f"{avg_study_hours:.1f}小时")
    fig4 = px.histogram(big_data, x='期末考试分数', nbins=20, title='大数据管理专业期末成绩分布')
    fig5 = px.box(big_data, y='每周学习时长（小时）', title='大数据管理专业学习时长分布')
    col11, col12 = st.columns(2)
    with col11:
        st.plotly_chart(fig4, use_container_width=True)
    with col12:
        st.plotly_chart(fig5, use_container_width=True)
