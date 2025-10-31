import pandas as pd
import numpy as np

def load_data(file_path):
    """读取学生成绩数据"""
    df = pd.read_csv(file_path)
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def feature_engineering(df):
    """特征工程：生成衍生特征、编码类别特征等"""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['专业编码'] = le.fit_transform(df['专业'])
    df['平时成绩平均分'] = df[['平时成绩1', '平时成绩2', '平时成绩3']].mean(axis=1)
    return df, le
