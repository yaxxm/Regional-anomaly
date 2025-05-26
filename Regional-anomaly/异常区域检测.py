# -*- coding: utf-8 -*-
"""
异常区域检测脚本

该脚本使用孤立森林算法从区域级别特征数据中检测异常区域。
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义常量
CPU_THRESHOLD = 20  # 百分比，转换为小数为0.20
BG_RATIO_THRESHOLD = 0.70  # 70%
SHORT_UNINSTALL_DAYS = 2  # 短期卸载天数阈值

# 文件路径
data_file = 'd:\\Desktop\\地区异常\\data.csv'

def load_and_preprocess_data(file_path):
    """
    加载并预处理区域级别特征数据
    """
    print(f"正在加载数据文件: {file_path}")
    
    # 加载数据
    df = pd.read_csv(file_path)
    
    print(f"数据加载完成，共 {len(df)} 个区域")
    print("数据预览:")
    print(df.head())
    
    # 检查数据类型和缺失值
    print("\n数据类型信息:")
    print(df.dtypes)
    
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    # 确保所有特征列为数值类型
    feature_columns = df.columns.tolist()
    feature_columns.remove('地区')  # 移除区域标识列
    
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 处理可能的缺失值
    if df[feature_columns].isnull().any().any():
        print("\n检测到缺失值，使用中位数填充...")
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].median())
    
    return df

def detect_anomalies(df, feature_columns, contamination=0.05):
    """
    使用孤立森林检测异常区域
    """
    print("\n开始异常检测...")
    
    # 提取特征数据
    X = df[feature_columns].values
    
    # 初始化并训练孤立森林模型
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    
    # 预测异常
    df['异常标签'] = model.predict(X)
    df['异常分数'] = model.decision_function(X)  # 异常分数，越小越异常
    
    # 转换标签为更直观的形式
    df['是否异常'] = df['异常标签'].map({1: '正常', -1: '异常'})
    
    return df

def analyze_results(df):
    """
    分析异常检测结果
    """
    # 统计异常和正常区域数量
    anomaly_count = (df['异常标签'] == -1).sum()
    normal_count = (df['异常标签'] == 1).sum()
    
    print(f"\n检测结果统计:\n异常区域数量: {anomaly_count}\n正常区域数量: {normal_count}")
    
    # 显示异常区域详情
    anomalies = df[df['异常标签'] == -1].sort_values('异常分数')
    
    print("\n异常区域详情 (按异常程度排序):")
    pd.set_option('display.max_columns', None)  # 显示所有列
    print(anomalies[['地区', '异常分数'] + [col for col in df.columns if col not in ['地区', '异常标签', '异常分数', '是否异常']]])
    
    return anomalies

def visualize_results(df, anomalies):
    """
    可视化异常检测结果
    """
    # 创建输出目录
    output_dir = 'd:\\Desktop\\地区异常\\输出结果'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 异常分数分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['异常分数'], kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('区域异常分数分布')
    plt.xlabel('异常分数 (越小越异常)')
    plt.ylabel('频数')
    plt.savefig(os.path.join(output_dir, '异常分数分布.png'))
    
    # 2. 热力图 - 异常区域特征
    if len(anomalies) > 0:
        plt.figure(figsize=(16, 10))
        feature_cols = [col for col in df.columns if col not in ['地区', '异常标签', '异常分数', '是否异常']]
        
        # 对特征进行归一化，便于在热力图中比较
        anomalies_features = anomalies[feature_cols].copy()
        for col in feature_cols:
            anomalies_features[col] = (anomalies_features[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        sns.heatmap(anomalies_features.set_index(anomalies['地区']), cmap='YlOrRd', annot=True, fmt='.2f')
        plt.title('异常区域特征热力图 (归一化值)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '异常区域特征热力图.png'))
    
    # 3. 散点图 - 选择两个重要特征
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['区域_最大后台CPU使用率'], df['区域_最大后台时间占比'], 
                c=df['异常标签'].map({1: 'blue', -1: 'red'}), 
                alpha=0.6, s=100)
    
    # 添加区域标签
    for i, txt in enumerate(df['地区']):
        plt.annotate(txt, (df['区域_最大后台CPU使用率'].iloc[i], df['区域_最大后台时间占比'].iloc[i]),
                    fontsize=9)
    
    plt.xlabel('区域_最大后台CPU使用率')
    plt.ylabel('区域_最大后台时间占比')
    plt.title('区域异常检测散点图')
    plt.legend(['正常', '异常'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, '区域异常检测散点图.png'))
    
    # 4. 特征重要性分析 (使用随机森林)
    from sklearn.ensemble import RandomForestClassifier
    
    X = df[[col for col in df.columns if col not in ['地区', '异常标签', '异常分数', '是否异常']]]
    y = df['异常标签']
    
    # 训练随机森林
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        '特征': X.columns,
        '重要性': rf.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='重要性', y='特征', data=feature_importance)
    plt.title('特征重要性排序')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '特征重要性排序.png'))
    
    print(f"\n可视化结果已保存到: {output_dir}")

def export_results(df, output_dir):
    """
    导出异常检测结果
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 导出完整结果
    result_file = os.path.join(output_dir, '异常检测结果.csv')
    df.to_csv(result_file, index=False, encoding='utf-8-sig')
    
    # 导出异常区域结果
    anomaly_file = os.path.join(output_dir, '异常区域详情.csv')
    df[df['异常标签'] == -1].to_csv(anomaly_file, index=False, encoding='utf-8-sig')
    
    print(f"\n结果已导出到:\n{result_file}\n{anomaly_file}")

def main():
    """
    主函数
    """
    print("开始执行异常区域检测...\n")
    
    # 1. 加载和预处理数据
    df = load_and_preprocess_data(data_file)
    
    # 2. 提取特征列
    feature_columns = df.columns.tolist()
    feature_columns.remove('地区')  # 移除区域标识列
    
    # 3. 检测异常
    df = detect_anomalies(df, feature_columns)
    
    # 4. 分析结果
    anomalies = analyze_results(df)
    
    # 5. 可视化结果
    visualize_results(df, anomalies)
    
    # 6. 导出结果
    export_results(df, 'd:\\Desktop\\地区异常\\输出结果')
    
    print("\n异常区域检测完成!")

if __name__ == "__main__":
    main()