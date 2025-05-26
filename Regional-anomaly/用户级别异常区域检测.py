# -*- coding: utf-8 -*-
"""
用户级别异常区域检测脚本

该脚本从用户级别的应用使用数据中聚合生成区域级别特征，然后使用孤立森林算法检测异常区域。
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Source Han Sans SC', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'SimHei', 'Arial Unicode MS']  # 添加思源黑体作为首选字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 注册思源黑体字体文件
import matplotlib.font_manager as fm
fm.fontManager.addfont('/tmp/SourceHanSansSC-Regular.otf')  # 添加下载的思源黑体字体文件

# 定义常量
CPU_THRESHOLD = 20  # 百分比
BG_RATIO_THRESHOLD = 0.70  # 70%
SHORT_UNINSTALL_DAYS = 2  # 短期卸载天数阈值

# 文件路径
data_file = '/mnt/ymj/vivo/地区异常/data2_extended.csv'
output_dir = '/mnt/ymj/vivo/地区异常/输出结果'

def load_and_preprocess_data(file_path):
    """
    加载并预处理用户级别数据
    """
    print(f"正在加载数据文件: {file_path}")
    
    # 加载数据
    df = pd.read_csv(file_path)
    
    print(f"数据加载完成，共 {len(df)} 条记录")
    print("数据预览:")
    print(df.head())
    
    # 检查数据类型和缺失值
    print("\n数据类型信息:")
    print(df.dtypes)
    
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    # 转换日期列为datetime对象
    date_columns = ['download_date', 'uninstall_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # 确保数值列为正确的数值类型
    numeric_columns = ['cpu_average_pct', 'cpu_peak_pct', 'foreground_minutes', 'background_minutes', 'alive_days']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 处理缺失值
    # 对于时长或CPU使用率，使用0填充
    time_cpu_columns = ['cpu_average_pct', 'cpu_peak_pct', 'foreground_minutes', 'background_minutes']
    for col in time_cpu_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 对于alive_days，使用中位数填充
    if 'alive_days' in df.columns and df['alive_days'].isnull().any():
        df['alive_days'] = df['alive_days'].fillna(df['alive_days'].median())
    
    # 计算辅助列
    # 是否卸载
    df['is_uninstalled'] = df['uninstall_date'].notna().astype(int)
    
    # 后台时间占比
    total_time = df['foreground_minutes'] + df['background_minutes']
    df['background_time_ratio'] = np.where(total_time > 0, 
                                          df['background_minutes'] / total_time, 
                                          0)  # 当总时长为0时，ratio设为0
    
    return df

def calculate_regional_features(df):
    """
    计算并聚合生成区域级别特征
    """
    print("\n开始计算区域级别特征...")
    
    # 创建一个空的DataFrame来存储区域级别特征
    regions = df['user_region'].unique()
    regional_features = pd.DataFrame({'地区': regions})
    
    # 对每个区域计算特征
    regional_features_list = []
    
    for region in regions:
        region_data = df[df['user_region'] == region]
        region_features = {'地区': region}
        
        # 获取该区域内的所有App
        apps_in_region = region_data['app_id'].unique()
        
        # 1. 区域_最大后台CPU使用率
        app_avg_cpu = region_data.groupby('app_id')['cpu_average_pct'].mean()
        region_features['区域_最大后台CPU使用率'] = app_avg_cpu.max() if len(app_avg_cpu) > 0 else 0
        
        # 2. 区域_平均后台CPU使用率
        region_features['区域_平均后台CPU使用率'] = app_avg_cpu.mean() if len(app_avg_cpu) > 0 else 0
        
        # 3. 区域_后台CPU超阈值App占比
        high_cpu_apps = sum(app_avg_cpu > CPU_THRESHOLD)
        region_features['区域_后台CPU超阈值App占比'] = high_cpu_apps / len(apps_in_region) if len(apps_in_region) > 0 else 0
        
        # 4. 区域_最大峰值CPU使用率
        app_peak_cpu = region_data.groupby('app_id')['cpu_peak_pct'].mean()
        region_features['区域_最大峰值CPU使用率'] = app_peak_cpu.max() if len(app_peak_cpu) > 0 else 0
        
        # 5. 区域_最大后台时间占比
        app_bg_ratio = region_data.groupby('app_id')['background_time_ratio'].mean()
        region_features['区域_最大后台时间占比'] = app_bg_ratio.max() if len(app_bg_ratio) > 0 else 0
        
        # 6. 区域_平均后台时间占比
        region_features['区域_平均后台时间占比'] = app_bg_ratio.mean() if len(app_bg_ratio) > 0 else 0
        
        # 7. 区域_后台时间占比超阈值App数
        high_bg_ratio_apps = sum(app_bg_ratio > BG_RATIO_THRESHOLD)
        region_features['区域_后台时间占比超阈值App数'] = high_bg_ratio_apps
        
        # 8. 区域_平均前台使用时长
        app_fg_time = region_data.groupby('app_id')['foreground_minutes'].mean()
        region_features['区域_平均前台使用时长'] = app_fg_time.mean() if len(app_fg_time) > 0 else 0
        
        # 9. 区域_最高卸载率App
        app_uninstall_rates = []
        for app in apps_in_region:
            app_data = region_data[region_data['app_id'] == app]
            uninstall_rate = app_data['is_uninstalled'].mean()
            app_uninstall_rates.append(uninstall_rate)
        
        region_features['区域_最高卸载率App'] = max(app_uninstall_rates) if app_uninstall_rates else 0
        
        # 10. 区域_平均卸载率
        region_features['区域_平均卸载率'] = np.mean(app_uninstall_rates) if app_uninstall_rates else 0
        
        # 11. 区域_短期卸载App占比
        uninstalled_data = region_data[region_data['is_uninstalled'] == 1]
        short_uninstalls = uninstalled_data[uninstalled_data['alive_days'] <= SHORT_UNINSTALL_DAYS]
        
        region_features['区域_短期卸载App占比'] = len(short_uninstalls) / len(uninstalled_data) if len(uninstalled_data) > 0 else 0
        
        # 12. 区域_高后台CPU且高后台时间占比App数
        high_cpu_and_bg_apps = 0
        for app in apps_in_region:
            app_data = region_data[region_data['app_id'] == app]
            avg_cpu = app_data['cpu_average_pct'].mean()
            avg_bg_ratio = app_data['background_time_ratio'].mean()
            
            if avg_cpu > CPU_THRESHOLD and avg_bg_ratio > BG_RATIO_THRESHOLD:
                high_cpu_and_bg_apps += 1
        
        region_features['区域_高后台CPU且高后台时间占比App数'] = high_cpu_and_bg_apps
        
        # 13. 区域_调度次数特征 (假设没有直接的调度次数字段，使用0作为占位符)
        region_features['区域_调度次数特征'] = 0
        
        regional_features_list.append(region_features)
    
    # 将所有区域的特征合并为一个DataFrame
    regional_features_df = pd.DataFrame(regional_features_list)
    
    print("区域级别特征计算完成，共 {} 个区域".format(len(regional_features_df)))
    print("特征预览:")
    print(regional_features_df.head())
    
    return regional_features_df

def detect_anomalies(df, contamination=0.05):
    """
    使用孤立森林检测异常区域
    """
    print("\n开始异常检测...")
    
    # 提取特征列
    feature_columns = df.columns.tolist()
    feature_columns.remove('地区')  # 移除区域标识列
    
    # 处理可能的缺失值
    if df[feature_columns].isnull().any().any():
        print("\n检测到缺失值，使用中位数填充...")
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].median())
    
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
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 异常分数分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['异常分数'], kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('区域异常分数分布')
    plt.xlabel('异常分数 (越小越异常)')
    plt.ylabel('频数')
    plt.savefig(os.path.join(output_dir, '异常分数分布.png'))
    plt.close()
    
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
        plt.close()
    
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
    plt.close()
    
    # 4. 特征重要性分析 (使用随机森林)
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
    plt.close()
    
    print(f"\n可视化结果已保存到: {output_dir}")

def export_results(df):
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
    print("开始执行用户级别异常区域检测...\n")
    
    # 1. 加载和预处理用户级别数据
    user_data = load_and_preprocess_data(data_file)
    
    # 2. 计算区域级别特征
    regional_features = calculate_regional_features(user_data)
    
    # 3. 检测异常
    regional_features = detect_anomalies(regional_features)
    
    # 4. 分析结果
    anomalies = analyze_results(regional_features)
    
    # 5. 可视化结果
    visualize_results(regional_features, anomalies)
    
    # 6. 导出结果
    export_results(regional_features)
    
    print("\n用户级别异常区域检测完成!")

if __name__ == "__main__":
    main()