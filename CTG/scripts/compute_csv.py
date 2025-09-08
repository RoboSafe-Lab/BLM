#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV指标均值计算脚本
用于计算不同场景下各个metrics的均值±标准差
输出格式：Vel Acc Jerk Min-TTC RSS Coll. Off-Road FDD
"""

import pandas as pd
import numpy as np
import sys
import os

def calculate_metrics_stats(csv_file_path):
    """
    计算CSV文件中各指标的均值和标准差
    
    Args:
        csv_file_path (str): CSV文件路径
    
    Returns:
        dict: 包含各指标统计信息的字典
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 定义指标映射关系
        metric_mapping = {
            'jsd_vel': 'Vel',
            'jsd_acc': 'Acc', 
            'jsd_jerk': 'Jerk',
            'min_ttc': 'Min-TTC',
            'rss_lon': 'RSS_Lon',
            'rss_lat': 'RSS_Lat',
            'collision_rate': 'Coll.',
            'off-road_rate': 'Off-Road',
            'fdd': 'FDD'
        }
        
        # 计算各指标的统计信息
        results = {}
        
        for csv_column, display_name in metric_mapping.items():
            if csv_column in df.columns:
                # 处理空值：将空字符串和NaN值排除
                valid_data = df[csv_column].replace('', np.nan).dropna()
                
                # 转换为数值类型
                try:
                    numeric_data = pd.to_numeric(valid_data, errors='coerce')
                    numeric_data = numeric_data.dropna()
                    
                    if len(numeric_data) > 0:
                        mean_value = numeric_data.mean()
                        std_value = numeric_data.std()
                        results[display_name] = {
                            'mean': round(mean_value, 3),
                            'std': round(std_value, 2)
                        }
                    else:
                        results[display_name] = None
                        
                except Exception as e:
                    print(f"警告：无法计算列 '{csv_column}' 的统计信息: {e}")
                    results[display_name] = None
            else:
                print(f"警告：列 '{csv_column}' 不存在于文件中")
                results[display_name] = None
        
        return results
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {csv_file_path}")
        return None
    except Exception as e:
        print(f"错误：读取文件时出现问题: {e}")
        return None

def print_results(stats, file_name):
    """
    打印计算结果
    
    Args:
        stats (dict): 各指标统计信息字典
        file_name (str): 文件名
    """
    if stats is None:
        return
    
    # 按照指定顺序输出
    output_order = ['Vel', 'Acc', 'Jerk', 'Min-TTC', 'RSS_Lon', 'RSS_Lat', 'Coll.', 'Off-Road', 'FDD']
    
    # 打印表头
    header = f"| {'File':<20} |"
    for metric in output_order:
        header += f" {metric:>12} |"
    print(header)
    
    # 打印分隔线
    separator = f"|{'-' * 22}|"
    for metric in output_order:
        separator += f"{'-' * 14}|"
    print(separator)
    
    # 打印数据行
    row = f"| {file_name:<20} |"
    for metric in output_order:
        if metric in stats and stats[metric] is not None:
            mean_val = stats[metric]['mean']
            std_val = stats[metric]['std']
            row += f" {mean_val:>6.3f}±{std_val:>5.2f} |"
        else:
            row += f" {'N/A':>12} |"
    print(row)

def main():
    if len(sys.argv) != 2:
        print("使用方法: python compute_csv.py <csv_file_path>")
        print("示例: python compute_csv.py metrics/ctg_common.csv")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    
    if not os.path.exists(csv_file_path):
        print(f"错误：文件 {csv_file_path} 不存在")
        sys.exit(1)
    
    stats = calculate_metrics_stats(csv_file_path)
    print_results(stats, os.path.basename(csv_file_path))

if __name__ == "__main__":
    main()