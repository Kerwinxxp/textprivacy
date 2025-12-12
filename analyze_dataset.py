# -*- coding: utf-8 -*-
"""
分析 WikiActors 数据集
包括数据行数、列数、数据类型、前三行详细信息等
"""

import json
import pandas as pd
from typing import Dict, List, Any
import os


def load_json_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 JSON 格式的数据集
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def analyze_dataset(file_path: str):
    """
    分析数据集的详细信息
    """
    print("=" * 100)
    print("📊 WikiActors 数据集分析")
    print("=" * 100)
    
    # Step 1: 加载数据
    print("\n1️⃣  加载数据...")
    data = load_json_dataset(file_path)
    
    # Step 2: 基本统计信息
    print("\n2️⃣  基本统计信息:")
    print("-" * 100)
    
    num_rows = len(data)
    print(f"   📈 总行数: {num_rows}")
    
    # 获取所有列
    if num_rows > 0:
        columns = list(data[0].keys())
        num_cols = len(columns)
        print(f"   📋 总列数: {num_cols}")
        print(f"   📝 列名: {columns}")
    else:
        print("   ⚠️  数据集为空！")
        return
    
    # Step 3: 数据类型和统计
    print("\n3️⃣  列的详细信息:")
    print("-" * 100)
    
    for col_idx, col_name in enumerate(columns, 1):
        print(f"\n   [{col_idx}] 列名: {col_name}")
        
        # 收集该列的所有值
        col_values = [row.get(col_name) for row in data]
        
        # 数据类型
        types = set(type(val).__name__ for val in col_values if val is not None)
        print(f"       数据类型: {types}")
        
        # 非空值统计
        non_null = sum(1 for val in col_values if val is not None)
        null_count = num_rows - non_null
        print(f"       非空值: {non_null} / {num_rows}")
        if null_count > 0:
            print(f"       空值: {null_count}")
        
        # 如果是字符串，统计长度
        if types == {'str'}:
            lengths = [len(str(val)) for val in col_values if val is not None]
            if lengths:
                print(f"       字符串长度 - 最小: {min(lengths)}, 最大: {max(lengths)}, 平均: {sum(lengths)/len(lengths):.1f}")
        
        # 如果是数字，显示范围
        if types in [{'int'}, {'float'}, {'int', 'float'}]:
            numeric_vals = [val for val in col_values if isinstance(val, (int, float)) and val is not None]
            if numeric_vals:
                print(f"       数值范围 - 最小: {min(numeric_vals)}, 最大: {max(numeric_vals)}, 平均: {sum(numeric_vals)/len(numeric_vals):.2f}")
    
    # Step 4: 前三行的详细信息
    print("\n" + "=" * 100)
    print("4️⃣  前三行的详细信息:")
    print("=" * 100)
    
    num_rows_to_show = min(3, num_rows)
    
    for row_idx in range(num_rows_to_show):
        print(f"\n📄 第 {row_idx + 1} 行:")
        print("-" * 100)
        
        row = data[row_idx]
        
        for col_idx, col_name in enumerate(columns, 1):
            value = row.get(col_name)
            
            # 格式化输出
            if isinstance(value, str):
                # 如果字符串太长，截断
                if len(value) > 150:
                    print(f"   [{col_idx}] {col_name}:")
                    print(f"       {value[:150]}...")
                    print(f"       (完整长度: {len(value)} 字符)")
                else:
                    print(f"   [{col_idx}] {col_name}: {value}")
            elif isinstance(value, list):
                print(f"   [{col_idx}] {col_name} (列表，{len(value)} 项):")
                # 显示列表的前 3 项
                for item_idx, item in enumerate(value[:3]):
                    print(f"       [{item_idx}] {item}")
                if len(value) > 3:
                    print(f"       ... 还有 {len(value) - 3} 项")
            elif isinstance(value, dict):
                print(f"   [{col_idx}] {col_name} (字典，{len(value)} 项):")
                # 显示字典的前 3 项
                for key_idx, (key, val) in enumerate(list(value.items())[:3]):
                    print(f"       {key}: {val}")
                if len(value) > 3:
                    print(f"       ... 还有 {len(value) - 3} 项")
            else:
                print(f"   [{col_idx}] {col_name}: {value}")
    
    # Step 5: 生成 DataFrame 摘要（可选）
    print("\n" + "=" * 100)
    print("5️⃣  转换为 Pandas DataFrame:")
    print("-" * 100)
    
    try:
        df = pd.DataFrame(data)
        print(f"\n   DataFrame 形状: {df.shape}")
        print(f"\n   数据类型:\n{df.dtypes}")
        print(f"\n   缺失值统计:\n{df.isnull().sum()}")
        print(f"\n   数值列统计:\n{df.describe()}")
    except Exception as e:
        print(f"   ⚠️  无法转换为 DataFrame: {e}")
    
    print("\n" + "=" * 100)
    print("✅ 分析完成！")
    print("=" * 100)


def save_analysis_report(file_path: str, output_file: str = None):
    """
    保存分析报告到文件
    """
    if output_file is None:
        output_file = file_path.replace('.json', '_analysis_report.txt')
    
    # 重定向输出到文件
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        analyze_dataset(file_path)
        report_content = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 分析报告已保存到: {output_file}")
    print(report_content)


if __name__ == "__main__":
    # 指定数据文件路径
    data_file = r"C:\Users\phdwf\Desktop\textreidentify\TextReIdentification\data\WikiActors_2000_filtered.json"
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"❌ 文件不存在: {data_file}")
        exit(1)
    
    # 分析数据集
    analyze_dataset(data_file)
    
    # 可选：保存报告
    # save_analysis_report(data_file)