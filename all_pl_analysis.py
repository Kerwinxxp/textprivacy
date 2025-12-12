
# filepath: c:\Users\phdwf\Desktop\textreidentify\TextReIdentification\all_pl_analysis.py
# -*- coding: utf-8 -*-
"""
所有 PL vs PIInum 关系图 - 原始数据直接绘制
支持多种多项式拟合 + 异常值排除 + 上限包络线
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import os
import seaborn as sns
from sklearn.metrics import r2_score


def calculate_all_pl_pairs(
    prior_probs: np.ndarray,
    posterior_probs: np.ndarray,
    labels: np.ndarray,
    class_metric: np.ndarray = None,
    epsilon: float = 1e-12
) -> Tuple[List[float], List[int]]:
    """
    计算所有类对的 PL 值（不聚合）
    """
    N, C = prior_probs.shape
    
    A = np.clip(prior_probs, epsilon, 1.0)
    B = np.clip(posterior_probs, epsilon, 1.0)
    A = A / A.sum(axis=1, keepdims=True)
    B = B / B.sum(axis=1, keepdims=True)
    
    if class_metric is None:
        D = np.ones((C, C), dtype=float)
        np.fill_diagonal(D, np.inf)
    else:
        D = np.array(class_metric, dtype=float)
        D = np.where(D <= 0, epsilon, D)
    
    all_pl_values = []
    all_sample_labels = []
    
    for s in range(N):
        la = np.log(A[s])
        lb = np.log(B[s])
        diff = np.abs((lb[:, None] - lb[None, :]) - (la[:, None] - la[None, :]))
        mask = ~np.eye(C, dtype=bool)
        normed = diff[mask] / D[mask]
        
        all_pl_values.extend(normed)
        all_sample_labels.extend([labels[s]] * len(normed))
    
    return all_pl_values, all_sample_labels


# ===================== 异常值检测 =====================

def detect_outliers_iqr(y_values, k=1.5):
    """
    使用 IQR 方法检测异常值
    k=1.5 是标准值，k=3 更严格（排除更少的点）
    返回: 布尔数组，True 表示异常值
    """
    Q1 = np.percentile(y_values, 25)
    Q3 = np.percentile(y_values, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    outliers = (y_values < lower_bound) | (y_values > upper_bound)
    
    return outliers, lower_bound, upper_bound


def detect_outliers_zscore(y_values, threshold=3):
    """
    使用 Z-Score 方法检测异常值
    threshold=3 表示超过 3 倍标准差的点
    返回: 布尔数组，True 表示异常值
    """
    mean = np.mean(y_values)
    std = np.std(y_values)
    
    z_scores = np.abs((y_values - mean) / std)
    outliers = z_scores > threshold
    
    return outliers


def detect_outliers_modified_zscore(y_values, threshold=3.5):
    """
    使用改进的 Z-Score 方法（基于中位绝对偏差）
    对极端异常值更敏感
    """
    median = np.median(y_values)
    mad = np.median(np.abs(y_values - median))
    
    modified_z_scores = 0.6745 * (y_values - median) / mad
    outliers = np.abs(modified_z_scores) > threshold
    
    return outliers


# ===================== 拟合函数 =====================

def fit_and_evaluate_polynomials(x_points, y_points, degrees=[1, 2, 3]):
    """
    用不同阶数的多项式拟合 + 对数拟合，并计算 R² 分数
    """
    results = {}
    
    # 1. 多项式拟合
    for degree in degrees:
        try:
            coeffs = np.polyfit(x_points, y_points, degree)
            poly = np.poly1d(coeffs)
            y_pred = poly(x_points)
            r2 = r2_score(y_points, y_pred)
            
            results[degree] = {
                'coeffs': coeffs,
                'poly': poly,
                'r2': r2,
                'label': f'Poly (d={degree})'
            }
        except Exception as e:
            print(f"   ⚠️ 拟合失败 (degree={degree}): {e}")

    # 2. 新增：对数拟合 (Logarithmic) y = a + b * ln(x)
    # 这种曲线符合"增长速度越来越慢，越往右越平"的特征
    try:
        # 过滤掉 x <= 0 的点 (log定义域)
        valid_mask = x_points > 0
        x_log = np.log(x_points[valid_mask])
        y_valid = y_points[valid_mask]
        
        # 线性拟合: y = slope * log(x) + intercept
        slope, intercept = np.polyfit(x_log, y_valid, 1)
        
        # 定义预测函数
        def log_func(x):
            x = np.array(x)
            # 避免 log(<=0)
            return slope * np.log(np.maximum(x, 1e-10)) + intercept
            
        y_pred_log = log_func(x_points)
        r2_log = r2_score(y_points, y_pred_log)
        
        results['log'] = {
            'coeffs': [slope, intercept],
            'poly': log_func,
            'r2': r2_log,
            'label': 'Logarithmic' # 标识为对数曲线
        }
    except Exception as e:
        print(f"   ⚠️ 对数拟合失败: {e}")
    
    return results
# ...existing code...
def plot_comparison_fits(
    x_max_points, y_max_points,
    x_max_points_filtered, y_max_points_filtered,
    outlier_indices,
    save_dir: str = "pii_leakage_analysis"
):
    """
    绘制多种拟合的对比图（带异常值标记）
    """
    print(f"\n📊 拟合曲线对比（排除异常值）...")
    
    # 计算拟合：1, 2, 3阶多项式 + 对数拟合
    fit_results = fit_and_evaluate_polynomials(
        x_max_points_filtered, y_max_points_filtered, 
        degrees=[1, 2, 3] 
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    x_smooth = np.linspace(min(x_max_points_filtered), max(x_max_points_filtered), 200)
    colors = ['red', 'orange', 'green', 'purple']
    
    # 自定义排序：数字键按大小排，'log' 排最后
    def sort_key(k):
        if k == 'log': return 100
        return k
    
    sorted_keys = sorted(list(fit_results.keys()), key=sort_key)
    
    for idx, key in enumerate(sorted_keys[:4]): # 最多画4个
        if idx >= len(axes): break
        
        result = fit_results[key]
        ax = axes[idx]
        
        # 绘制正常的点
        ax.scatter(x_max_points_filtered, y_max_points_filtered, 
                  alpha=0.8, s=120, c='steelblue', 
                  edgecolors='darkblue', linewidth=2,
                  label='Normal points', zorder=3)
        
        # 绘制异常值
        if len(outlier_indices) > 0:
            x_outliers = [x_max_points[i] for i in outlier_indices]
            y_outliers = [y_max_points[i] for i in outlier_indices]
            ax.scatter(x_outliers, y_outliers, 
                      alpha=0.8, s=200, c='red', 
                      edgecolors='darkred', linewidth=2.5,
                      marker='x', label='Outliers', zorder=4)
        
        # 绘制拟合曲线
        y_smooth = result['poly'](x_smooth)
        ax.plot(x_smooth, y_smooth, 
               color=colors[idx], linewidth=3, 
               linestyle='--', alpha=0.9,
               label=result['label'])
        
        ax.set_xlabel('PIInum', fontsize=12, fontweight='bold')
        ax.set_ylabel('Max PL Value', fontsize=12, fontweight='bold')
        ax.set_title(f"{result['label']} (R² = {result['r2']:.6f})", 
                    fontsize=14, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='lower right')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "polynomial_comparison_filtered.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 拟合对比图已保存: {save_path}")
    
    # 打印 R² 对比
    print("\n📈 R² 分数对比:")
    print("-" * 60)
    for key in sorted_keys:
        r2 = fit_results[key]['r2']
        label = fit_results[key]['label']
        print(f"  {label}: R² = {r2:.6f}")
    print("-" * 60)
    
    return fit_results
# ...existing code...


def plot_all_pl_scatter_with_best_fit(
    pii_counts: Dict[str, int],
    all_pl_values: List[float],
    all_sample_labels: List[int],
    label_to_name: Dict[int, str],
    fit_results: Dict,
    x_max_points, y_max_points,
    x_max_points_filtered, y_max_points_filtered,
    outlier_indices,
    save_dir: str = "pii_leakage_analysis",
    confidence_percentile: float = 95.0  # 新增参数：置信度百分比
):
    """
    绘制所有点 + 最佳拟合曲线 + 上限包络线（带异常值标记）
    """
    # 转换为 PIInum
    x_data = []
    y_data = []
    
    for i, label in enumerate(all_sample_labels):
        name = label_to_name.get(label, f"Unknown_{label}")
        if name in pii_counts:
            x_data.append(pii_counts[name])
            y_data.append(all_pl_values[i])
    
    if len(x_data) == 0:
        return
    
    # ================= 修改开始 =================
    # 强制使用 Log 拟合作为基准（只要计算成功），忽略 R2 大小
    if 'log' in fit_results:
        best_key = 'log'
        print(f"\n🔒 已强制锁定使用 Logarithmic 曲线作为 Upper Bound 基准")
    else:
        # 如果 Log 拟合失败（例如数据全为负），才回退到自动选择
        best_key = max(fit_results.keys(), key=lambda k: fit_results[k]['r2'])
        print(f"\n⚠️ 未找到 Log 拟合结果，回退到最佳 R2: {best_key}")
    # ================= 修改结束 =================

    best_result = fit_results[best_key]
    best_poly = best_result['poly']
    best_r2 = best_result['r2']
    best_label = best_result['label'] 
    
    print(f"📊 绘制拟合图: {best_label}")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    
    # ① 绘制所有散点（半透明）
    ax.scatter(x_data, y_data, 
              alpha=0.15, s=15,
              c='lightblue', edgecolors='none',
              label=f'All PL values (n={len(x_data)})')
    
    # ② 绘制正常的 Max PL 点（蓝色）
    ax.scatter(x_max_points_filtered, y_max_points_filtered,
              alpha=0.85, s=130,
              c='steelblue', edgecolors='darkblue', linewidth=2.5,
              marker='o', zorder=5,
              label=f'Normal Max PL (n={len(x_max_points_filtered)})')
    
    # ③ 绘制异常值点（红色 X）
    if len(outlier_indices) > 0:
        x_outliers = [x_max_points[i] for i in outlier_indices]
        y_outliers = [y_max_points[i] for i in outlier_indices]
        ax.scatter(x_outliers, y_outliers,
                  alpha=0.9, s=200,
                  c='red', edgecolors='darkred', linewidth=2.5,
                  marker='x', zorder=6,
                  label=f'Outliers (n={len(outlier_indices)})')
    
    # ④ 绘制最佳拟合曲线 (Mean Trend)
    x_smooth = np.linspace(min(x_max_points_filtered), max(x_max_points_filtered), 300)
    y_smooth = best_poly(x_smooth)
    
    ax.plot(x_smooth, y_smooth, 
           color='darkgreen', linewidth=4, 
           linestyle='--', alpha=0.9,
           label=f'Best Fit: {best_label} (R²={best_r2:.4f})')

    # ===================== 绘制上限包络线 =====================
    # 1. 计算残差
    y_pred_filtered = best_poly(x_max_points_filtered)
    residuals = y_max_points_filtered - y_pred_filtered
    
    # 2. 计算偏移量
    upper_shift = np.percentile(residuals, confidence_percentile)
    
    # 3. 生成上限曲线 (基准曲线 + 偏移量)
    # 如果基准是 Log 曲线，这个上限曲线也是 Log 曲线，符合"弯弯的变平"的要求
    y_upper_bound = y_smooth + upper_shift
    
    
    # 4. 绘制上限曲线
    ax.plot(x_smooth, y_upper_bound, 
           color='crimson', linewidth=3, 
           linestyle='-.', alpha=0.8,
           label=f'Upper Bound ({confidence_percentile}% Confidence)')
    
    # 5. 填充区域 (可选，增加视觉效果)
    ax.fill_between(x_smooth, y_smooth, y_upper_bound, color='green', alpha=0.05)
    
    # 6. 计算实际覆盖率 (检查有多少 Max PL 点在这条线下面)
    # 注意：这里我们检查所有的 Max PL 点（包括异常值）
    y_pred_all_max = best_poly(x_max_points) + upper_shift
    covered_mask = y_max_points <= y_pred_all_max
    coverage_count = np.sum(covered_mask)
    coverage_pct = (coverage_count / len(y_max_points)) * 100
    
    print(f"   📈 上限包络线统计:")
    print(f"     - 设定置信度: {confidence_percentile}% (基于正常点)")
    print(f"     - 实际覆盖率: {coverage_pct:.2f}% (所有 Max PL 点)")
    print(f"     - 偏移量: +{upper_shift:.4f}")
    # ==============================================================
    
    # ⑤ 计算相关系数
    corr_all = np.corrcoef(x_data, y_data)[0, 1]
    corr_normal = np.corrcoef(x_max_points_filtered, y_max_points_filtered)[0, 1]
    
    ax.set_xlabel('PIInum', fontsize=16, fontweight='bold')
    ax.set_ylabel('PL Value', fontsize=16, fontweight='bold')
    ax.set_title(f'All PL Values vs PIInum with Upper Bound Envelope\n'
                f'Trend R²={best_r2:.4f} | Upper Bound Covers {coverage_pct:.1f}% of Max Points', 
                fontsize=18, fontweight='bold')
    
    # 显示拟合方程和上限方程
    
    # 显示拟合方程
    if best_key == 'log':
        coeffs = best_result['coeffs']
        equation_text = (f"Trend: y = {coeffs[0]:.4f} * ln(x) + {coeffs[1]:.4f}\n"
                         f"Upper: y = Trend + {upper_shift:.4f}")
    else:
        equation_text = (f"Trend: {best_label}\n"
                         f"Upper: y = Trend + {upper_shift:.4f}")

    ax.text(0.02, 0.98, equation_text, 
           transform=ax.transAxes, fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
           family='monospace')
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13, loc='upper left', framealpha=0.95)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "all_pl_scatter_best_fit_filtered.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 最佳拟合图（排除异常值）已保存: {save_path}")
    print(f"   - 总点数: {len(x_data)}")
    print(f"   - 正常 Max PL 点数: {len(x_max_points_filtered)}")
    print(f"   - 异常值点数: {len(outlier_indices)}")
    print(f"   - 正常点相关系数: {corr_normal:.4f}")


def print_outlier_info(x_max_points, y_max_points, outlier_indices, label_to_name, pii_counts):
    """
    打印异常值的详细信息
    """
    print("\n" + "=" * 80)
    print("🔍 异常值详细信息")
    print("=" * 80)
    
    for idx in sorted(outlier_indices):
        pii = x_max_points[idx]
        max_pl = y_max_points[idx]
        
        # 找对应的演员
        actor_name = None
        for name, pii_num in pii_counts.items():
            if pii_num == pii:
                actor_name = name
                break
        
        print(f"\n⚠️  异常值 #{idx+1}")
        print(f"   演员: {actor_name or 'Unknown'}")
        print(f"   PIInum: {pii}")
        print(f"   Max PL: {max_pl:.4f}")


# ===================== 辅助函数 =====================

def load_pii_counts(data_file: str) -> Dict[str, int]:
    """加载 PIInum"""
    df = pd.read_json(data_file)
    if 'name' not in df.columns or 'PIInum' not in df.columns:
        raise ValueError(f"数据文件必须包含 'name' 和 'PIInum' 列")
    return dict(zip(df['name'], df['PIInum']))


def aggregate_by_label_probs_mult(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12):
    """聚合为对象级概率"""
    labels = labels.astype(int)
    uniq = np.unique(labels)
    P = np.clip(probs, eps, 1.0)
    agg_logp = []
    agg_labels = []
    for lb in uniq:
        block = P[labels == lb]
        logp = np.log(block).sum(axis=0)
        agg_logp.append(logp)
        agg_labels.append(lb)
    agg_logp = np.vstack(agg_logp)
    
    m = np.max(agg_logp, axis=1, keepdims=True)
    ex = np.exp(agg_logp - m)
    agg_probs = ex / ex.sum(axis=1, keepdims=True)
    return agg_probs, np.array(agg_labels)


# ===================== 主函数 =====================

def all_pl_comprehensive_analysis(
    data_file: str,
    prior_prob_file: str,
    posterior_prob_file: str,
    class_metric_file: str = "class_distance_matrix.json",
    output_dir: str = "pii_leakage_analysis",
    outlier_method: str = "iqr",  # "iqr", "zscore", 或 "modified_zscore"
    outlier_k: float = 1.5  # IQR 参数：1.5（标准）、3.0（严格）
):
    """
    绘制所有 PL 值的关系图（支持异常值排除）
    
    outlier_method: "iqr" (推荐), "zscore", "modified_zscore"
    outlier_k: IQR 方法的参数，1.5=标准，3.0=严格
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"所有 PL 值 vs PIInum 关系分析（异常值排除：{outlier_method}）")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n1️⃣  加载 PIInum...")
    pii_counts = load_pii_counts(data_file)
    
    print("\n2️⃣  加载概率数据...")
    with open(prior_prob_file, 'r', encoding='utf-8') as f:
        prior_data = json.load(f)
    with open(posterior_prob_file, 'r', encoding='utf-8') as f:
        posterior_data = json.load(f)
    
    print("\n3️⃣  聚合为对象级概率...")
    prior_probs, prior_labels = aggregate_by_label_probs_mult(
        np.array(prior_data['probs']), np.array(prior_data['labels']))
    posterior_probs, posterior_labels = aggregate_by_label_probs_mult(
        np.array(posterior_data['probs']), np.array(posterior_data['labels']))
    
    print("\n4️⃣  对齐数据...")
    labels = np.array([i for i in range(len(prior_probs)) if i < len(posterior_probs)])
    prior_probs = prior_probs[:len(labels)]
    posterior_probs = posterior_probs[:len(labels)]
    
    print("\n5️⃣  加载类别距离矩阵...")
    try:
        with open(class_metric_file, 'r', encoding='utf-8') as f:
            distance_data = json.load(f)
        class_metric = np.array(distance_data['distance_matrix'])
    except FileNotFoundError:
        class_metric = None
    
    print("\n6️⃣  计算所有类对的 PL 值...")
    all_pl_values, all_sample_labels = calculate_all_pl_pairs(
        prior_probs, posterior_probs, labels, class_metric)
    
    print("\n7️⃣  构建标签映射...")
    df = pd.read_json(data_file)
    if 'name' in df.columns:
        unique_names = sorted(df['name'].unique())
        label_to_name = {i: name for i, name in enumerate(unique_names)}
    else:
        label_to_name = {i: f"User_{i}" for i in range(len(labels))}
    
    print("\n8️⃣  计算每个 PIInum 的 MAX PL...")
    x_data = []
    y_data = []
    for i, label in enumerate(all_sample_labels):
        name = label_to_name.get(label, f"Unknown_{label}")
        if name in pii_counts:
            x_data.append(pii_counts[name])
            y_data.append(all_pl_values[i])
    
    pii_to_max_pl = {}
    for x, y in zip(x_data, y_data):
        if x not in pii_to_max_pl:
            pii_to_max_pl[x] = y
        else:
            pii_to_max_pl[x] = max(pii_to_max_pl[x], y)
    
    x_max_points = np.array(sorted(list(pii_to_max_pl.keys())))
    y_max_points = np.array([pii_to_max_pl[x] for x in x_max_points])
    
    # ===== 关键步骤：异常值检测 =====
    print(f"\n9️⃣  检测异常值（方法：{outlier_method}）...")
    
    if outlier_method == "iqr":
        outliers, lower, upper = detect_outliers_iqr(y_max_points, k=outlier_k)
        print(f"   IQR 范围: [{lower:.4f}, {upper:.4f}]")
    elif outlier_method == "zscore":
        outliers = detect_outliers_zscore(y_max_points, threshold=3)
    elif outlier_method == "modified_zscore":
        outliers = detect_outliers_modified_zscore(y_max_points, threshold=3.5)
    else:
        raise ValueError(f"未知的异常值检测方法: {outlier_method}")
    
    outlier_indices = np.where(outliers)[0]
    print(f"   检测到异常值: {len(outlier_indices)} 个")
    
    if len(outlier_indices) > 0:
        print(f"   异常值索引: {list(outlier_indices)}")
    
    # 过滤数据
    x_max_points_filtered = x_max_points[~outliers]
    y_max_points_filtered = y_max_points[~outliers]
    
    print(f"   过滤后点数: {len(x_max_points_filtered)} 个")
    
    # 打印异常值信息
    print_outlier_info(x_max_points, y_max_points, outlier_indices, label_to_name, pii_counts)
    
    # 拟合多项式
    print("\n🔟 多项式拟合对比...")
    fit_results = plot_comparison_fits(
        x_max_points, y_max_points,
        x_max_points_filtered, y_max_points_filtered,
        outlier_indices,
        output_dir
    )
    
    # 绘制最佳拟合
    print("\n1️⃣1️⃣ 绘制最佳拟合图...")
    plot_all_pl_scatter_with_best_fit(
        pii_counts, all_pl_values, all_sample_labels, 
        label_to_name, fit_results,
        x_max_points, y_max_points,
        x_max_points_filtered, y_max_points_filtered,
        outlier_indices,
        output_dir,
        confidence_percentile=95.0  # 95% 置信度包络线
    )
    
    print("\n" + "=" * 80)
    print("✅ 所有分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    data_file = r"C:\Users\phdwf\OneDrive\Desktop\textreidentify\TextReIdentification\data\WikiActors_50_masked_cleaned.json"
    prior_prob_file = "budget_2.0_independent_masked_abstract.json"
    posterior_prob_file = "budget_2.0_independent_noise_abstract.json"
    
    all_pl_comprehensive_analysis(
        data_file=data_file,
        prior_prob_file=prior_prob_file,
        posterior_prob_file=posterior_prob_file,
        class_metric_file="noise_2.0_independent_distance_matrix.json",
        output_dir="pii_leakage_analysis",

        outlier_method="iqr",      # "iqr", "zscore", "modified_zscore"
        outlier_k=1.5              # IQR 参数：1.5（标准）、3.0（严格）
    )