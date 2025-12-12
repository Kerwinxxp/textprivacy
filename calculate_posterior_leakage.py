# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, List


# ===================== 聚合与对齐 =====================
def extract_and_compute_class_metric(tri_model_path: str, num_classes: int):
    """
    从 TRI 模型提取类别嵌入并计算距离矩阵
    """
    from transformers import AutoModelForSequenceClassification
    from sklearn.metrics.pairwise import cosine_distances
    
    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(tri_model_path)
    
    # 提取分类器权重
    if hasattr(model, 'classifier'):
        if hasattr(model.classifier, 'weight'):
            class_embeddings = model.classifier.weight.detach().cpu().numpy()
        elif hasattr(model.classifier, 'out_proj'):
            class_embeddings = model.classifier.out_proj.weight.detach().cpu().numpy()
    else:
        raise AttributeError("Cannot find classifier in model")
    
    # 计算余弦距离
    distance_matrix = cosine_distances(class_embeddings)
    np.fill_diagonal(distance_matrix, 1e-12)
    
    print(f"✅ 从模型提取类别距离矩阵: {distance_matrix.shape}")
    return distance_matrix

def aggregate_by_label_logits(logits: np.ndarray, labels: np.ndarray):
    """
    将同一 label 的多个窗口级 logits 相加（≈独立证据相乘），
    再 softmax 得到“对象级”概率分布。
    返回：agg_probs (num_labels, C), agg_labels (num_labels,)
    """
    labels = labels.astype(int)
    uniq = np.unique(labels)
    agg_logits = []
    agg_labels = []
    for lb in uniq:
        L = logits[labels == lb]          # 该人的所有窗口 (k, C)
        summed = L.sum(axis=0)            # logits 相加
        agg_logits.append(summed)
        agg_labels.append(lb)
    agg_logits = np.vstack(agg_logits)    # (num_labels, C)

    # 数值稳定 softmax
    m = np.max(agg_logits, axis=1, keepdims=True)
    ex = np.exp(agg_logits - m)
    probs = ex / ex.sum(axis=1, keepdims=True)
    return probs, np.array(agg_labels)


def aggregate_by_label_probs_mult(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12):
    """
    备选：当没有 logits 时，把同一 label 的窗口级概率相乘（log 概率相加）再归一化。
    返回：agg_probs (num_labels, C), agg_labels (num_labels,)
    """
    labels = labels.astype(int)
    uniq = np.unique(labels)
    P = np.clip(probs, eps, 1.0)
    agg_logp = []
    agg_labels = []
    for lb in uniq:
        block = P[labels == lb]           # (k, C)
        logp = np.log(block).sum(axis=0)  # 概率乘积 → log 概率相加
        agg_logp.append(logp)
        agg_labels.append(lb)
    agg_logp = np.vstack(agg_logp)

    # log-sum-exp 归一化
    m = np.max(agg_logp, axis=1, keepdims=True)
    ex = np.exp(agg_logp - m)
    agg_probs = ex / ex.sum(axis=1, keepdims=True)
    return agg_probs, np.array(agg_labels)


def align_by_label_after_aggregation(
    prior_probs: np.ndarray, prior_labels: np.ndarray,
    posterior_probs: np.ndarray, posterior_labels: np.ndarray
):
    """
    假设两侧都已按 label 聚合为对象级分布。
    对共同的 label 排序后对齐，保证同一 label 在 A/B 同一行。
    """
    common = np.intersect1d(np.unique(prior_labels), np.unique(posterior_labels))
    prior_idx = {lb: i for i, lb in enumerate(prior_labels)}
    post_idx  = {lb: i for i, lb in enumerate(posterior_labels)}
    A, B, L = [], [], []
    for lb in sorted(common):
        A.append(prior_probs[prior_idx[lb]])
        B.append(posterior_probs[post_idx[lb]])
        L.append(lb)
    return np.vstack(A), np.vstack(B), np.array(L)


# ===================== Pairwise mPL 计算 =====================

def calculate_pairwise_posterior_leakage(
    prior_probs: np.ndarray,      # shape: (N, C)  —— A
    posterior_probs: np.ndarray,  # shape: (N, C)  —— B
    class_metric: np.ndarray = None,  # shape: (C, C), d_{i,j}; 不提供则全为1
    epsilon: float = 1e-12
) -> Dict[str, Any]:
    """
    逐对 (i,j) 类别计算 | log(B_i/B_j) - log(A_i/A_j) | / d_{i,j}
    返回所有 pairwise PL 值的集合，而不是 per-sample 平均。
    """
    assert prior_probs.shape == posterior_probs.shape, "prior/posterior 维度不一致"
    N, C = prior_probs.shape

    # 数值稳定 + 行归一化
    A = np.clip(prior_probs, epsilon, 1.0)
    B = np.clip(posterior_probs, epsilon, 1.0)
    A = A / A.sum(axis=1, keepdims=True)
    B = B / B.sum(axis=1, keepdims=True)

    # 类间度量
    if class_metric is None:
        D = np.ones((C, C), dtype=float)
        np.fill_diagonal(D, np.inf)  # i==j 不计
    else:
        D = np.array(class_metric, dtype=float)
        assert D.shape == (C, C)
        D = np.where(D <= 0, epsilon, D)

    pairwise_pl_values = []

    for s in range(N):
        la = np.log(A[s])   # (C,)
        lb = np.log(B[s])   # (C,)
        # 逐对差值：[lb_i - lb_j] - [la_i - la_j]
        diff = np.abs((lb[:, None] - lb[None, :]) - (la[:, None] - la[None, :]))  # (C,C)
        mask = ~np.eye(C, dtype=bool)
        normed = diff[mask] / D[mask]
        pairwise_pl_values.extend(normed.tolist())

    arr = np.array(pairwise_pl_values)
    stats = {
        'mean_pl': float(np.mean(arr)) if arr.size else 0.0,
        'std_pl': float(np.std(arr)) if arr.size else 0.0,
        'min_pl': float(np.min(arr)) if arr.size else 0.0,
        'max_pl': float(np.max(arr)) if arr.size else 0.0,
        'median_pl': float(np.median(arr)) if arr.size else 0.0,
        'total_counts': int(arr.size)
    }
    return {'pairwise_pl': pairwise_pl_values, 'statistics': stats}


# ===================== 可视化（pairwise 版本） =====================

def create_pl_distribution_plot(
    pl_values: List[float],
    save_path: str,
    dataset_comparison: str,
    violation_threshold: float = 3.0
):
    """把每个 (i,j) 的 PL 当作一个样本/计数来画直方图"""
    arr = np.array(pl_values)
    if len(arr) == 0:
        print("No PL values to plot")
        return

    plt.figure(figsize=(12, 8))

    counts, bin_edges = np.histogram(arr, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    not_violated_mask = bin_centers <= violation_threshold
    violated_mask = bin_centers > violation_threshold

    plt.bar(bin_centers[not_violated_mask], counts[not_violated_mask],
            width=bin_width, alpha=0.7, edgecolor='black',
            color='steelblue', label='Not violated')

    if np.any(violated_mask):
        plt.bar(bin_centers[violated_mask], counts[violated_mask],
                width=bin_width, alpha=0.7, edgecolor='black',
                color='red', label='Violated')

    plt.axvline(violation_threshold, color='darkred', linestyle='-', linewidth=3,
                label=f'Violation threshold (ε = {violation_threshold})')
    plt.axvline(np.mean(arr), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(arr):.4f}')

    violated_values = arr[arr > violation_threshold]
    violation_ratio = len(violated_values) / len(arr) * 100 if len(arr) > 0 else 0.0

    plt.xlabel('Pairwise Posterior Leakage (PL)', fontsize=12)
    plt.ylabel('Count (number of class pairs)', fontsize=12)
    plt.title(f'Posterior Leakage Distribution (pairwise)\n{dataset_comparison}\nViolation ratio: {violation_ratio:.1f}%',
              fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Pairwise PL distribution plot saved: {save_path}")


# ===================== 主流程 =====================

def analyze_posterior_leakage_between_datasets(
    prior_file: str,
    posterior_file: str,
    save_dir: str = "posterior_leakage_results",
    class_metric: np.ndarray = None,   # 可选：传入 (C,C) 的类间距离矩阵
    violation_threshold: float = 3.0
):
    """
    分析两个数据集之间的后验概率泄露（pairwise 统计）：
    - 窗口级 -> （按 label 聚合）对象级概率
    - 对同一对象的所有类对 (i,j) 产生一个 PL 值（count=1）
    - 直方图按所有 pairwise PL 值作图
    """
    print(f"\n{'='*60}")
    print(f"分析后验泄露（pairwise 成对赔率定义）")
    print(f"Prior (未加噪): {prior_file}")
    print(f"Posterior (加噪): {posterior_file}")
    print(f"{'='*60}\n")

    # 加载数据
    with open(prior_file, 'r', encoding='utf-8') as f:
        prior_data = json.load(f)
    with open(posterior_file, 'r', encoding='utf-8') as f:
        posterior_data = json.load(f)

    prior_labels_raw = np.array(prior_data['labels'])
    posterior_labels_raw = np.array(posterior_data['labels'])

    # --- 优先使用 logits 做“联合观测”聚合；无 logits 时退化为概率乘积聚合 ---
    if 'logits' in prior_data and 'logits' in posterior_data:
        prior_logits_raw = np.array(prior_data['logits'])
        posterior_logits_raw = np.array(posterior_data['logits'])

        prior_probs_agg, prior_labels_agg = aggregate_by_label_logits(prior_logits_raw, prior_labels_raw)
        posterior_probs_agg, posterior_labels_agg = aggregate_by_label_logits(posterior_logits_raw, posterior_labels_raw)
        print("已使用 logits 聚合为对象级概率（联合观测）。")
    else:
        prior_probs_raw = np.array(prior_data['probs'])
        posterior_probs_raw = np.array(posterior_data['probs'])

        prior_probs_agg, prior_labels_agg = aggregate_by_label_probs_mult(prior_probs_raw, prior_labels_raw)
        posterior_probs_agg, posterior_labels_agg = aggregate_by_label_probs_mult(posterior_probs_raw, posterior_labels_raw)
        print("未发现 logits，已使用概率乘积近似聚合为对象级概率。")

    # --- 对齐（对象级）---
    prior_probs, posterior_probs, labels = align_by_label_after_aggregation(
        prior_probs_agg, prior_labels_agg,
        posterior_probs_agg, posterior_labels_agg
    )

    print(f"对齐后对象数：{len(labels)}")
    print(f"类别数：{prior_probs.shape[1]}")

    # 计算 pairwise 后验泄露
    print("计算 pairwise 后验泄露（同一对象内成对类赔率变化）...")
    pl_result = calculate_pairwise_posterior_leakage(
        prior_probs, posterior_probs,
        class_metric=class_metric
    )

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 提取数据集名称
    prior_name = os.path.basename(prior_file).replace('per_sample_probs_', '').replace('.json', '')
    posterior_name = os.path.basename(posterior_file).replace('per_sample_probs_', '').replace('.json', '')
    comparison_name = f"{prior_name}_vs_{posterior_name}"

    # 保存详细结果
    detailed_result = {
        'comparison': {
            'prior_dataset': prior_name,
            'posterior_dataset': posterior_name,
            'num_aligned_objects': int(len(labels)),
            'num_classes': int(prior_probs.shape[1]),
            'aggregation': 'logits_sum_softmax' if 'logits' in prior_data and 'logits' in posterior_data else 'prob_product_norm',
            'pairwise_total_counts': pl_result['statistics']['total_counts']
        },
        'pairwise_posterior_leakage': {
            'pairwise_pl': pl_result['pairwise_pl'],
            'statistics': pl_result['statistics']
        }
    }

    detailed_path = os.path.join(save_dir, f"{comparison_name}_pairwise_leakage_detailed.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_result, f, indent=2, ensure_ascii=False)
    print(f"✅ 详细结果已保存: {detailed_path}")

    # 创建分布图（pairwise）
    distribution_path = os.path.join(save_dir, f"{comparison_name}_pairwise_distribution.png")
    create_pl_distribution_plot(
        pl_result['pairwise_pl'],
        distribution_path,
        f"{prior_name} (prior) vs {posterior_name} (posterior)",
        violation_threshold=violation_threshold
    )

    # 生成摘要
    stats = pl_result['statistics']
    summary_path = os.path.join(save_dir, f"{comparison_name}_pairwise_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Posterior Leakage Analysis Summary (Pairwise mPL)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Prior Dataset: {prior_name}\n")
        f.write(f"Posterior Dataset: {posterior_name}\n")
        f.write(f"Aligned Objects: {len(labels)}\n")
        f.write(f"Number of Classes: {prior_probs.shape[1]}\n")
        f.write(f"Aggregation: {'logits_sum_softmax' if 'logits' in prior_data and 'logits' in posterior_data else 'prob_product_norm'}\n\n")

        f.write("Pairwise Posterior Leakage Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Counts (pairs across all objects): {stats['total_counts']}\n")
        f.write(f"Mean PL: {stats['mean_pl']:.6f}\n")
        f.write(f"Std PL: {stats['std_pl']:.6f}\n")
        f.write(f"Median PL: {stats['median_pl']:.6f}\n")
        f.write(f"Min PL: {stats['min_pl']:.6f}\n")
        f.write(f"Max PL: {stats['max_pl']:.6f}\n")

    print(f"✅ 摘要已保存: {summary_path}")

    # 打印统计信息
    print(f"\n{'='*60}")
    print("后验泄露统计 (Pairwise mPL):")
    print(f"{'='*60}")
    print(f"Counts: {stats['total_counts']}")
    print(f"平均 PL: {stats['mean_pl']:.6f}")
    print(f"标准差 PL: {stats['std_pl']:.6f}")
    print(f"中位数 PL: {stats['median_pl']:.6f}")
    print(f"最小 PL: {stats['min_pl']:.6f}")
    print(f"最大 PL: {stats['max_pl']:.6f}")
    print(f"{'='*60}\n")

    return detailed_result


# ===================== CLI =====================

if __name__ == "__main__":
    # 可配置项：修改预算、策略与阈值
    BUDGET = 0.0
    STRATEGY = "independent"

    try:
        budget_tag = f"{float(BUDGET):.1f}"
    except Exception:
        budget_tag = str(BUDGET)
    # 构建文件名
    prior_file = f"budget_{budget_tag}_{STRATEGY}_original_abstract.json"
    posterior_file = f"budget_{budget_tag}_{STRATEGY}_noise_abstract.json"
    distance_fname = f"noise_{budget_tag}_{STRATEGY}_distance_matrix.json"

    files = [prior_file, posterior_file]

    # 加载类别距离矩阵（如果存在）
    try:
        with open(distance_fname, "r", encoding="utf-8") as f:
            distance_data = json.load(f)
        class_metric = np.array(distance_data['distance_matrix'])
        print(f"✅ 加载类别距离矩阵: {class_metric.shape} ({distance_fname})")
    except FileNotFoundError:
        print(f"⚠️ 未找到类别距离矩阵: {distance_fname}，使用默认值 (全1)")
        class_metric = None

    analyze_posterior_leakage_between_datasets(
        prior_file=files[0],
        posterior_file=files[1],
        save_dir="posterior_leakage_results",
        class_metric=class_metric,
        violation_threshold=BUDGET
    )

    print("\n🎉 分析完成!")