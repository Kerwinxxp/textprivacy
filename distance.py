import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import json
import os

def extract_class_embeddings_from_classifier(tri_model_path: str, num_classes: int):
    """
    从训练好的分类器的最后一层权重提取类别嵌入
    分类器头部：Linear(hidden_dim, num_classes)
    权重形状：(num_classes, hidden_dim)
    每一行就是一个类别的嵌入向量
    """
    from transformers import AutoModelForSequenceClassification
    
    # 检查路径是否存在
    if not os.path.exists(tri_model_path):
        raise FileNotFoundError(f"模型路径不存在: {tri_model_path}")
    
    # 加载模型（使用 local_files_only=True）
    print(f"从本地路径加载模型: {tri_model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        tri_model_path,
        local_files_only=True
    )
    
    # 提取分类器头部的权重
    # 对于 DistilBERT: model.classifier.weight
    # 对于 BERT: model.classifier.weight
    # 对于 RoBERTa: model.classifier.out_proj.weight
    
    if hasattr(model, 'classifier'):
        if hasattr(model.classifier, 'weight'):
            # DistilBERT / BERT
            class_embeddings = model.classifier.weight.detach().cpu().numpy()
        elif hasattr(model.classifier, 'out_proj'):
            # RoBERTa
            class_embeddings = model.classifier.out_proj.weight.detach().cpu().numpy()
    else:
        raise AttributeError("Cannot find classifier head in model")
    
    # class_embeddings shape: (num_classes, hidden_dim)
    print(f"提取的类别嵌入形状: {class_embeddings.shape}")
    assert class_embeddings.shape[0] == num_classes, f"类别数量不匹配: {class_embeddings.shape[0]} vs {num_classes}"
    
    return class_embeddings


def compute_class_distance_matrix(class_embeddings: np.ndarray, metric: str = 'cosine'):
    """
    计算类别之间的距离矩阵
    
    Args:
        class_embeddings: (num_classes, embedding_dim)
        metric: 'cosine', 'euclidean', 'manhattan'
    
    Returns:
        distance_matrix: (num_classes, num_classes)
    """
    if metric == 'cosine':
        # 余弦距离 = 1 - 余弦相似度
        distance_matrix = cosine_distances(class_embeddings)
    elif metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        distance_matrix = euclidean_distances(class_embeddings)
    elif metric == 'manhattan':
        from sklearn.metrics.pairwise import manhattan_distances
        distance_matrix = manhattan_distances(class_embeddings)
    else:
        raise ValueError(f"不支持的距离度量: {metric}")
    
    # 确保对角线为一个小值（避免除以0）
    np.fill_diagonal(distance_matrix, 1e-12)
    
    print(f"距离矩阵统计:")
    print(f"  均值: {distance_matrix.mean():.4f}")
    print(f"  标准差: {distance_matrix.std():.4f}")
    print(f"  最小值: {distance_matrix.min():.4f}")
    print(f"  最大值: {distance_matrix.max():.4f}")
    
    return distance_matrix


def save_class_metric(distance_matrix: np.ndarray, label_to_name: dict, save_path: str):
    """
    保存类别距离矩阵和元数据
    """
    data = {
        'distance_matrix': distance_matrix.tolist(),
        'label_to_name': label_to_name,
        'shape': list(distance_matrix.shape),
        'metric': 'cosine'
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 类别距离矩阵已保存到: {save_path}")


def load_class_metric(save_path: str):
    """
    加载类别距离矩阵
    """
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    distance_matrix = np.array(data['distance_matrix'])
    label_to_name = {int(k): v for k, v in data['label_to_name'].items()}
    
    print(f"✅ 从 {save_path} 加载类别距离矩阵")
    print(f"   形状: {distance_matrix.shape}")
    print(f"   度量: {data.get('metric', 'unknown')}")
    
    return distance_matrix, label_to_name


if __name__ == "__main__":
    # 配置（使用正斜杠或原始字符串）
    tri_model_path = "outputs/WikiActors/noise_3"
    num_classes = 50
    output_path = "class_distance_matrix.json"
    
    # 1. 提取类别嵌入
    print("=" * 80)
    print("提取类别嵌入...")
    class_embeddings = extract_class_embeddings_from_classifier(tri_model_path, num_classes)
    
    # 2. 计算距离矩阵
    print("\n" + "=" * 80)
    print("计算类别距离矩阵...")
    distance_matrix = compute_class_distance_matrix(class_embeddings, metric='cosine')
    
    # 3. 加载 label_to_name 映射（从你的 TRI 对象中）
    # 如果没有现成的，可以从数据文件中提取
    print("\n" + "=" * 80)
    print("加载 label_to_name 映射...")
    
    # 方法1：从 TRI 对象加载（如果可用）
    try:
        from tri import TRI
        with open("config.json", "r") as f:
            config = json.load(f)
        tri = TRI(**config)
        tri.run_data(verbose=False)
        label_to_name = tri.label_to_name
    except Exception as e:
        print(f"无法从 TRI 对象加载: {e}")
        # 方法2：从数据文件中提取
        print("尝试从数据文件提取...")
        import pandas as pd
        data_file = "data/WikiActors_50_eval_with_noise.json"
        df = pd.read_json(data_file)
        unique_names = df['name'].unique()
        label_to_name = {i: name for i, name in enumerate(sorted(unique_names))}
        print(f"从数据文件提取了 {len(label_to_name)} 个类别")
    
    # 4. 保存
    print("\n" + "=" * 80)
    save_class_metric(distance_matrix, label_to_name, output_path)
    
    # 5. 可视化前10个类别的距离
    print("\n" + "=" * 80)
    print("前10个类别之间的距离矩阵:")
    print(distance_matrix[:10, :10])
    
    # 6. 显示距离最近和最远的类别对
    print("\n" + "=" * 80)
    print("距离最近的5对类别:")
    np.fill_diagonal(distance_matrix, np.inf)  # 排除自己
    for _ in range(5):
        min_idx = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
        i, j = min_idx
        dist = distance_matrix[i, j]
        print(f"  {label_to_name.get(i, i)} <-> {label_to_name.get(j, j)}: {dist:.4f}")
        distance_matrix[i, j] = np.inf
        distance_matrix[j, i] = np.inf