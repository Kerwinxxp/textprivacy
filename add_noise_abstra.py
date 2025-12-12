import pandas as pd
import spacy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re
import random
from typing import List, Dict, Any, Tuple

# 1. 设置随机种子以保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 2. 加载spaCy NER模型（用于实体识别）
print("正在加载 spaCy 模型...")
nlp = spacy.load("en_core_web_lg")

# 3. 加载Transformer模型
print("正在加载 DistilBERT 模型...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model.to(device)
model.eval()

# 设置每行数据的总隐私预算
TOTAL_EPSILON_BUDGET = 30
# 预算分配策略: 'shared' (均分) 或 'independent' (独立)
BUDGET_ALLOCATION_STRATEGY = 'independent' 

# 数据文件路径
data_file = r"C:\Users\phdwf\OneDrive\Desktop\textreidentify\TextReIdentification\data\WikiActors_50_masked_cleaned.json"
# 动态生成输出文件名
output_file = fr"C:\Users\phdwf\OneDrive\Desktop\textreidentify\TextReIdentification\data\noise_budget_{TOTAL_EPSILON_BUDGET}_{BUDGET_ALLOCATION_STRATEGY}.json"

# 加载数据
df = pd.read_json(data_file)

class SimpleExponentialMechanism:
    """
    改进的指数机制：使用 Top-K 最近邻作为候选池
    """
    def __init__(self, model, tokenizer, candidate_pool_size=50):
        self.model = model
        self.tokenizer = tokenizer
        # self.epsilon = epsilon  <-- 移除固定的 epsilon
        self.candidate_pool_size = candidate_pool_size
        self.vocab_size = tokenizer.vocab_size
        self.device = model.device
        
        print("正在预加载全词表 Embeddings...")
        # 预加载所有token embedding，并转为 Tensor 放在设备上以加速计算
        with torch.no_grad():
            all_token_ids = torch.arange(self.vocab_size).to(self.device)
            # 获取静态 embedding table
            self.all_embeddings = model.get_input_embeddings()(all_token_ids)
    
    # def get_token_embedding(self, token: str) -> torch.Tensor:
    #     """获取token的embedding (返回 Tensor)"""
    #     inputs = tokenizer(token, return_tensors="pt", add_special_tokens=False).to(self.device)
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #         # 获取 embedding (取平均或直接取第一个token)
    #         embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    #     return embedding
    def get_token_embedding(self, token: str) -> torch.Tensor:
        """
        修正版：直接获取 Input Embedding，而不是模型输出的 Hidden State
        """
        # 1. 将 token string (如 "##lie") 转回 token id
        # 注意：这里直接使用 convert_tokens_to_ids，它能正确处理 "##" 前缀
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        
        # 处理未知词的情况（以防万一）
        if token_id == self.tokenizer.unk_token_id and token != self.tokenizer.unk_token:
            print(f"Warning: Token '{token}' unknown to tokenizer.")
            
        # 2. 转为 Tensor
        token_id_tensor = torch.tensor(token_id).to(self.device)
        
        # 3. 直接从预加载的 all_embeddings 中查表
        # 或者使用 self.model.get_input_embeddings()(token_id_tensor)
        # 这里为了利用你已经加载的 self.all_embeddings，直接索引即可
        embedding = self.all_embeddings[token_id_tensor]
        
        return embedding
    def select_noisy_token(self, original_embedding: torch.Tensor, epsilon: float) -> Tuple[str, np.ndarray]:
        """
        1. 计算原词与全词表的距离
        2. 选出 Top-K 个最近的词作为候选池
        3. 在这 K 个词中应用指数机制
        """
        # 1. 计算与所有词的欧氏距离 (利用 PyTorch 广播机制加速)
        # diff shape: [vocab_size, hidden_dim]
        diff = self.all_embeddings - original_embedding
        # distances shape: [vocab_size]
        distances = torch.norm(diff, p=2, dim=1)
        
        # 2. 筛选 Top-K (选出距离最小的 candidate_pool_size 个)
        # values 是距离，indices 是 token_id
        topk_distances, topk_indices = torch.topk(distances, self.candidate_pool_size, largest=False)
        
        # 转回 numpy 进行概率计算
        candidate_ids = topk_indices.cpu().numpy()
        candidate_distances = topk_distances.cpu().numpy()
        
        # 3. 指数机制：计算概率
        # score = -distance (距离越小，分数越高)
        # P(t) = exp(epsilon * score / 2)
        # 使用传入的动态 epsilon
        scaled_scores = -(epsilon / 2) * candidate_distances
        
        # 数值稳定性处理：减去最大值防止 exp 溢出
        exp_values = np.exp(scaled_scores - np.max(scaled_scores))
        probabilities = exp_values / np.sum(exp_values)
        
        # 4. 采样
        selected_idx = np.random.choice(len(candidate_ids), p=probabilities)
        selected_token_id = candidate_ids[selected_idx]
        
        # 解码
        selected_token = tokenizer.decode([selected_token_id])
        
        return selected_token, probabilities

# 初始化指数机制 (不再传入固定的 epsilon)
mechanism = SimpleExponentialMechanism(model, tokenizer, candidate_pool_size=50)

# 全局计数器，用于打印前3个实体
entity_count = 0

# 处理每一行，添加noise_abstract
def process_row(row):
    global entity_count
    text = row['original_abstract']
    doc = nlp(text)
    new_text = text

    # --- 统计逻辑 ---
    all_entity_texts = [ent.text for ent in doc.ents]
    total_ent_count = len(all_entity_texts)
    unique_ent_count = len(set(all_entity_texts))
    
    # --- 动态预算分配 ---
    if BUDGET_ALLOCATION_STRATEGY == 'shared':
        # 如果有实体，将总预算均分给每个实体
        if total_ent_count > 0:
            current_epsilon = TOTAL_EPSILON_BUDGET / total_ent_count
        else:
            current_epsilon = TOTAL_EPSILON_BUDGET 
    elif BUDGET_ALLOCATION_STRATEGY == 'independent':
        # 每个实体独立加噪分配，使用完整预算
        current_epsilon = TOTAL_EPSILON_BUDGET
    else:
        raise ValueError(f"Unknown strategy: {BUDGET_ALLOCATION_STRATEGY}")
    # ----------------

    # 识别实体并替换
    for ent in doc.ents:
        entity_text = ent.text
        
        # 对实体进行tokenize（按token加噪）
        tokens = tokenizer.tokenize(entity_text)
        noisy_tokens = []
        probabilities_list = []
        
        for token in tokens:
            # 获取 embedding (Tensor)
            token_embedding = mechanism.get_token_embedding(token)
            # 选择噪声词，传入计算好的 current_epsilon
            noisy_token, probabilities = mechanism.select_noisy_token(token_embedding, epsilon=current_epsilon)
            
            # 清理 token (去除 BERT 的 subword 前缀 '##')
            clean_noisy_token = noisy_token.replace("##", "")
            noisy_tokens.append(clean_noisy_token)
            probabilities_list.append(probabilities)
        
        # 重新组合噪声token
        noisy_entity = " ".join(noisy_tokens)
        noisy_entity = noisy_entity.replace(" .", ".").replace(" ,", ",")
        
        # 打印前3个实体的信息
        if entity_count < 3:
            print(f"\n🔍 Entity {entity_count + 1}:")
            print(f"  Original entity: '{entity_text}'")
            print(f"  Budget: {current_epsilon:.4f} (Total: {TOTAL_EPSILON_BUDGET} / {total_ent_count} entities)")
            print(f"  Tokens: {tokens}")
            print(f"  Noisy entity: '{noisy_entity}'")
            if probabilities_list:
                max_prob_idx = np.argmax(probabilities_list[0])
                print(f"  Max prob for 1st token: {probabilities_list[0][max_prob_idx]:.4f}")
            entity_count += 1
        
        # 替换逻辑
        new_text = re.sub(r'\b' + re.escape(entity_text) + r'\b', noisy_entity, new_text)
    
    # 返回三个值
    return new_text, total_ent_count, unique_ent_count

print("开始处理数据...")
# 应用处理，result_type='expand' 将元组拆分为多列
df[['noise_abstract', 'entity_count', 'unique_entity_count']] = df.apply(process_row, axis=1, result_type='expand')

# 保存更新后的数据
df.to_json(output_file, orient='records')
print(f"处理完成，保存到 {output_file}")

# 打印统计信息
print("\n=== 统计报告 ===")
print(f"平均每行实体数 (Total Entities): {df['entity_count'].mean():.2f}")
print(f"平均每行唯一实体数 (Unique Entities): {df['unique_entity_count'].mean():.2f}")
print(f"最多实体的行包含: {df['entity_count'].max()} 个实体")
print(f"最少实体的行包含: {df['entity_count'].min()} 个实体")

# 打印第一行的对比
print("\n第一行数据对比：")
if not df.empty:
    print(f"  Name: {df.loc[0, 'name']}")
    print(f"  Entity Count: {df.loc[0, 'entity_count']}")
    print(f"  Unique Entity Count: {df.loc[0, 'unique_entity_count']}")
    print(f"  Original Abstract: {df.loc[0, 'original_abstract']}")
    print(f"  Noise Abstract: {df.loc[0, 'noise_abstract']}")