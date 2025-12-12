import pandas as pd
import spacy
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re
from typing import List, Dict, Any, Tuple
import time
from tqdm import tqdm

# 加载spaCy NER模型（用于实体识别）
nlp = spacy.load("en_core_web_lg")

# 加载Transformer模型（参考TRI代码中的DistilBERT）
model_name = "distilbert-base-uncased"  # 或从config.json中读取
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()  # 设置为评估模式

# 数据文件路径
data_file = r"c:\Users\phdwf\Desktop\textreidentify\TextReIdentification\data\WikiActors_50_eval.json"
output_file = r"c:\Users\phdwf\Desktop\textreidentify\TextReIdentification\data\WikiActors_50_eval_with_noisy_public_knowledge.json"

# 加载数据
df = pd.read_json(data_file)

class SimpleExponentialMechanism:
    """
    简化的指数机制，用于token加噪（基于embedding距离）
    """
    def __init__(self, model, tokenizer, epsilon=1.0, candidate_pool_size=500):
        self.model = model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.candidate_pool_size = candidate_pool_size
        self.vocab_size = tokenizer.vocab_size
        
        # 预加载所有token embedding（优化）
        with torch.no_grad():
            all_token_ids = torch.arange(self.vocab_size)
            self.all_embeddings = model.get_input_embeddings()(all_token_ids).detach().cpu().numpy()
    
    def get_token_embedding(self, token: str) -> np.ndarray:
        """获取token的embedding"""
        inputs = tokenizer(token, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding
    
    def select_noisy_token(self, original_embedding: np.ndarray) -> Tuple[str, np.ndarray]:
        """使用指数机制选择噪声token，并返回概率分布"""
        # 随机选择候选池
        candidate_ids = np.random.choice(self.vocab_size, size=self.candidate_pool_size, replace=False)
        candidate_embeddings = self.all_embeddings[candidate_ids]
        
        # 计算距离
        distances = np.linalg.norm(candidate_embeddings - original_embedding, axis=1)
        
        # 指数机制：计算概率
        scaled_distances = -(self.epsilon / 2) * distances
        exp_values = np.exp(scaled_distances - np.max(scaled_distances))
        probabilities = exp_values / np.sum(exp_values)
        
        # 采样
        selected_idx = np.random.choice(len(candidate_ids), p=probabilities)
        selected_token_id = candidate_ids[selected_idx]
        selected_token = tokenizer.decode([selected_token_id])
        
        return selected_token, probabilities

# 初始化指数机制
mechanism = SimpleExponentialMechanism(model, tokenizer, epsilon=1.0)  # ε可调

# 全局计数器，用于打印前3个实体
entity_count = 0

# 处理每一行，添加noisy_public_knowledge
def process_row(row):
    global entity_count
    text = row['public_knowledge']  # 对background_knowledge_column加噪
    doc = nlp(text)
    new_text = text
    
    # 识别实体并替换
    for ent in doc.ents:
        entity_text = ent.text
        
        # 对实体进行tokenize（按token加噪）
        tokens = tokenizer.tokenize(entity_text)
        noisy_tokens = []
        probabilities_list = []
        
        for token in tokens:
            token_embedding = mechanism.get_token_embedding(token)
            noisy_token, probabilities = mechanism.select_noisy_token(token_embedding)
            noisy_tokens.append(noisy_token)
            probabilities_list.append(probabilities)
        
        # 重新组合噪声token
        noisy_entity = tokenizer.convert_tokens_to_string(noisy_tokens)
        
        # 打印前3个实体的信息
        if entity_count < 3:
            print(f"\n🔍 Entity {entity_count + 1}:")
            print(f"  Original entity: '{entity_text}'")
            print(f"  Tokens: {tokens}")
            print(f"  Noisy entity: '{noisy_entity}'")
            print(f"  Noisy tokens: {noisy_tokens}")
            if probabilities_list:
                print(f"  Probability distribution for first token (first 10): {probabilities_list[0][:10]}")
            entity_count += 1
        
        # 精确替换实体
        new_text = re.sub(r'\b' + re.escape(entity_text) + r'\b', re.escape(noisy_entity), new_text)
    
    return new_text

# 记录开始时间
start_time = time.time()

# 应用处理（使用tqdm显示进度）
print("开始加噪处理...")
noisy_list = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    noisy_text = process_row(row)
    noisy_list.append(noisy_text)

df['noisy_public_knowledge'] = noisy_list

# 记录结束时间
end_time = time.time()
total_time = end_time - start_time

# 保存更新后的数据
df.to_json(output_file, orient='records')
print(f"添加 noisy_public_knowledge 列完成，保存到 {output_file}")
print(f"总处理时间: {total_time:.2f} 秒")

# 打印第一行的对比
print("\n第一行数据对比：")
print(f"  Name: {df.loc[0, 'name']}")
print(f"  Original Public Knowledge: {df.loc[0, 'public_knowledge']}")
print(f"  Noisy Public Knowledge: {df.loc[0, 'noisy_public_knowledge']}")