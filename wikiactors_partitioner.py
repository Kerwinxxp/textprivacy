# -*- coding: utf-8 -*-
"""
WikiActors 数据集 PII 分组 Partition
严格按照 news_pii_partition_pipeline.py 的格式处理
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

# 导入优化版本的管道模块
import sys
sys.path.append('.')

from news_pii_partition_pipeline import (
    SpacyModelLoader, TextProcessor, OutputWriter, StatsCollector
)

# ===================== 配置 =====================
DATA_DIR = Path("./data")
INPUT_FILE = DATA_DIR / "WikiActors_50_masked_cleaned.json"
OUTPUT_FILE = Path("./Wikiactors_partition.jsonl")
OUTPUT_CSV = Path("./Wikiactors_partition.csv")
SPACY_MODEL = "en_core_web_sm"


# ===================== 主类 =====================

class WikiActorsPartitioner:
    """WikiActors 数据集 Partition 处理器"""
    
    def __init__(self, model_name: str = SPACY_MODEL):
        """初始化分区处理器"""
        print("=" * 80)
        print("🎬 WikiActors 数据集 PII 分组 Partition")
        print("=" * 80)
        
        # 加载 spaCy 模型
        print(f"\n1️⃣  加载 spaCy 模型: {model_name}")
        self.nlp = SpacyModelLoader.load(model_name)
        SpacyModelLoader.enable_senter(self.nlp)
        self.coref_enabled = SpacyModelLoader.enable_coref(self.nlp)
        print(f"   共指消解: {'✅ 启用' if self.coref_enabled else '⚠️ 未启用'}")
    
    def load_wikiactors(self, json_file: Path) -> List[Dict[str, Any]]:
        """加载 WikiActors JSON 数据"""
        try:
            print(f"\n   📂 数据路径: {json_file.absolute()}")
            
            if not json_file.exists():
                print(f"   ❌ 文件不存在: {json_file}")
                return []
            
            with open(json_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
            
            print(f"   ✅ 加载成功：{len(records)} 个演员档案")
            return records
        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            return []
    
    def process_dataset(self, json_file: Path) -> tuple:
        """处理整个 WikiActors 数据集 - 返回 JSONL 记录和统计信息"""
        records = self.load_wikiactors(json_file)
        
        if not records:
            print("❌ 没有数据可处理")
            return [], []
        
        print(f"\n2️⃣  处理 {len(records)} 个演员档案...")
        
        jsonl_records = []
        stats = StatsCollector()
        
        for idx, actor_record in enumerate(tqdm(records, desc="处理演员档案")):
            name = actor_record.get('name', 'Unknown')
            abstract = actor_record.get('original_abstract', '')
            
            # 跳过空的摘要
            if not abstract or len(abstract.strip()) < 10:
                continue
            
            try:
                # 使用 TextProcessor 处理文本 - 严格按照 pipeline 的方式
                clusters, ner_info = TextProcessor.process(
                    self.nlp, 
                    abstract, 
                    save_ner=False,  # 不保存 NER 信息
                    has_coref=self.coref_enabled
                )
                
                has_fallback = any(v.get("is_fallback", False) for v in clusters.values())
                
                # 使用 OutputWriter.write_clusters 来生成记录 - 保持格式一致
                rec = OutputWriter.write_clusters(
                    clusters, 
                    OUTPUT_FILE, 
                    idx, 
                    abstract, 
                    has_fallback, 
                    ner_info=None  # 不保存 NER
                )
                
                # 添加 actor 信息
                rec["actor_name"] = name
                
                jsonl_records.append(rec)
                
                # 更新统计
                stats.update(clusters, has_fallback, num_ner=0)
                
            except Exception as e:
                print(f"   ⚠️ 处理 {name} 失败: {e}")
                continue
        
        return jsonl_records, stats
    
    def save_results(self, jsonl_records: List[Dict]) -> None:
        """保存结果到 JSONL 和 CSV"""
        if not jsonl_records:
            print("❌ 没有记录可保存")
            return
        
        # 保存 JSONL
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                for record in jsonl_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"\n✅ JSONL 已保存到: {OUTPUT_FILE.absolute()}")
            print(f"   - 总记录数: {len(jsonl_records)}")
        except Exception as e:
            print(f"❌ 保存 JSONL 失败: {e}")
        
        # 保存 CSV - 展平结构以便查看
        try:
            csv_records = []
            for rec in jsonl_records:
                csv_rec = {
                    'row_id': rec.get('row_id'),
                    'actor_name': rec.get('actor_name', ''),
                    'text_length': rec.get('text_length'),
                    'num_persons': rec.get('num_persons'),
                    'num_pii_mentions': rec.get('num_pii_mentions'),
                    'has_fallback_anchor': rec.get('has_fallback_anchor'),
                    'persons': '|'.join(rec.get('persons', [])),
                    'pii_types': self._extract_pii_types(rec)
                }
                csv_records.append(csv_rec)
            
            df = pd.DataFrame(csv_records)
            df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
            print(f"✅ CSV 已保存到: {OUTPUT_CSV.absolute()}")
            print(f"   - 总行数: {len(csv_records)}")
            print(f"   - 总列数: {len(df.columns)}")
        except Exception as e:
            print(f"❌ 保存 CSV 失败: {e}")
    
    @staticmethod
    def _extract_pii_types(record: Dict) -> str:
        """从 clusters 中提取所有 PII 类型"""
        pii_types = set()
        for cluster in record.get('clusters', {}).values():
            pii_types.update(cluster.get('pii_types', []))
        return '|'.join(sorted(pii_types))


def main():
    """主函数"""
    
    # 初始化处理器
    partitioner = WikiActorsPartitioner(model_name=SPACY_MODEL)
    
    # 处理数据集
    jsonl_records, stats = partitioner.process_dataset(INPUT_FILE)
    
    if jsonl_records:
        # 保存结果
        partitioner.save_results(jsonl_records)
        
        # 打印统计信息
        print("\n3️⃣  处理统计:")
        stats.print_summary(save_ner=False)
        
        print("\n" + "=" * 80)
        print("✅ 处理完成!")
        print("=" * 80)
    else:
        print("❌ 没有生成结果")


if __name__ == "__main__":
    main()