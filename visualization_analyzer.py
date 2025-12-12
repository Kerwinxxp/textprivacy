# -*- coding: utf-8 -*-
"""
WikiActors PII 分组结果可视化分析 - 精简版
只生成核心的 4 个图表
"""

import json
import platform
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams


# ===================== 配置 =====================
INPUT_FILE = Path("./Wikiactors_partition.jsonl")
OUTPUT_DIR = Path("./visualization_output")


# ===================== 字体配置 =====================

class FontConfig:
    """字体配置管理器"""
    
    @staticmethod
    def setup_fonts() -> None:
        """根据系统自动配置适合的中文字体"""
        system = platform.system()
        
        font_candidates = []
        
        if system == 'Windows':
            font_candidates = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        elif system == 'Darwin':
            font_candidates = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti']
        elif system == 'Linux':
            font_candidates = ['SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
        
        available_fonts = set(rcParams['font.sans-serif'])
        selected_font = None
        
        for font in font_candidates:
            if font in available_fonts:
                selected_font = font
                break
        
        if not selected_font:
            selected_font = 'DejaVu Sans'
        
        rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
        rcParams['axes.unicode_minus'] = False
        print(f"✅ Font: {selected_font}")


FontConfig.setup_fonts()

# 设置风格
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


# ===================== 数据加载 =====================

class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load_clusters(jsonl_file: Path) -> List[Dict[str, Any]]:
        """加载 JSONL 文件"""
        records = []
        try:
            if not jsonl_file.exists():
                print(f"❌ File not found: {jsonl_file.absolute()}")
                return []
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            
            print(f"✅ Loaded: {len(records)} records from {jsonl_file.name}")
            return records
        except Exception as e:
            print(f"❌ Load failed: {e}")
            return []


# ===================== 统计分析 =====================

class StatsAnalyzer:
    """统计分析器"""
    
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records
        self.stats = {}
        self._compute_stats()
    
    def _compute_stats(self) -> None:
        """计算统计指标"""
        if not self.records:
            return
        
        # PII 类型分布
        self.stats['pii_type_distribution'] = self._compute_pii_type_dist()
        
        # 文本长度统计
        self.stats['text_length_stats'] = self._compute_text_length_stats()
    
    def _compute_pii_type_dist(self) -> Dict[str, int]:
        """计算 PII 类型分布"""
        pii_counts = defaultdict(int)
        for record in self.records:
            clusters = record.get('clusters', {})
            if isinstance(clusters, dict):
                for cluster in clusters.values():
                    for pii_type in cluster.get('pii_types', []):
                        # 计算该类型的 PII 总数
                        pii_data = cluster.get('pii', {})
                        if isinstance(pii_data, dict):
                            pii_counts[pii_type] += len(pii_data.get(pii_type, []))
                        else:
                            pii_counts[pii_type] += 1
        return dict(pii_counts)
    
    def _compute_text_length_stats(self) -> Dict[str, float]:
        """计算文本长度统计"""
        lengths = [r.get('text_length', 0) for r in self.records]
        return {
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'mean': np.mean(lengths) if lengths else 0,
            'median': np.median(lengths) if lengths else 0,
            'std': np.std(lengths) if lengths else 0,
        }


# ===================== 可视化 =====================

class Visualizer:
    """可视化器"""
    
    def __init__(self, records: List[Dict[str, Any]], analyzer: StatsAnalyzer):
        self.records = records
        self.analyzer = analyzer
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(exist_ok=True)
    
    def visualize_all(self) -> None:
        """生成所有图表"""
        print("\n🎨 Generating charts...")
        
        self.plot_text_length_distribution()
        self.plot_pii_type_distribution()
        self.plot_pii_per_person()
        self.plot_mentions_distribution()
        
        print(f"✅ All charts saved to: {self.output_dir.absolute()}")
    
    @staticmethod
    def _close_and_save(fig, output_path: Path) -> None:
        """统一的图表保存方法"""
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
            plt.close(fig)
        except Exception as e:
            print(f"⚠️ Save failed: {e}")
            plt.close(fig)
    
    def plot_text_length_distribution(self) -> None:
        """文本长度分布"""
        text_lengths = [r.get('text_length', 0) for r in self.records]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 直方图
        axes[0, 0].hist(text_lengths, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(text_lengths), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(text_lengths):.0f}')
        axes[0, 0].set_xlabel('Text Length', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Text Length Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 箱线图
        axes[0, 1].boxplot(text_lengths, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='mediumpurple', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
        axes[0, 1].set_ylabel('Text Length', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Text Length (Box Plot)', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 散点图：文本长度 vs 人物数
        persons = [r.get('num_persons', 0) for r in self.records]
        axes[1, 0].scatter(text_lengths, persons, alpha=0.6, s=50, color='teal', edgecolors='black')
        axes[1, 0].set_xlabel('Text Length', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Persons', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Text Length vs Persons', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 散点图：文本长度 vs PII 提及数
        mentions = [r.get('num_pii_mentions', 0) for r in self.records]
        axes[1, 1].scatter(text_lengths, mentions, alpha=0.6, s=50, color='orangered', edgecolors='black')
        axes[1, 1].set_xlabel('Text Length', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('PII Mentions', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Text Length vs PII Mentions', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._close_and_save(fig, self.output_dir / '1_text_length_distribution.png')
        print("  ✅ Text Length Distribution")
    
    def plot_pii_type_distribution(self) -> None:
        """PII 类型分布 - 柱状图"""
        pii_dist = self.analyzer.stats['pii_type_distribution']
        
        if not pii_dist:
            print("  ⚠️ No PII type data")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 按数量排序
        sorted_pii = dict(sorted(pii_dist.items(), key=lambda x: x[1], reverse=True))
        types = list(sorted_pii.keys())
        counts = list(sorted_pii.values())
        colors = sns.color_palette("husl", len(types))
        
        # 柱状图
        bars = ax.bar(types, counts, color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
        ax.set_xlabel('PII Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mentions', fontsize=14, fontweight='bold')
        ax.set_title('PII Type Distribution (Bar Chart)', fontsize=16, fontweight='bold', pad=20)
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        self._close_and_save(fig, self.output_dir / '2_pii_type_distribution.png')
        print("  ✅ PII Type Distribution (Bar Chart)")
    
    def plot_pii_per_person(self) -> None:
        """每个人的 PII 数分布"""
        pii_per_person = []
        
        for record in self.records:
            clusters = record.get('clusters', {})
            if isinstance(clusters, dict):
                for cluster in clusters.values():
                    # 计算每个人的 PII 提及数
                    evidence = cluster.get('evidence', [])
                    pii_count = len(evidence) if evidence else 0
                    if pii_count > 0:
                        pii_per_person.append(pii_count)
        
        if not pii_per_person:
            print("  ⚠️ No data for PII per person")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 直方图
        ax.hist(pii_per_person, bins=30, color='lightgreen', edgecolor='black', alpha=0.8, linewidth=1.5)
        ax.axvline(np.mean(pii_per_person), color='red', linestyle='--', linewidth=2.5, 
                  label=f'Mean: {np.mean(pii_per_person):.2f}')
        ax.axvline(np.median(pii_per_person), color='blue', linestyle='--', linewidth=2.5,
                  label=f'Median: {np.median(pii_per_person):.2f}')
        
        ax.set_xlabel('PII Mentions per Person', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title('PII per Person Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._close_and_save(fig, self.output_dir / '3_pii_per_person.png')
        print("  ✅ PII per Person Distribution")
    
    def plot_mentions_distribution(self) -> None:
        """PII 提及分布"""
        mentions_per_record = [r.get('num_pii_mentions', 0) for r in self.records if r.get('clusters')]
        
        if not mentions_per_record:
            print("  ⚠️ No data for mentions distribution")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 直方图
        ax.hist(mentions_per_record, bins=40, color='lightcoral', edgecolor='black', alpha=0.8, linewidth=1.5)
        ax.axvline(np.mean(mentions_per_record), color='red', linestyle='--', linewidth=2.5,
                  label=f'Mean: {np.mean(mentions_per_record):.2f}')
        ax.axvline(np.median(mentions_per_record), color='blue', linestyle='--', linewidth=2.5,
                  label=f'Median: {np.median(mentions_per_record):.2f}')
        
        ax.set_xlabel('PII Mentions per Record', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title('PII Mentions Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._close_and_save(fig, self.output_dir / '4_mentions_distribution.png')
        print("  ✅ PII Mentions Distribution")


# ===================== 主程序 =====================

def main():
    print("=" * 80)
    print("🎨 WikiActors - PII Visualization (Lite Version)")
    print("=" * 80)
    
    # 加载数据
    print(f"\n📂 Loading: {INPUT_FILE.name}")
    records = DataLoader.load_clusters(INPUT_FILE)
    
    if not records:
        print("❌ No valid data")
        return
    
    # 统计分析
    print("\n📊 Analyzing...")
    analyzer = StatsAnalyzer(records)
    
    # 打印统计摘要
    print("\n📈 Statistics Summary:")
    print(f"   - Total Records: {len(records)}")
    print(f"   - Avg Text Length: {analyzer.stats['text_length_stats']['mean']:.0f}")
    print(f"   - PII Types: {len(analyzer.stats['pii_type_distribution'])}")
    
    # 生成图表
    visualizer = Visualizer(records, analyzer)
    visualizer.visualize_all()
    
    print("\n" + "=" * 80)
    print("✅ Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()