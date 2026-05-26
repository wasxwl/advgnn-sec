"""Regenerate figure2_defense.png with correct MGTAB defense data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# Left: ASR under different defenses (eps=5.0)
defenses = ['No Defense', 'Adv. Train', 'SVD', 'FeatureRand. (GCN)',
            'FeatureRand. (GraphSAGE)', 'FeatureRand. (RGCN)', 'TRADES (all)']
asr_values = [100.0, 100.0, 78.3, 100.0, 96.3, 99.3, 100.0]
colors = ['#C44E52'] * len(defenses)
colors[4] = '#8172B2'

axes[0].barh(defenses, asr_values, color=colors)
axes[0].set_xlabel('ASR (%)', fontsize=11)
axes[0].set_title('Defense Effectiveness on MGTAB', fontsize=12, fontweight='bold')
axes[0].set_xlim(70, 105)
axes[0].invert_yaxis()
axes[0].grid(True, alpha=0.3, axis='x')

# Right: Clean accuracy trade-off
methods = ['Baseline', 'Adv. Train', 'FeatureRand. (GCN)', 'TRADES (Rel-GCN)']
clean_f1 = [92.7, 58.2, 70.7, 87.3]
bar_colors = ['#4C72B0', '#C44E52', '#DD8452', '#8172B2']
bars = axes[1].bar(methods, clean_f1, color=bar_colors)
axes[1].set_ylabel('Macro F1 (%)', fontsize=11)
axes[1].set_title('Clean Accuracy Trade-off', fontsize=12, fontweight='bold')
axes[1].set_ylim(0, 105)
axes[1].tick_params(axis='x', rotation=30)
for bar, val in zip(bars, clean_f1):
    axes[1].text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                 f'{val}%', ha='center', fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('paper/figure2_defense.png', dpi=150, bbox_inches='tight')
plt.savefig('paper/figure2_defense.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("figure2_defense regenerated successfully")
