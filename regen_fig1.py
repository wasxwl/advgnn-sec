"""Regenerate figure1_attack_mgtab.png with correct MGTAB attack data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42

fig, ax = plt.subplots(figsize=(7, 4.2))

models = ['Rel-GCN', 'Rel-GAT', 'GraphSAGE', 'RGCN']
asr_05  = [45.4, 56.7, 78.5, 47.4]
asr_10  = [85.6, 91.5, 95.2, 86.0]
asr_50  = [100.0, 100.0, 100.0, 100.0]

x = np.arange(len(models))
width = 0.25

ax.bar(x - width, asr_05, width, label=r'$\varepsilon = 0.5$', color='#4C72B0')
ax.bar(x, asr_10, width, label=r'$\varepsilon = 1.0$', color='#DD8452')
ax.bar(x + width, asr_50, width, label=r'$\varepsilon = 5.0$', color='#C44E52')

ax.set_ylabel('Attack Success Rate (%)', fontsize=13)
ax.set_ylim(0, 110)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_title('MGTAB: Semantic Feature Attack Effectiveness', fontsize=14, fontweight='bold', pad=30)
ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True)
ax.grid(True, alpha=0.3, axis='y')

for i, vals in enumerate([asr_05, asr_10, asr_50]):
    for j, v in enumerate(vals):
        ax.text(j + (i - 1) * width, v + 1.5, f'{v:.1f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('paper/figure1_attack_mgtab.png', dpi=150, bbox_inches='tight')
plt.savefig('paper/figure1_attack_mgtab.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("figure1_attack_mgtab regenerated successfully (no watermark)")
