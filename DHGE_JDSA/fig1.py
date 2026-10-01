import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42})
fig, ax = plt.subplots(figsize=(7.0, 3.0)); ax.set_xlim(0, 10.6); ax.set_ylim(0.2, 3.8); ax.axis('off')
def box(x, y, w, h, label, sub='', fc='#dbe9f9', ec='#2779c4', fs=7.0):
    ax.add_patch(mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle='round,pad=0.06', fc=fc, ec=ec, lw=1.2))
    ax.text(x, y+(0.11 if sub else 0), label, ha='center', va='center', fontsize=fs, fontweight='bold', color='#1a3a5c')
    if sub: ax.text(x, y-0.2, sub, ha='center', va='center', fontsize=6.2, color='#444')
def arrow(x1, y1, x2, y2, **k):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='#555', lw=1.1, **k))
# snapshot stack
for i, dy in enumerate([0.24, 0.12, 0.0]):
    box(0.75+dy*0.4, 2.2+dy, 1.1, 0.72, '', fc='#f4eef9', ec='#7c4daa')
ax.text(0.75, 2.26, r'$G_{\tau_1}\!\subseteq\!\cdots\!\subseteq\!G_{\tau_K}$', ha='center', fontsize=7)
ax.text(0.75, 2.02, 'growing cascade', ha='center', fontsize=6.2, color='#444')
arrow(1.38, 2.2, 1.78, 2.2)
box(2.3, 2.2, 0.95, 0.72, r'exp$_0$ lift', 'posts + arrival\ntime', fc='#fffbe6', ec='#b8860b')
arrow(2.78, 2.2, 3.12, 2.2)
box(3.65, 2.2, 0.95, 0.72, 'HypGAT L1', 'curvature\nattention')
arrow(4.13, 2.2, 4.47, 2.2)
box(5.0, 2.2, 0.95, 0.72, 'HypGAT L2', 'curvature\nattention')
ax.text(5.0, 2.72, r'$h^{(2)}_v$', ha='center', fontsize=7.5, color='#1a3a5c')
# both heads from L2 output
arrow(5.48, 2.35, 5.95, 2.9); arrow(5.48, 2.05, 5.95, 1.5)
box(6.45, 2.95, 0.95, 0.62, r'log$_0$ pool', r'$\mathbf{g}_k$', fc='#fffbe6', ec='#b8860b')
box(6.45, 1.45, 0.95, 0.62, 'risk score', r'$r_k=\overline{\|h^{(2)}_v\|}$', fc='#fdecea', ec='#c62828')
arrow(6.93, 2.95, 7.35, 2.35); arrow(6.93, 1.45, 7.35, 2.05)
box(7.85, 2.2, 0.95, 0.72, 'GRU', r'state $\mathbf{s}_k$ over $k$', fc='#eef7f7', ec='#00796b')
arrow(8.33, 2.2, 8.72, 2.2)
box(9.25, 2.2, 0.95, 0.72, 'MLP', r'$[\mathbf{s}_K;\mathbf{g}_K;r_K]$', fc='#e8f5e9', ec='#2e7d32')
arrow(9.73, 2.2, 10.05, 2.2); ax.text(10.33, 2.2, 'NR/TR\nFR/UR', ha='center', va='center', fontsize=7)
ax.plot([0.12, 7.05, 7.05, 0.12, 0.12], [0.85, 0.85, 3.4, 3.4, 0.85], ls=':', lw=0.8, color='#999')
ax.text(3.6, 0.62, r'shared weights, applied to every snapshot $G_{\tau_k}$ ($k=1,\ldots,K$)', ha='center', fontsize=6.5, color='#555')
fig.tight_layout(); fig.savefig('Fig1.pdf', bbox_inches='tight'); fig.savefig('Fig1.png', dpi=110, bbox_inches='tight')
