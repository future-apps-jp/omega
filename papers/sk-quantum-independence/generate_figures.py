"""
論文用 Multiway Graph 図の生成
==============================

Figure 1: (K a b) (K c d) のダイヤモンド構造
Figure 2: S (K a) (K b) c のより複雑な構造
"""

import sys
sys.path.insert(0, '/home/hkohashi/research/sk-quantum/phase0')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from collections import deque

from sk_parser import parse, to_string
from multiway import build_multiway_graph
from reduction import RedexType

# Academic style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def format_expr(expr_str: str) -> str:
    """式をLaTeX用にフォーマット"""
    # 変数名をイタリックに
    result = expr_str.replace('(', '(').replace(')', ')')
    return result


def draw_multiway_graph(expr_str: str, output_path: str, title: str = None):
    """
    Multiway Graphをアカデミック品質で描画
    """
    expr = parse(expr_str)
    graph = build_multiway_graph(expr, max_depth=10)
    
    # ノード位置を計算（層ごとに配置）
    positions = {}
    node_depths = {}
    
    # BFSで深さを計算
    queue = deque([(graph.root, 0)])
    visited = {graph.root.node_id}
    depth_nodes = {}
    
    while queue:
        node, depth = queue.popleft()
        node_depths[node.node_id] = depth
        if depth not in depth_nodes:
            depth_nodes[depth] = []
        depth_nodes[depth].append(node)
        
        for child in node.children.values():
            if child.node_id not in visited:
                visited.add(child.node_id)
                queue.append((child, depth + 1))
    
    # 位置を計算
    max_depth = max(depth_nodes.keys()) if depth_nodes else 0
    for depth, nodes in depth_nodes.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (n-1)/2) * 3.0
            y = -depth * 2.0
            positions[node.node_id] = (x, y)
    
    # Figure作成
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 辺を描画
    for node in graph.nodes.values():
        if node.node_id in positions:
            x1, y1 = positions[node.node_id]
            for redex_path, child in node.children.items():
                if child.node_id in positions:
                    x2, y2 = positions[child.node_id]
                    
                    # 辺の種類を取得
                    redex = node.redex_info.get(redex_path)
                    if redex:
                        color = '#2E86AB' if redex.type == RedexType.S_REDEX else '#A23B72'
                        label = 'S' if redex.type == RedexType.S_REDEX else 'K'
                    else:
                        color = '#666666'
                        label = '?'
                    
                    # 矢印を描画
                    ax.annotate('', xy=(x2, y2+0.4), xytext=(x1, y1-0.4),
                               arrowprops=dict(arrowstyle='->', color=color, 
                                             lw=2.0, shrinkA=0, shrinkB=0,
                                             connectionstyle='arc3,rad=0'))
                    
                    # ラベルを描画（辺の中点）
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    # 左右にオフセット
                    offset_x = 0.2 if x2 >= x1 else -0.2
                    ax.text(mx + offset_x, my, f'{label}-red', fontsize=9, color=color, 
                           fontweight='bold', ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                    edgecolor='none', alpha=0.8))
    
    # ノードを描画
    for node_id, (x, y) in positions.items():
        node = graph.nodes[node_id]
        expr_text = to_string(node.expr)
        
        # テキストを短縮
        if len(expr_text) > 25:
            expr_text = expr_text[:22] + '...'
        
        # ノードの色
        if node.is_terminal:
            facecolor = '#E8F5E9'
            edgecolor = '#2E7D32'
            linewidth = 2.5
        elif node == graph.root:
            facecolor = '#E3F2FD'
            edgecolor = '#1565C0'
            linewidth = 2.5
        else:
            facecolor = '#FAFAFA'
            edgecolor = '#616161'
            linewidth = 1.5
        
        # ボックスのサイズを計算
        box_width = max(len(expr_text) * 0.12, 1.5)
        
        # ボックスを描画
        bbox = FancyBboxPatch((x - box_width/2, y - 0.3), box_width, 0.6,
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor=facecolor, edgecolor=edgecolor, 
                              linewidth=linewidth, zorder=10)
        ax.add_patch(bbox)
        
        # テキストを描画
        ax.text(x, y, expr_text, fontsize=9, ha='center', va='center',
               zorder=11, family='monospace', fontweight='medium')
    
    # 凡例（グラフの外側に配置）
    s_patch = mpatches.Patch(color='#2E86AB', label='S-reduction: $Sxyz \\to (xz)(yz)$')
    k_patch = mpatches.Patch(color='#A23B72', label='K-reduction: $Kxy \\to x$')
    start_patch = mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1565C0', 
                                  label='Initial expression', linewidth=2)
    end_patch = mpatches.Patch(facecolor='#E8F5E9', edgecolor='#2E7D32', 
                                label='Normal form (terminal)', linewidth=2)
    ax.legend(handles=[s_patch, k_patch, start_patch, end_patch], 
             loc='upper left', fontsize=9, framealpha=0.95,
             edgecolor='#CCCCCC', fancybox=True,
             bbox_to_anchor=(1.02, 1.0))
    
    # タイトル
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # 余白調整
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    margin_x = 2.0
    margin_y = 1.0
    ax.set_xlim(min(all_x) - margin_x, max(all_x) + margin_x)
    ax.set_ylim(min(all_y) - margin_y, max(all_y) + margin_y)
    
    # 保存
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, format='png', bbox_inches='tight')
    print(f"  PDF: {output_path}")
    print(f"  PNG: {png_path}")
    plt.close()


def main():
    output_dir = '/home/hkohashi/research/papers/sk-quantum-independence/figures'
    
    print("=== 論文用図の生成 ===\n")
    
    # Figure 1: Simple diamond structure
    print("Figure 1: Diamond structure (K a b) (K c d)")
    draw_multiway_graph(
        "(K a b) (K c d)",
        f"{output_dir}/figure1_diamond.pdf",
        title="Multiway graph for $(K\\,a\\,b)\\,(K\\,c\\,d)$"
    )
    print()
    
    # Figure 2: S combinator structure
    print("Figure 2: S combinator S (K a) (K b) c")
    draw_multiway_graph(
        "S (K a) (K b) c",
        f"{output_dir}/figure2_s_combinator.pdf",
        title="Multiway graph for $S\\,(K\\,a)\\,(K\\,b)\\,c$"
    )
    print()
    
    print("=== 図の生成完了 ===")


if __name__ == '__main__':
    main()

