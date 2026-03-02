#!/usr/bin/env python3
"""生成论文技术架构图（专业级）"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import numpy as np

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

OUT = 'thesis_figures'

# ============ 通用绘制函数 ============
def draw_box(ax, x, y, w, h, text, color='#E3F2FD', edge='#1565C0',
             fontsize=9, fontweight='bold', textcolor='#1A237E', style='round',
             linewidth=1.5, subtext=None, subtextsize=7):
    """绘制圆角矩形模块"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.1" if style == 'round' else f"square,pad=0.05",
                          facecolor=color, edgecolor=edge,
                          linewidth=linewidth, zorder=3)
    ax.add_patch(box)
    if subtext:
        ax.text(x, y + 0.15, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=textcolor, zorder=4)
        ax.text(x, y - 0.2, subtext, ha='center', va='center',
                fontsize=subtextsize, color='#555', zorder=4)
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color=textcolor, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color='#333', style='->', lw=1.5, label=None):
    """绘制箭头"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.15, my, label, fontsize=7, color='#666',
                ha='left', va='center', zorder=4)


# ========== 图A: 系统整体流水线架构 ==========
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1, 7)
    ax.axis('off')

    # 标题
    ax.text(7.5, 6.5, '基于头部姿态估计的口岸人员可疑行为识别系统架构',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#1A237E')

    # ---- 阶段1: 输入 ----
    draw_box(ax, 1.2, 5, 2.0, 1.0, '原始监控视频', '#FFF3E0', '#E65100',
             textcolor='#BF360C', subtext='1920×1080 @30fps')

    # ---- 阶段2: 检测+跟踪 ----
    draw_box(ax, 4.5, 5, 2.0, 0.7, 'YOLOv8', '#E8F5E9', '#2E7D32',
             textcolor='#1B5E20', subtext='行人检测')
    draw_box(ax, 7.0, 5, 2.2, 0.7, 'StrongSORT', '#E8F5E9', '#2E7D32',
             textcolor='#1B5E20', subtext='多目标跟踪 + Re-ID')

    draw_arrow(ax, 2.2, 5, 3.5, 5)
    draw_arrow(ax, 5.5, 5, 5.9, 5)

    # 阶段标签
    ax.add_patch(FancyBboxPatch((3.2, 5.8), 6.2, 0.4,
                                 boxstyle="round,pad=0.1", facecolor='#C8E6C9',
                                 edgecolor='#2E7D32', linewidth=1, alpha=0.6))
    ax.text(6.3, 6.0, '阶段一：多目标检测与跟踪', ha='center', fontsize=9,
            fontweight='bold', color='#1B5E20')

    # ---- 阶段3: 头部姿态估计 (双路径) ----
    # 主路径
    draw_box(ax, 1.5, 3.0, 2.0, 0.7, 'SSD 人脸检测', '#E3F2FD', '#1565C0',
             textcolor='#0D47A1', subtext='conf_thresh=0.45')
    draw_box(ax, 4.2, 3.0, 2.0, 0.7, '头部框扩展', '#E3F2FD', '#1565C0',
             textcolor='#0D47A1', subtext='上60% 下15% 侧35%')
    draw_box(ax, 7.3, 3.0, 2.2, 0.7, 'WHENet', '#E3F2FD', '#1565C0',
             textcolor='#0D47A1', subtext='头部姿态估计 (ONNX)')

    draw_arrow(ax, 7.0, 4.6, 1.5, 3.4, label='人体ROI')
    draw_arrow(ax, 2.5, 3.0, 3.2, 3.0)
    draw_arrow(ax, 5.2, 3.0, 6.2, 3.0)

    # Fallback 路径
    draw_box(ax, 1.5, 1.5, 2.2, 0.7, '人体框估计头部', '#FFF8E1', '#F57F17',
             textcolor='#E65100', subtext='h=22% w=55% 人体比例')

    # 判断节点
    diamond_x, diamond_y = 3.2, 2.2
    diamond = plt.Polygon([[diamond_x, diamond_y+0.35], [diamond_x+0.5, diamond_y],
                            [diamond_x, diamond_y-0.35], [diamond_x-0.5, diamond_y]],
                           facecolor='#FFECB3', edgecolor='#F57F17', linewidth=1.5, zorder=3)
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y, '检测\n成功?', ha='center', va='center',
            fontsize=7, fontweight='bold', color='#E65100', zorder=4)

    draw_arrow(ax, 1.5, 2.65, diamond_x - 0.3, diamond_y + 0.25, color='#1565C0')
    ax.text(2.2, 2.55, '是', fontsize=7, color='#2E7D32', fontweight='bold')
    draw_arrow(ax, diamond_x, diamond_y - 0.35, 1.5, 1.85, color='#F57F17')
    ax.text(2.3, 1.75, '否', fontsize=7, color='#E65100', fontweight='bold')
    draw_arrow(ax, 2.6, 1.5, 6.2, 2.7, color='#F57F17')
    draw_arrow(ax, 4.2, 2.65, 6.2, 2.85, color='#1565C0')

    # 阶段标签
    ax.add_patch(FancyBboxPatch((0.2, 3.8), 8.5, 0.4,
                                 boxstyle="round,pad=0.1", facecolor='#BBDEFB',
                                 edgecolor='#1565C0', linewidth=1, alpha=0.6))
    ax.text(4.45, 4.0, '阶段二：双路径头部姿态估计（创新点二：Fallback容错机制）',
            ha='center', fontsize=9, fontweight='bold', color='#0D47A1')

    # 输出 yaw/pitch/roll
    draw_box(ax, 9.8, 3.0, 1.5, 0.7, 'yaw, pitch, roll', '#F3E5F5', '#6A1B9A',
             textcolor='#4A148C', subtext='三维姿态角')
    draw_arrow(ax, 8.4, 3.0, 9.05, 3.0)

    # ---- 阶段4: 行为识别 (三级混合) ----
    draw_box(ax, 10.0, 5.0, 1.8, 0.65, '姿态门控', '#FCE4EC', '#C62828',
             textcolor='#B71C1C', subtext='|yaw|>40° pitch>28°')
    draw_box(ax, 12.2, 5.0, 1.8, 0.65, 'Transformer', '#F3E5F5', '#6A1B9A',
             textcolor='#4A148C', subtext='时序分类 (seq=90)')
    draw_box(ax, 14.2, 5.0, 1.6, 0.65, '规则检测器', '#FFF8E1', '#F57F17',
             textcolor='#E65100', subtext='V形/角速度')

    draw_arrow(ax, 9.8, 3.35, 10.0, 4.65)
    draw_arrow(ax, 10.0, 3.35, 12.2, 4.65)

    draw_box(ax, 12.2, 3.5, 2.0, 0.65, '混合决策融合', '#FFCDD2', '#C62828',
             textcolor='#B71C1C', subtext='门控→模型→规则')
    draw_arrow(ax, 10.0, 4.65, 11.8, 3.85, color='#C62828')
    draw_arrow(ax, 12.2, 4.65, 12.2, 3.85, color='#6A1B9A')
    draw_arrow(ax, 14.2, 4.65, 12.6, 3.85, color='#F57F17')

    draw_box(ax, 12.2, 2.2, 2.0, 0.65, '时序平滑投票', '#E8EAF6', '#283593',
             textcolor='#1A237E', subtext='滑动窗口 w=8')
    draw_arrow(ax, 12.2, 3.15, 12.2, 2.55)

    # 阶段标签
    ax.add_patch(FancyBboxPatch((9.0, 5.8), 6.2, 0.4,
                                 boxstyle="round,pad=0.1", facecolor='#F3E5F5',
                                 edgecolor='#6A1B9A', linewidth=1, alpha=0.6))
    ax.text(12.1, 6.0, '阶段三：三级混合行为识别（创新点一+三）',
            ha='center', fontsize=9, fontweight='bold', color='#4A148C')

    # ---- 输出 ----
    draw_box(ax, 12.2, 0.8, 2.5, 0.8, '5类行为标注 + 可疑预警', '#E8F5E9', '#2E7D32',
             fontsize=10, textcolor='#1B5E20', subtext='Normal/Glancing/QuickTurn/Prolonged/LookDown')
    draw_arrow(ax, 12.2, 1.85, 12.2, 1.25)

    # 虚线框 grouping
    for rect_args, label_args in [
        # 检测跟踪
        (dict(xy=(3.1, 4.5), width=6.3, height=1.3), None),
        # 姿态估计
        (dict(xy=(0.1, 0.95), width=10.2, height=3.2), None),
        # 行为识别
        (dict(xy=(8.9, 1.6), width=6.5, height=4.5), None),
    ]:
        rect = plt.Rectangle(**rect_args, fill=False, edgecolor='#999',
                              linestyle='--', linewidth=0.8, zorder=1)
        ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(f'{OUT}/arch_fig1_pipeline.png', bbox_inches='tight', pad_inches=0.3)
    plt.savefig(f'{OUT}/arch_fig1_pipeline.pdf', bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print("  架构图1: 系统整体流水线")


# ========== 图B: Transformer 时序行为分类模型 ==========
def fig_transformer():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 7)
    ax.axis('off')

    ax.text(7, 6.6, 'Temporal Transformer 行为分类模型架构', ha='center',
            fontsize=14, fontweight='bold', color='#1A237E')

    # 输入
    draw_box(ax, 1.5, 5.5, 2.5, 0.7, '姿态序列输入', '#FFF3E0', '#E65100',
             textcolor='#BF360C', subtext='[B, T=90, 3] (yaw,pitch,roll)')

    # 线性投影
    draw_box(ax, 1.5, 4.3, 2.5, 0.6, 'Linear Projection', '#E3F2FD', '#1565C0',
             textcolor='#0D47A1', subtext='3 → d_model=64')
    draw_arrow(ax, 1.5, 5.15, 1.5, 4.65)

    # CLS token
    draw_box(ax, 4.8, 4.3, 1.8, 0.6, '[CLS] Token', '#FCE4EC', '#C62828',
             textcolor='#B71C1C', subtext='可学习参数')
    draw_arrow(ax, 2.75, 4.3, 3.9, 4.3, label='Concat')

    # 位置编码
    draw_box(ax, 1.5, 3.1, 2.5, 0.6, 'Positional Encoding', '#E8F5E9', '#2E7D32',
             textcolor='#1B5E20', subtext='正弦位置编码 [B, 91, 64]')
    draw_arrow(ax, 1.5, 3.95, 1.5, 3.45)
    draw_arrow(ax, 4.8, 3.95, 2.5, 3.45)

    # Transformer Encoder 大框
    te_x, te_y, te_w, te_h = 7.5, 3.2, 5.5, 3.5
    ax.add_patch(FancyBboxPatch((te_x - te_w/2, te_y - te_h/2), te_w, te_h,
                                 boxstyle="round,pad=0.15", facecolor='#F5F5F5',
                                 edgecolor='#6A1B9A', linewidth=2, zorder=1))
    ax.text(te_x, te_y + te_h/2 - 0.3, 'Transformer Encoder × 2 层',
            ha='center', fontsize=11, fontweight='bold', color='#4A148C', zorder=4)

    # 内部模块
    # Multi-Head Self-Attention
    draw_box(ax, 6.2, 3.8, 2.2, 0.6, 'Multi-Head\nSelf-Attention', '#E8EAF6', '#283593',
             fontsize=8, textcolor='#1A237E', subtext='H=4, d_k=16')
    # Add & Norm
    draw_box(ax, 6.2, 2.9, 2.2, 0.45, 'Add & LayerNorm', '#E8EAF6', '#283593',
             fontsize=8, textcolor='#1A237E')
    # FFN
    draw_box(ax, 9.0, 3.8, 2.0, 0.6, 'Feed-Forward\nNetwork', '#F3E5F5', '#6A1B9A',
             fontsize=8, textcolor='#4A148C', subtext='dim=256, GELU')
    # Add & Norm 2
    draw_box(ax, 9.0, 2.9, 2.0, 0.45, 'Add & LayerNorm', '#F3E5F5', '#6A1B9A',
             fontsize=8, textcolor='#4A148C')

    draw_arrow(ax, 6.2, 3.45, 6.2, 3.15)
    draw_arrow(ax, 7.3, 3.8, 8.0, 3.8)
    draw_arrow(ax, 9.0, 3.45, 9.0, 3.15)

    # 残差连接 (虚线)
    ax.annotate('', xy=(5.5, 2.9), xytext=(5.5, 3.8),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1, linestyle='--'))
    ax.annotate('', xy=(10.5, 2.9), xytext=(10.5, 3.8),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1, linestyle='--'))
    ax.text(5.1, 3.35, '残差', fontsize=6, color='#999', rotation=90, ha='center')
    ax.text(10.9, 3.35, '残差', fontsize=6, color='#999', rotation=90, ha='center')

    # 输入箭头
    draw_arrow(ax, 2.75, 3.1, 4.75, 3.5)

    # 自注意力可视化说明
    attn_x = 6.2
    for i, (label, color) in enumerate([
        ('Head1: 局部变化', '#2196F3'), ('Head2: 中程模式', '#4CAF50'),
        ('Head3: 长程依赖', '#FF9800'), ('Head4: 全局统计', '#9C27B0')
    ]):
        ax.text(5.0 + i * 1.7, 4.6, label, fontsize=6.5, color=color, ha='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.8))

    # CLS 输出
    draw_box(ax, 7.5, 1.2, 2.5, 0.6, '[CLS] 输出提取', '#FCE4EC', '#C62828',
             textcolor='#B71C1C', subtext='[B, d_model=64]')
    draw_arrow(ax, 7.5, 2.6, 7.5, 1.55)

    # 分类头
    draw_box(ax, 11.5, 1.2, 3.2, 0.6, '分类头 (MLP)', '#E8F5E9', '#2E7D32',
             textcolor='#1B5E20', subtext='64→128→64→N_cls  ReLU+Dropout')
    draw_arrow(ax, 8.75, 1.2, 9.9, 1.2)

    # 输出
    colors_out = ['#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#2196F3']
    labels_out = ['Normal', 'Glancing', 'QuickTurn', 'Prolonged', 'LookDown']
    for i, (lab, col) in enumerate(zip(labels_out, colors_out)):
        bx = 11.5 + (i - 2) * 0.7
        draw_box(ax, bx, 0.2, 0.6, 0.4, lab, col, col,
                 fontsize=6, textcolor='white', linewidth=1)
    draw_arrow(ax, 11.5, 0.85, 11.5, 0.45)
    ax.text(11.5, 0.55, 'Softmax', fontsize=7, ha='center', color='#555')

    plt.tight_layout()
    plt.savefig(f'{OUT}/arch_fig2_transformer.png', bbox_inches='tight', pad_inches=0.3)
    plt.savefig(f'{OUT}/arch_fig2_transformer.pdf', bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print("  架构图2: Transformer 模型结构")


# ========== 图C: 双路径 Fallback 机制详细图 ==========
def fig_fallback():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 6)
    ax.axis('off')

    ax.text(7, 5.6, '双路径头部姿态估计与 Fallback 容错机制', ha='center',
            fontsize=14, fontweight='bold', color='#1A237E')

    # 输入
    draw_box(ax, 1.5, 4.5, 2.2, 0.7, '人体跟踪框', '#FFF3E0', '#E65100',
             textcolor='#BF360C', subtext='YOLOv8+StrongSORT')
    draw_arrow(ax, 2.6, 4.5, 3.5, 4.5)

    # ROI提取
    draw_box(ax, 4.5, 4.5, 2.0, 0.7, 'ROI 提取', '#E3F2FD', '#1565C0',
             textcolor='#0D47A1', subtext='上半身 60% 区域')
    draw_arrow(ax, 5.5, 4.5, 6.5, 4.5)

    # SSD检测
    draw_box(ax, 7.5, 4.5, 2.0, 0.7, 'SSD 人脸检测', '#E3F2FD', '#1565C0',
             textcolor='#0D47A1', subtext='300×300, Caffe')
    draw_arrow(ax, 8.5, 4.5, 9.3, 4.5)

    # 判断
    diamond_x, diamond_y = 10.0, 4.5
    diamond = plt.Polygon([[diamond_x, diamond_y+0.4], [diamond_x+0.6, diamond_y],
                            [diamond_x, diamond_y-0.4], [diamond_x-0.6, diamond_y]],
                           facecolor='#FFECB3', edgecolor='#F57F17', linewidth=2, zorder=3)
    ax.add_patch(diamond)
    ax.text(diamond_x, diamond_y, 'conf\n≥0.45?', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#E65100', zorder=4)

    # ---- 主路径 (上) ----
    ax.add_patch(FancyBboxPatch((0.2, 3.0), 13.2, 2.7,
                                 boxstyle="round,pad=0.1", facecolor='#E3F2FD',
                                 edgecolor='#1565C0', linewidth=1.5, alpha=0.15, zorder=0))
    ax.text(0.5, 5.15, '主路径', fontsize=9, fontweight='bold', color='#1565C0',
            bbox=dict(boxstyle='round', facecolor='#BBDEFB', edgecolor='#1565C0'))

    # 是 → 头部框扩展
    draw_box(ax, 10.0, 3.2, 2.0, 0.65, '人脸框→头部框', '#E3F2FD', '#1565C0',
             textcolor='#0D47A1', subtext='↑60% ↓15% ←→35%')
    draw_arrow(ax, 10.0, 4.05, 10.0, 3.55, color='#2E7D32')
    ax.text(10.3, 3.85, '是', fontsize=9, color='#2E7D32', fontweight='bold')

    # BBox平滑
    draw_box(ax, 12.5, 3.2, 2.0, 0.65, 'BBox 时序平滑', '#E8EAF6', '#283593',
             textcolor='#1A237E', subtext='EMA α=0.4')
    draw_arrow(ax, 11.0, 3.2, 11.5, 3.2)

    # ---- Fallback 路径 (下) ----
    ax.add_patch(FancyBboxPatch((0.2, 0.3), 13.2, 1.9,
                                 boxstyle="round,pad=0.1", facecolor='#FFF8E1',
                                 edgecolor='#F57F17', linewidth=1.5, alpha=0.15, zorder=0))
    ax.text(0.5, 1.85, 'Fallback\n路径', fontsize=9, fontweight='bold', color='#E65100',
            bbox=dict(boxstyle='round', facecolor='#FFECB3', edgecolor='#F57F17'))

    draw_box(ax, 3.5, 1.2, 2.5, 0.7, '人体比例先验估计', '#FFF8E1', '#F57F17',
             textcolor='#E65100', subtext='头部h=22%×体高 w=55%×体宽')
    draw_box(ax, 6.8, 1.2, 2.5, 0.7, '头部区域裁剪', '#FFF8E1', '#F57F17',
             textcolor='#E65100', subtext='框顶部中心区域')

    draw_arrow(ax, 10.0, 4.05, 3.5, 1.6, color='#F57F17')
    ax.text(6.5, 2.7, '否 (Fallback)', fontsize=9, color='#E65100', fontweight='bold')
    draw_arrow(ax, 4.75, 1.2, 5.55, 1.2, color='#F57F17')

    # 汇合 → WHENet
    draw_box(ax, 10.5, 1.2, 2.0, 0.7, 'WHENet', '#F3E5F5', '#6A1B9A',
             fontsize=11, textcolor='#4A148C', subtext='ONNX 224×224')
    draw_arrow(ax, 8.05, 1.2, 9.5, 1.2, color='#F57F17')
    draw_arrow(ax, 12.5, 2.85, 10.5, 1.6, color='#1565C0')

    # 输出
    draw_box(ax, 13.0, 1.2, 1.2, 0.7, 'yaw\npitch\nroll', '#E8F5E9', '#2E7D32',
             fontsize=9, textcolor='#1B5E20')
    draw_arrow(ax, 11.5, 1.2, 12.4, 1.2)

    # 统计标注
    ax.text(7.5, 0.0, '主路径: 77.9% 帧  |  Fallback: 22.1% 帧  →  合计覆盖率: 100%',
            ha='center', fontsize=10, fontweight='bold', color='#1A237E',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8EAF6', edgecolor='#283593'))

    plt.tight_layout()
    plt.savefig(f'{OUT}/arch_fig3_fallback.png', bbox_inches='tight', pad_inches=0.3)
    plt.savefig(f'{OUT}/arch_fig3_fallback.pdf', bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print("  架构图3: Fallback 机制")


# ========== 图D: 三级混合决策框架 ==========
def fig_hybrid_decision():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 8)
    ax.axis('off')

    ax.text(7, 7.5, '三级混合决策行为识别框架', ha='center',
            fontsize=14, fontweight='bold', color='#1A237E')

    # 输入
    draw_box(ax, 2, 7.0, 3.0, 0.6, '当前帧姿态 + 历史缓冲区', '#FFF3E0', '#E65100',
             textcolor='#BF360C', subtext='(yaw, pitch, roll) × T 帧')

    # ---- 第1级: 姿态门控 ----
    ax.add_patch(FancyBboxPatch((0.2, 5.3), 4.8, 1.3,
                                 boxstyle="round,pad=0.1", facecolor='#FCE4EC',
                                 edgecolor='#C62828', linewidth=1.5, alpha=0.3))
    ax.text(0.5, 6.35, '第1级', fontsize=8, fontweight='bold', color='#C62828',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', edgecolor='#C62828'))

    draw_box(ax, 2.5, 5.8, 2.2, 0.5, '实时姿态门控', '#FCE4EC', '#C62828',
             fontsize=9, textcolor='#B71C1C')

    conditions = ['|yaw| > 40° → Prolonged', 'pitch > 28° → LookUp', 'pitch < -28° → LookDown']
    for i, cond in enumerate(conditions):
        ax.text(1.0, 5.45 - i*0.2, '• ' + cond, fontsize=7, color='#C62828', family='monospace')

    draw_arrow(ax, 2, 6.65, 2, 6.1)

    # 判断1
    d1x, d1y = 5.5, 5.8
    d1 = plt.Polygon([[d1x, d1y+0.35], [d1x+0.5, d1y],
                       [d1x, d1y-0.35], [d1x-0.5, d1y]],
                      facecolor='#FFECB3', edgecolor='#F57F17', linewidth=1.5, zorder=3)
    ax.add_patch(d1)
    ax.text(d1x, d1y, '触发?', ha='center', va='center', fontsize=8,
            fontweight='bold', color='#E65100', zorder=4)
    draw_arrow(ax, 3.6, 5.8, 5.0, 5.8)
    ax.text(5.7, 6.25, '是', fontsize=8, color='#2E7D32', fontweight='bold')

    # ---- 第2级: Transformer ----
    ax.add_patch(FancyBboxPatch((0.2, 3.3), 4.8, 1.3,
                                 boxstyle="round,pad=0.1", facecolor='#F3E5F5',
                                 edgecolor='#6A1B9A', linewidth=1.5, alpha=0.3))
    ax.text(0.5, 4.35, '第2级', fontsize=8, fontweight='bold', color='#6A1B9A',
            bbox=dict(boxstyle='round', facecolor='#E1BEE7', edgecolor='#6A1B9A'))

    draw_box(ax, 2.5, 3.8, 2.2, 0.5, 'Transformer 推理', '#F3E5F5', '#6A1B9A',
             fontsize=9, textcolor='#4A148C')
    ax.text(1.0, 3.45, '• 输入: 最近90帧序列', fontsize=7, color='#6A1B9A')
    ax.text(1.0, 3.25, '• 输出: Softmax概率 (conf > 0.3)', fontsize=7, color='#6A1B9A')

    draw_arrow(ax, 5.5, 5.45, 2.5, 4.1, color='#C62828')
    ax.text(3.5, 4.85, '否', fontsize=8, color='#C62828', fontweight='bold')

    # ---- 第3级: 规则 ----
    ax.add_patch(FancyBboxPatch((0.2, 1.3), 4.8, 1.3,
                                 boxstyle="round,pad=0.1", facecolor='#FFF8E1',
                                 edgecolor='#F57F17', linewidth=1.5, alpha=0.3))
    ax.text(0.5, 2.35, '第3级', fontsize=8, fontweight='bold', color='#E65100',
            bbox=dict(boxstyle='round', facecolor='#FFECB3', edgecolor='#F57F17'))

    draw_box(ax, 2.5, 1.8, 2.2, 0.5, '规则检测器', '#FFF8E1', '#F57F17',
             fontsize=9, textcolor='#E65100')
    rules = ['角速度: 5帧Δyaw>25°→QuickTurn',
             'V形: 极值差>45°→QuickTurn',
             '偏视: 3s |yaw|>30°→Prolonged',
             '张望: 3次方向切换→Glancing']
    for i, r in enumerate(rules):
        ax.text(1.0, 1.5 - i*0.18, '• ' + r, fontsize=6.5, color='#E65100', family='monospace')

    draw_arrow(ax, 2.5, 3.25, 2.5, 2.1)

    # ---- 右侧: 融合决策 ----
    ax.add_patch(FancyBboxPatch((6.5, 1.0), 7.0, 5.5,
                                 boxstyle="round,pad=0.15", facecolor='#FAFAFA',
                                 edgecolor='#283593', linewidth=2, alpha=0.3))
    ax.text(10, 6.2, '混合决策逻辑', ha='center', fontsize=12,
            fontweight='bold', color='#1A237E')

    # 直接输出 (门控)
    draw_box(ax, 8, 5.5, 3.0, 0.5, '① 姿态门控触发 → 直接采用', '#FFCDD2', '#C62828',
             fontsize=8, textcolor='#B71C1C')
    draw_arrow(ax, 6.0, 5.8, 6.5, 5.5, color='#2E7D32')

    # 模型+规则协调
    draw_box(ax, 8, 4.5, 3.5, 0.5, '② 模型有输出 & 规则检测到异常', '#E8EAF6', '#283593',
             fontsize=8, textcolor='#1A237E')
    draw_arrow(ax, 3.6, 3.8, 6.5, 4.5)
    draw_arrow(ax, 3.6, 1.8, 6.5, 4.3)

    # 子逻辑
    draw_box(ax, 11.5, 3.7, 2.5, 0.5, '模型判Normal\n且conf>0.90?', '#FFECB3', '#F57F17',
             fontsize=7, textcolor='#E65100')
    draw_arrow(ax, 8, 4.2, 11.5, 4.0)

    draw_box(ax, 9.5, 2.8, 2.5, 0.45, '是→信任模型(保守)', '#E8F5E9', '#2E7D32',
             fontsize=7.5, textcolor='#1B5E20')
    draw_box(ax, 12.5, 2.8, 2.0, 0.45, '否→采用规则', '#FFF8E1', '#F57F17',
             fontsize=7.5, textcolor='#E65100')
    draw_arrow(ax, 10.7, 3.45, 9.5, 3.05, color='#2E7D32')
    draw_arrow(ax, 12.3, 3.45, 12.5, 3.05, color='#F57F17')
    ax.text(10.0, 3.3, '是', fontsize=7, color='#2E7D32', fontweight='bold')
    ax.text(12.7, 3.3, '否', fontsize=7, color='#E65100', fontweight='bold')

    # 第三条路径
    draw_box(ax, 8, 2.0, 3.0, 0.5, '③ 仅规则有输出 → 采用规则', '#FFF8E1', '#F57F17',
             fontsize=8, textcolor='#E65100')

    # 最终输出
    draw_box(ax, 10, 1.0, 3.0, 0.6, '时序平滑 (滑动窗口 w=8)', '#E8EAF6', '#283593',
             fontsize=9, textcolor='#1A237E', subtext='加权投票 → 最终行为标签')
    draw_arrow(ax, 8, 1.7, 10, 1.35, color='#283593')
    draw_arrow(ax, 9.5, 2.55, 10, 1.35, color='#283593')
    draw_arrow(ax, 12.5, 2.55, 10.5, 1.35, color='#283593')
    draw_arrow(ax, 8, 5.2, 10, 1.35, color='#C62828')

    # 输出框
    colors_o = ['#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#2196F3']
    labels_o = ['Normal', 'Glancing', 'QuickTurn', 'Prolonged', 'LookDown']
    for i, (lab, col) in enumerate(zip(labels_o, colors_o)):
        draw_box(ax, 8 + i * 1.2, 0.0, 1.0, 0.4, lab, col, col,
                 fontsize=7, textcolor='white', linewidth=1)
    draw_arrow(ax, 10, 0.65, 10, 0.25)

    plt.tight_layout()
    plt.savefig(f'{OUT}/arch_fig4_hybrid.png', bbox_inches='tight', pad_inches=0.3)
    plt.savefig(f'{OUT}/arch_fig4_hybrid.pdf', bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print("  架构图4: 三级混合决策框架")


if __name__ == '__main__':
    print("=" * 50)
    print("  生成技术架构图")
    print("=" * 50)
    fig_pipeline()
    fig_transformer()
    fig_fallback()
    fig_hybrid_decision()
    print("=" * 50)
    print("  完成！共4张架构图")
    print("=" * 50)
