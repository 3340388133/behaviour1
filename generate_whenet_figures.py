#!/usr/bin/env python3
"""生成 WHENet 头部姿态估计专题可视化（5张大图）"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Circle, Wedge
import matplotlib.gridspec as gridspec
import numpy as np
import json, glob, random
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

OUT = 'thesis_figures'
STATS_DIR = 'data/batch_inference_output'
POSE_DIR = 'data/pose_output'

BEH_NAMES = {0: '正常行为', 1: '频繁张望', 2: '快速回头', 3: '长时间观察', 4: '持续低头'}
BEH_COLORS = {0: '#4CAF50', 1: '#FF9800', 2: '#F44336', 3: '#9C27B0', 4: '#2196F3'}
BEH_NAMES_EN = {0: 'Normal', 1: 'Glancing', 2: 'QuickTurn', 3: 'Prolonged', 4: 'LookDown'}

# ============ 加载数据 ============
def load_all_pose_and_behavior():
    """加载所有姿态数据和行为标签"""
    all_tracks = {}  # {video_trackid: {'poses': [...], 'behavior': int}}

    for stats_f in sorted(glob.glob(f'{STATS_DIR}/*_inference_stats.json')):
        stats = json.load(open(stats_f))
        vname = stats['video_name']
        if stats['total_frames_processed'] < 100:
            continue

        track_behaviors = stats.get('track_behaviors', {})

        pose_f = f'{POSE_DIR}/{vname}_poses.json'
        try:
            pose_data = json.load(open(pose_f))
        except:
            continue

        for tid, beh in track_behaviors.items():
            poses = pose_data.get('tracks', {}).get(tid, {}).get('poses', [])
            if len(poses) > 10:
                all_tracks[f'{vname}_{tid}'] = {
                    'poses': poses,
                    'behavior': int(beh),
                    'video': vname,
                }
    return all_tracks

print("加载数据...")
all_tracks = load_all_pose_and_behavior()
print(f"  共 {len(all_tracks)} 条有效轨迹")

# 提取所有姿态点
all_yaws, all_pitchs, all_behs = [], [], []
for tid, info in all_tracks.items():
    beh = info['behavior']
    for p in info['poses']:
        all_yaws.append(p['yaw'])
        all_pitchs.append(p['pitch'])
        all_behs.append(beh)
all_yaws = np.array(all_yaws)
all_pitchs = np.array(all_pitchs)
all_behs = np.array(all_behs)
print(f"  共 {len(all_yaws):,} 个姿态数据点")


# ========== 大图1: 头部姿态角度-颜色映射 + 行为判定区域 ==========
def whenet_fig1_pose_zones():
    """Yaw-Pitch 平面上的行为判定区域 + 实际数据点密度"""
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

    # (a) 行为判定区域图（颜色越深=越可疑）
    ax1 = fig.add_subplot(gs[0])

    yaw_range = np.linspace(-180, 180, 360)
    pitch_range = np.linspace(-90, 90, 180)
    YAW, PITCH = np.meshgrid(yaw_range, pitch_range)

    # 计算每个点的"可疑度"——模拟系统的判定逻辑
    suspicion = np.zeros_like(YAW)

    # 正常区域: |yaw| < 30, |pitch| < 20 → 低可疑
    # Prolonged: |yaw| > 30 → 中高可疑
    # 姿态门控: |yaw| > 40 → 高可疑
    # LookDown: pitch < -20 → 中高
    # LookUp: pitch > 20 → 中高
    # 极端区域: |yaw| > 60 or |pitch| > 40 → 最高

    for i in range(len(pitch_range)):
        for j in range(len(yaw_range)):
            y, p = YAW[i, j], PITCH[i, j]
            s = 0
            # yaw贡献
            ay = abs(y)
            if ay > 60: s += 0.9
            elif ay > 40: s += 0.7  # 姿态门控阈值
            elif ay > 30: s += 0.4  # Prolonged阈值
            elif ay > 15: s += 0.15
            # pitch贡献
            ap = abs(p)
            if ap > 40: s += 0.4
            elif ap > 28: s += 0.3  # 门控阈值
            elif ap > 20: s += 0.2  # LookDown/Up阈值
            suspicion[i, j] = min(s, 1.0)

    # 自定义colormap: 绿→黄→橙→红
    colors_cmap = ['#E8F5E9', '#C8E6C9', '#FFF9C4', '#FFE082', '#FFB74D', '#FF8A65', '#EF5350', '#C62828']
    cmap = LinearSegmentedColormap.from_list('suspicion', colors_cmap, N=256)

    im = ax1.pcolormesh(YAW, PITCH, suspicion, cmap=cmap, shading='auto', alpha=0.9)

    # 画判定边界线
    # |yaw| = 40 姿态门控
    ax1.axvline(40, color='#C62828', linewidth=2, linestyle='--', alpha=0.8)
    ax1.axvline(-40, color='#C62828', linewidth=2, linestyle='--', alpha=0.8)
    ax1.text(42, 75, '姿态门控\n|yaw|=40°', fontsize=8, color='#C62828', fontweight='bold')

    # |yaw| = 30 Prolonged阈值
    ax1.axvline(30, color='#9C27B0', linewidth=1.5, linestyle='-.', alpha=0.7)
    ax1.axvline(-30, color='#9C27B0', linewidth=1.5, linestyle='-.', alpha=0.7)
    ax1.text(-29, -75, 'Prolonged\n|yaw|=30°', fontsize=7, color='#9C27B0')

    # pitch = ±28 门控
    ax1.axhline(28, color='#00BCD4', linewidth=1.5, linestyle='--', alpha=0.7)
    ax1.axhline(-28, color='#2196F3', linewidth=1.5, linestyle='--', alpha=0.7)
    ax1.text(100, 30, 'LookUp pitch=28°', fontsize=7, color='#00BCD4')
    ax1.text(100, -32, 'LookDown pitch=-28°', fontsize=7, color='#2196F3')

    # pitch = ±20 规则阈值
    ax1.axhline(20, color='#00BCD4', linewidth=1, linestyle=':', alpha=0.5)
    ax1.axhline(-20, color='#2196F3', linewidth=1, linestyle=':', alpha=0.5)

    # 正常区域标注
    ax1.add_patch(plt.Rectangle((-30, -20), 60, 40, fill=False,
                                 edgecolor='#4CAF50', linewidth=2.5, linestyle='-'))
    ax1.text(0, 0, 'Normal\n正常区域', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    cbar = plt.colorbar(im, ax=ax1, shrink=0.85)
    cbar.set_label('可疑程度', fontsize=10)
    cbar.set_ticks([0, 0.3, 0.6, 0.9])
    cbar.set_ticklabels(['低', '中', '高', '极高'])

    ax1.set_xlabel('Yaw (水平转头角度) °', fontsize=11)
    ax1.set_ylabel('Pitch (俯仰角度) °', fontsize=11)
    ax1.set_title('(a) 头部姿态角度-可疑程度映射与行为判定区域', fontsize=12, fontweight='bold')
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)

    # (b) 实际数据点密度 (2D histogram)
    ax2 = fig.add_subplot(gs[1])
    # 限制范围避免极端值
    mask = (np.abs(all_yaws) < 180) & (np.abs(all_pitchs) < 90)
    h = ax2.hist2d(all_yaws[mask], all_pitchs[mask], bins=[120, 60],
                    cmap='YlOrRd', range=[[-180, 180], [-90, 90]])
    plt.colorbar(h[3], ax=ax2, shrink=0.85, label='数据点密度')

    # 同样画阈值线
    ax2.axvline(40, color='white', linewidth=1.5, linestyle='--', alpha=0.8)
    ax2.axvline(-40, color='white', linewidth=1.5, linestyle='--', alpha=0.8)
    ax2.axvline(30, color='white', linewidth=1, linestyle=':', alpha=0.6)
    ax2.axvline(-30, color='white', linewidth=1, linestyle=':', alpha=0.6)
    ax2.axhline(28, color='white', linewidth=1, linestyle='--', alpha=0.6)
    ax2.axhline(-28, color='white', linewidth=1, linestyle='--', alpha=0.6)

    ax2.set_xlabel('Yaw °', fontsize=11)
    ax2.set_ylabel('Pitch °', fontsize=11)
    ax2.set_title('(b) 实际姿态数据分布密度', fontsize=12, fontweight='bold')

    fig.suptitle('WHENet 头部姿态角度分析与行为判定区域',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/whenet_fig1_pose_zones.png', bbox_inches='tight')
    plt.savefig(f'{OUT}/whenet_fig1_pose_zones.pdf', bbox_inches='tight')
    plt.close()
    print("  WHENet图1: 姿态角度-可疑程度映射")


# ========== 大图2: 各行为类别的姿态分布 ==========
def whenet_fig2_behavior_pose():
    """各行为类别在yaw/pitch空间的分布对比"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    beh_keys = [0, 1, 2, 3, 4]

    for idx, beh in enumerate(beh_keys):
        ax = axes.flat[idx]
        mask = all_behs == beh
        yaws = all_yaws[mask]
        pitchs = all_pitchs[mask]

        # 限制数据量 for display
        if len(yaws) > 8000:
            sel = np.random.choice(len(yaws), 8000, replace=False)
            yaws, pitchs = yaws[sel], pitchs[sel]

        # 散点图 + 密度
        ax.scatter(yaws, pitchs, s=1, alpha=0.15, color=BEH_COLORS[beh])
        # 画判定区域
        ax.axvline(40, color='#C62828', linewidth=1, linestyle='--', alpha=0.5)
        ax.axvline(-40, color='#C62828', linewidth=1, linestyle='--', alpha=0.5)
        ax.axhline(28, color='#00BCD4', linewidth=1, linestyle='--', alpha=0.5)
        ax.axhline(-28, color='#2196F3', linewidth=1, linestyle='--', alpha=0.5)
        ax.add_patch(plt.Rectangle((-30, -20), 60, 40, fill=False,
                                     edgecolor='#4CAF50', linewidth=1.5, linestyle='-', alpha=0.5))

        n = np.sum(all_behs == beh)
        ax.set_title(f'({chr(97+idx)}) {BEH_NAMES[beh]} ({BEH_NAMES_EN[beh]}, n={n:,})',
                    fontsize=11, fontweight='bold', color=BEH_COLORS[beh])
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('Yaw °')
        ax.set_ylabel('Pitch °')

        # 标注特征区域
        if beh == 0:
            ax.text(0, 0, '集中于\n中心区域', ha='center', fontsize=8, color='#2E7D32',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        elif beh == 3:
            ax.text(-100, 0, '大量分布于\n|yaw|>30°区域', ha='center', fontsize=8, color='#6A1B9A',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        elif beh == 4:
            ax.text(0, -50, '集中于\npitch<-20°', ha='center', fontsize=8, color='#1565C0',
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    # 最后一个子图：总览对比
    ax_all = axes.flat[5]
    for beh in beh_keys:
        mask = all_behs == beh
        yaws = all_yaws[mask]
        pitchs = all_pitchs[mask]
        if len(yaws) > 3000:
            sel = np.random.choice(len(yaws), 3000, replace=False)
            yaws, pitchs = yaws[sel], pitchs[sel]
        ax_all.scatter(yaws, pitchs, s=1, alpha=0.1, color=BEH_COLORS[beh],
                      label=BEH_NAMES[beh])
    ax_all.set_title('(f) 全部行为叠加对比', fontsize=11, fontweight='bold')
    ax_all.set_xlim(-180, 180)
    ax_all.set_ylim(-90, 90)
    ax_all.set_xlabel('Yaw °')
    ax_all.set_ylabel('Pitch °')
    ax_all.legend(markerscale=10, fontsize=8, loc='upper right')

    fig.suptitle('各行为类别在 Yaw-Pitch 姿态空间中的分布特征',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/whenet_fig2_behavior_pose.png', bbox_inches='tight')
    plt.savefig(f'{OUT}/whenet_fig2_behavior_pose.pdf', bbox_inches='tight')
    plt.close()
    print("  WHENet图2: 各行为姿态分布")


# ========== 大图3: 典型行为的时序姿态轨迹 ==========
def whenet_fig3_temporal():
    """展示典型行为的yaw/pitch时间变化曲线"""
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=False)

    # 为每种行为找一个典型轨迹
    for beh_idx, (beh, ax) in enumerate(zip([0, 1, 2, 3, 4], axes)):
        # 找一个该行为的较长轨迹
        candidates = [(tid, info) for tid, info in all_tracks.items()
                      if info['behavior'] == beh and len(info['poses']) > 60]
        if not candidates:
            candidates = [(tid, info) for tid, info in all_tracks.items()
                          if info['behavior'] == beh and len(info['poses']) > 20]

        if not candidates:
            ax.text(0.5, 0.5, f'无数据', transform=ax.transAxes, ha='center')
            continue

        # 选一个中等长度的
        candidates.sort(key=lambda x: len(x[1]['poses']))
        pick = candidates[len(candidates)//2]
        tid, info = pick
        poses = info['poses']

        frames = [p['frame'] for p in poses]
        yaws = [p['yaw'] for p in poses]
        pitchs = [p['pitch'] for p in poses]

        # 时间轴 (帧转秒)
        t = [(f - frames[0]) / 30.0 for f in frames]

        # 绘制 yaw
        ax.plot(t, yaws, color=BEH_COLORS[beh], linewidth=1.5, alpha=0.9, label='Yaw')
        ax.plot(t, pitchs, color=BEH_COLORS[beh], linewidth=1, alpha=0.5,
                linestyle='--', label='Pitch')

        # 画阈值区域
        ax.axhspan(-30, 30, alpha=0.08, color='#4CAF50')  # 正常区域
        ax.axhline(40, color='#C62828', linewidth=0.8, linestyle=':', alpha=0.5)
        ax.axhline(-40, color='#C62828', linewidth=0.8, linestyle=':', alpha=0.5)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='-', alpha=0.3)

        ax.set_ylabel('角度 (°)', fontsize=10)
        ax.set_title(f'({chr(97+beh_idx)}) {BEH_NAMES[beh]} ({BEH_NAMES_EN[beh]}) — 轨迹 {tid.split("_track")[0]}',
                    fontsize=11, fontweight='bold', color=BEH_COLORS[beh], loc='left')
        ax.set_ylim(-100, 100)
        ax.legend(fontsize=8, loc='upper right', ncol=2)

        # 标注行为特征
        if beh == 0:
            ax.text(t[-1]*0.5, 25, '视线稳定，yaw波动小', fontsize=9, color='#2E7D32',
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        elif beh == 1:
            # 标注方向切换
            ax.text(t[-1]*0.5, 70, '←频繁左右切换→', fontsize=9, color='#E65100',
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        elif beh == 2:
            ax.text(t[-1]*0.5, 70, '突发大幅度yaw变化', fontsize=9, color='#C62828',
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        elif beh == 3:
            ax.text(t[-1]*0.5, 70, '持续偏离中心(|yaw|>30°)', fontsize=9, color='#6A1B9A',
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        elif beh == 4:
            ax.text(t[-1]*0.5, -60, '持续负pitch(低头)', fontsize=9, color='#1565C0',
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    axes[-1].set_xlabel('时间 (秒)', fontsize=11)

    fig.suptitle('典型行为的头部姿态时序轨迹对比',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/whenet_fig3_temporal.png', bbox_inches='tight')
    plt.savefig(f'{OUT}/whenet_fig3_temporal.pdf', bbox_inches='tight')
    plt.close()
    print("  WHENet图3: 时序姿态轨迹")


# ========== 大图4: Yaw/Pitch 统计分布 (箱线图+小提琴图) ==========
def whenet_fig4_distribution():
    """各行为类别的yaw/pitch分布统计"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    beh_keys = [0, 1, 2, 3, 4]
    beh_labels = [f'{BEH_NAMES[k]}\n({BEH_NAMES_EN[k]})' for k in beh_keys]

    # (a) Yaw 箱线图
    ax = axes[0, 0]
    yaw_data = [all_yaws[all_behs == k] for k in beh_keys]
    bp = ax.boxplot(yaw_data, labels=beh_labels, patch_artist=True, showfliers=False,
                    widths=0.6, medianprops=dict(color='black', linewidth=2))
    for patch, beh in zip(bp['boxes'], beh_keys):
        patch.set_facecolor(BEH_COLORS[beh])
        patch.set_alpha(0.6)
    ax.axhline(40, color='#C62828', linewidth=1, linestyle='--', label='门控阈值 ±40°')
    ax.axhline(-40, color='#C62828', linewidth=1, linestyle='--')
    ax.axhline(30, color='#9C27B0', linewidth=1, linestyle=':', label='Prolonged ±30°')
    ax.axhline(-30, color='#9C27B0', linewidth=1, linestyle=':')
    ax.set_ylabel('Yaw (°)')
    ax.set_title('(a) 各行为类别 Yaw 角度分布', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    # (b) Pitch 箱线图
    ax = axes[0, 1]
    pitch_data = [all_pitchs[all_behs == k] for k in beh_keys]
    bp = ax.boxplot(pitch_data, labels=beh_labels, patch_artist=True, showfliers=False,
                    widths=0.6, medianprops=dict(color='black', linewidth=2))
    for patch, beh in zip(bp['boxes'], beh_keys):
        patch.set_facecolor(BEH_COLORS[beh])
        patch.set_alpha(0.6)
    ax.axhline(28, color='#00BCD4', linewidth=1, linestyle='--', label='LookUp 28°')
    ax.axhline(-28, color='#2196F3', linewidth=1, linestyle='--', label='LookDown -28°')
    ax.set_ylabel('Pitch (°)')
    ax.set_title('(b) 各行为类别 Pitch 角度分布', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)

    # (c) |Yaw| 均值对比
    ax = axes[1, 0]
    yaw_means = [np.mean(np.abs(all_yaws[all_behs == k])) for k in beh_keys]
    yaw_stds = [np.std(np.abs(all_yaws[all_behs == k])) for k in beh_keys]
    bars = ax.bar([BEH_NAMES[k] for k in beh_keys], yaw_means,
                  yerr=yaw_stds, capsize=4, color=[BEH_COLORS[k] for k in beh_keys],
                  edgecolor='white', alpha=0.8)
    for bar, m in zip(bars, yaw_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{m:.1f}°', ha='center', fontsize=9, fontweight='bold')
    ax.axhline(40, color='#C62828', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(30, color='#9C27B0', linewidth=1, linestyle=':', alpha=0.5)
    ax.set_ylabel('平均 |Yaw| (°)')
    ax.set_title('(c) 各行为类别平均水平转头幅度', fontsize=11, fontweight='bold')

    # (d) Yaw 变化率（角速度）对比
    ax = axes[1, 1]
    angular_velocities = {}
    for beh in beh_keys:
        avs = []
        for tid, info in all_tracks.items():
            if info['behavior'] == beh and len(info['poses']) > 15:
                yaws_t = [p['yaw'] for p in info['poses']]
                # 每帧的yaw变化
                diffs = np.abs(np.diff(yaws_t))
                # 过滤掉跳变 (>100说明可能是角度wrap-around)
                diffs = diffs[diffs < 100]
                if len(diffs) > 0:
                    avs.append(np.mean(diffs) * 30)  # 转换为°/秒
        angular_velocities[beh] = avs

    av_data = [angular_velocities.get(k, [0]) for k in beh_keys]
    bp2 = ax.boxplot(av_data, labels=[BEH_NAMES[k] for k in beh_keys],
                     patch_artist=True, showfliers=False, widths=0.6,
                     medianprops=dict(color='black', linewidth=2))
    for patch, beh in zip(bp2['boxes'], beh_keys):
        patch.set_facecolor(BEH_COLORS[beh])
        patch.set_alpha(0.6)
    ax.set_ylabel('角速度 (°/秒)')
    ax.set_title('(d) 各行为类别头部转动角速度', fontsize=11, fontweight='bold')

    fig.suptitle('WHENet 姿态角度统计分析',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/whenet_fig4_distribution.png', bbox_inches='tight')
    plt.savefig(f'{OUT}/whenet_fig4_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("  WHENet图4: 姿态角度统计分布")


# ========== 大图5: 头部朝向极坐标玫瑰图 + 可疑度径向图 ==========
def whenet_fig5_polar():
    """极坐标下的头部朝向分布"""
    fig = plt.figure(figsize=(16, 7))

    # (a) 各行为的yaw方向分布 (玫瑰图)
    ax1 = fig.add_subplot(121, polar=True)

    n_bins = 36  # 每10度一个bin
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)

    for beh in [0, 1, 2, 3, 4]:
        mask = all_behs == beh
        yaw_rad = np.deg2rad(all_yaws[mask])
        counts, _ = np.histogram(yaw_rad, bins=bin_edges)
        # 归一化
        counts = counts / counts.sum() * 100
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        width = 2 * np.pi / n_bins

        ax1.bar(centers, counts, width=width, alpha=0.35,
                color=BEH_COLORS[beh], label=BEH_NAMES[beh], edgecolor='white', linewidth=0.3)

    ax1.set_title('(a) 各行为头部朝向分布 (Yaw)', fontsize=11, fontweight='bold', pad=20)
    ax1.set_theta_zero_location('N')  # 0°在上方（正前方）
    ax1.set_theta_direction(-1)  # 顺时针
    ax1.legend(loc='lower left', bbox_to_anchor=(-0.15, -0.15), fontsize=8, ncol=3)

    # 标注方向
    ax1.text(0, ax1.get_ylim()[1]*1.15, '正前方\n(yaw=0°)', ha='center', fontsize=8, color='#2E7D32')
    ax1.text(np.pi/2, ax1.get_ylim()[1]*1.15, '右转\n(90°)', ha='center', fontsize=8, color='#999')
    ax1.text(-np.pi/2, ax1.get_ylim()[1]*1.15, '左转\n(-90°)', ha='center', fontsize=8, color='#999')
    ax1.text(np.pi, ax1.get_ylim()[1]*1.15, '背面\n(180°)', ha='center', fontsize=8, color='#C62828')

    # (b) 可疑度径向图 - yaw绝对值 vs 可疑占比
    ax2 = fig.add_subplot(122)

    yaw_bins = np.arange(0, 181, 10)
    total_per_bin = []
    suspicious_per_bin = []

    for i in range(len(yaw_bins) - 1):
        lo, hi = yaw_bins[i], yaw_bins[i+1]
        mask = (np.abs(all_yaws) >= lo) & (np.abs(all_yaws) < hi)
        total = np.sum(mask)
        susp = np.sum(mask & (all_behs > 0))
        total_per_bin.append(total)
        suspicious_per_bin.append(susp / total * 100 if total > 0 else 0)

    centers = (yaw_bins[:-1] + yaw_bins[1:]) / 2

    # 双Y轴
    ax2_twin = ax2.twinx()

    # 柱状图: 数据量
    bars = ax2.bar(centers, total_per_bin, width=8, alpha=0.3, color='#90CAF9',
                   edgecolor='white', label='数据量')
    ax2.set_ylabel('数据点数', color='#1565C0')

    # 折线图: 可疑率
    ax2_twin.plot(centers, suspicious_per_bin, 'o-', color='#C62828',
                  linewidth=2, markersize=5, label='可疑行为占比')
    ax2_twin.set_ylabel('可疑行为占比 (%)', color='#C62828')
    ax2_twin.set_ylim(0, 105)

    # 标注关键阈值
    ax2.axvline(30, color='#9C27B0', linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.axvline(40, color='#C62828', linewidth=1.5, linestyle='--', alpha=0.7)
    ax2.text(31, max(total_per_bin)*0.9, 'Prolonged\n30°', fontsize=8, color='#9C27B0')
    ax2.text(41, max(total_per_bin)*0.8, '门控\n40°', fontsize=8, color='#C62828')

    ax2.set_xlabel('|Yaw| (°)', fontsize=11)
    ax2.set_title('(b) Yaw 绝对值与可疑行为占比关系', fontsize=11, fontweight='bold')

    # 合并图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    fig.suptitle('头部朝向方向分析与 Yaw 角度-可疑度关系',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/whenet_fig5_polar.png', bbox_inches='tight')
    plt.savefig(f'{OUT}/whenet_fig5_polar.pdf', bbox_inches='tight')
    plt.close()
    print("  WHENet图5: 极坐标朝向+可疑度关系")


if __name__ == '__main__':
    print("=" * 50)
    print("  生成 WHENet 姿态专题可视化")
    print("=" * 50)
    whenet_fig1_pose_zones()
    whenet_fig2_behavior_pose()
    whenet_fig3_temporal()
    whenet_fig4_distribution()
    whenet_fig5_polar()
    print("=" * 50)
    print("  完成！共5张 WHENet 专题大图")
    print("=" * 50)
