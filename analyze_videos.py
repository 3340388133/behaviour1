#!/usr/bin/env python3
"""
视频数据集分析脚本
自动扫描 data/raw_videos 目录下的所有视频，生成结构化的数据集说明文档
"""

import os
import json
import cv2
import re
from pathlib import Path
from typing import Dict, List, Any


def get_video_info(video_path: str) -> Dict[str, Any]:
    """获取视频的基本信息"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # 计算时长
    duration = frame_count / fps if fps > 0 else 0

    # 获取编码格式
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    cap.release()

    return {
        "fps": round(fps, 2),
        "total_frames": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 2),
        "codec": codec.strip()
    }


def infer_scene_from_filename(filename: str) -> Dict[str, str]:
    """从文件名推断场景信息"""
    filename_lower = filename.lower()

    # 推断摄像头视角
    if "正" in filename or "front" in filename_lower:
        camera_view = "front"
    elif "侧" in filename or "side" in filename_lower:
        camera_view = "side"
    else:
        camera_view = "unknown"

    # 推断光照条件
    if "室内" in filename or "indoor" in filename_lower:
        lighting = "indoor"
    elif "室外" in filename or "outdoor" in filename_lower:
        lighting = "outdoor"
    else:
        lighting = "unknown"

    # 推断场景类型
    if "地铁" in filename or "metro" in filename_lower or "gate" in filename_lower:
        scene = "gate"
    elif "闸机" in filename:
        scene = "gate"
    else:
        scene = "manual_queue"

    # 推断遮挡情况
    occlusion = []
    if "遮" in filename or "mask" in filename_lower:
        occlusion.append("mask_or_hat")
    if "单人" in filename or "single" in filename_lower:
        density = "single"
    else:
        density = "multiple"

    return {
        "camera_view": camera_view,
        "lighting": lighting,
        "scene": scene,
        "occlusion": occlusion,
        "density": density
    }


def evaluate_usability(video_info: Dict, scene_info: Dict) -> Dict[str, Any]:
    """评估视频的可用性"""
    risk_flags = []
    recommended_tasks = []

    fps = video_info.get("fps", 0)
    width = video_info.get("width", 0)
    height = video_info.get("height", 0)
    duration = video_info.get("duration_sec", 0)

    # 检查 FPS
    if fps < 25:
        risk_flags.append("low_fps")
    else:
        recommended_tasks.append("frame_extraction")

    # 检查分辨率 (720p = 1280x720)
    if width >= 1280 and height >= 720:
        recommended_tasks.append("head_pose")
    else:
        risk_flags.append("low_resolution")

    # 检查时长
    if duration < 5:
        risk_flags.append("too_short")

    # 检查遮挡
    if scene_info.get("occlusion"):
        risk_flags.append("potential_occlusion")

    # 检查人群密度
    if scene_info.get("density") == "multiple":
        risk_flags.append("crowd_occlusion")

    # 行为标注建议
    if duration >= 10 and fps >= 20:
        recommended_tasks.append("behavior_annotation")

    # 判断是否可用于数据集
    usable = len(risk_flags) <= 2 and fps >= 15 and duration >= 5

    return {
        "usable_for_dataset": usable,
        "recommended_tasks": recommended_tasks,
        "risk_flags": risk_flags
    }


def analyze_all_videos(video_dir: str) -> List[Dict[str, Any]]:
    """分析目录下所有视频"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
    results = []

    video_path = Path(video_dir)

    for file_path in sorted(video_path.iterdir()):
        if file_path.suffix in video_extensions:
            print(f"正在分析: {file_path.name}")

            # 获取视频基本信息
            video_info = get_video_info(str(file_path))

            if video_info is None:
                print(f"  警告: 无法读取视频 {file_path.name}")
                continue

            # 从文件名推断场景信息
            scene_info = infer_scene_from_filename(file_path.name)

            # 评估可用性
            usability = evaluate_usability(video_info, scene_info)

            # 组合结果
            result = {
                "video_id": file_path.name,
                "scene": scene_info["scene"],
                "camera_view": scene_info["camera_view"],
                "lighting": scene_info["lighting"],
                "density": scene_info["density"],
                "duration_sec": video_info["duration_sec"],
                "fps": video_info["fps"],
                "resolution": f"{video_info['width']}x{video_info['height']}",
                "total_frames": video_info["total_frames"],
                "codec": video_info["codec"],
                "usable_for_dataset": usability["usable_for_dataset"],
                "recommended_tasks": usability["recommended_tasks"],
                "risk_flags": usability["risk_flags"]
            }

            results.append(result)
            print(f"  完成: {result['resolution']}, {result['fps']}fps, {result['duration_sec']}s")

    return results


def save_results(results: List[Dict], output_dir: str):
    """保存分析结果"""
    output_path = Path(output_dir)

    # 保存 JSON 文件
    json_path = output_path / "video_dataset_overview.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已生成 JSON 文件: {json_path}")

    # 保存 CSV 文件
    csv_path = output_path / "video_dataset_overview.csv"
    if results:
        headers = list(results[0].keys())
        with open(csv_path, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write(','.join(headers) + '\n')
            # 写入数据
            for row in results:
                values = []
                for h in headers:
                    val = row[h]
                    if isinstance(val, list):
                        val = '|'.join(val)
                    elif isinstance(val, bool):
                        val = str(val).lower()
                    values.append(str(val))
                f.write(','.join(values) + '\n')
        print(f"已生成 CSV 文件: {csv_path}")

    # 打印统计摘要
    print("\n" + "="*60)
    print("数据集概览统计")
    print("="*60)
    print(f"总视频数: {len(results)}")
    usable_count = sum(1 for r in results if r['usable_for_dataset'])
    print(f"可用于数据集: {usable_count}")
    print(f"不建议使用: {len(results) - usable_count}")

    # 按场景统计
    scenes = {}
    for r in results:
        scene = r['scene']
        scenes[scene] = scenes.get(scene, 0) + 1
    print(f"\n场景分布:")
    for scene, count in scenes.items():
        print(f"  - {scene}: {count}")

    # 按视角统计
    views = {}
    for r in results:
        view = r['camera_view']
        views[view] = views.get(view, 0) + 1
    print(f"\n视角分布:")
    for view, count in views.items():
        print(f"  - {view}: {count}")

    # 按光照统计
    lightings = {}
    for r in results:
        light = r['lighting']
        lightings[light] = lightings.get(light, 0) + 1
    print(f"\n光照分布:")
    for light, count in lightings.items():
        print(f"  - {light}: {count}")


def main():
    video_dir = "data/raw_videos"
    output_dir = "data"

    print("="*60)
    print("视频数据集分析工具")
    print("="*60)
    print(f"扫描目录: {video_dir}")
    print()

    # 分析所有视频
    results = analyze_all_videos(video_dir)

    if not results:
        print("未找到任何视频文件!")
        return

    # 保存结果
    save_results(results, output_dir)

    print("\n分析完成!")


if __name__ == "__main__":
    main()
