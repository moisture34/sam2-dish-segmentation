#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 食材识别系统 - 网页版（Gradio GUI）· SAM2 版

用法:
    python predictor_web_sam2.py
然后在浏览器打开: http://127.0.0.1:7860

功能:
    1. 上传食材图片 + xlsx 配置文件（可选，不上传则用默认参数）
    2. 自动识别食材（支持 auto / auto_fine / color_fine 三种模式）
    3. 网页同时显示输入图片和识别结果
    4. 所有输出保存到 当前目录/output/时间戳/ 文件夹，可打包 zip 下载

依赖:
    pip install gradio pandas openpyxl matplotlib opencv-python
    pip install git+https://github.com/facebookresearch/sam2.git
"""

import os
# 修复 OpenMP 冲突错误
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =============================================================================
# ============ 固定配置（模型路径等，按你本机实际位置修改）=====================
# =============================================================================

SAM_CHECKPOINT_PATH = r"./checkpoints/sam2.1_hiera_large.pt"   # ← SAM2 权重路径
SAM2_CONFIG         = "configs/sam2.1/sam2.1_hiera_l.yaml"     # ← SAM2 配置（相对 sam2 库根或项目根）

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"        # 有 GPU 自动用

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")
TEMPLATE_XLSX = os.path.join(SCRIPT_DIR, "配置模板.xlsx")
SERVER_PORT = 7860

# =============================================================================
# ===================== 以下为代码逻辑，一般无需修改 ============================
# =============================================================================

import sys
import io
import zipfile
import datetime
import contextlib
from contextlib import nullcontext

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import gradio as gr

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# =============================================================================
# ===================== 默认配置（xlsx 里没填的项就用这些值）====================
# =============================================================================

DEFAULT_CONFIG = {
    # ---- 识别模式: auto / auto_fine / color_fine ----
    "MODE": "color_fine",

    # ---- 搜索区域 [x1, y1, x2, y2]：左锅大概位置 ----
    "AUTO_SEARCH_ROI": [250, 130, 460, 380],

    # ---- 自动模式参数 ----
    "AUTO_MIN_AREA_RATIO": 0.08,
    "AUTO_MAX_AREA_RATIO": 0.6,
    "AUTO_STABILITY_THRESHOLD": 0.7,
    "AUTO_SELECT_BY": "area",
    "AUTO_FINE_BOX_PADDING": 0.2,

    # ---- 颜色引导模式参数 ----
    "COLOR_CLOSE_KERNEL": 15,
    "COLOR_MIN_AREA": 500,
    "COLOR_FINE_BOX_PADDING": 0.35,
    "COLOR_FINE_NUM_POINTS": 25,
    "COLOR_FINE_MASK_INTERSECT": False,
    "COLOR_SELECT_BY": "center",
    "COLOR_MERGE_ALL": True,

    # ---- 后处理参数 ----
    "APPLY_MORPHOLOGY": True,
    "MORPH_KERNEL_SIZE": 3,
    "KEEP_LARGEST_COMPONENT": True,

    # ---- HSV 颜色范围（sheet2里可以改）----
    "COLOR_RANGES": [
        [10,  80,  60,  40, 255, 255],   # 黄色/金黄色
        [0,   60,  40,  15, 255, 255],   # 红色/红棕色
        [165, 60,  40, 179, 255, 255],   # 红色2
        [5,   40,  40,  25, 180, 180],   # 棕色/褐色
        [40,  30,  40,  80, 200, 200],   # 绿色
        [0,    0, 120, 179,  40, 255],   # 白色
    ],
}

PARAM_DOCS = [
    ("MODE",                      "color_fine",        "识别模式: auto / auto_fine / color_fine(推荐)"),
    ("AUTO_SEARCH_ROI",           "250,130,460,380",   "搜索区域 x1,y1,x2,y2 (逗号分隔)"),
    ("AUTO_MIN_AREA_RATIO",       "0.08",              "最小面积占搜索区比例"),
    ("AUTO_MAX_AREA_RATIO",       "0.6",               "最大面积占搜索区比例"),
    ("AUTO_STABILITY_THRESHOLD",  "0.7",               "稳定性阈值 (0~1)"),
    ("AUTO_SELECT_BY",            "area",              "auto模式筛选策略: area / stability"),
    ("AUTO_FINE_BOX_PADDING",     "0.2",               "auto_fine框扩展比例"),
    ("COLOR_CLOSE_KERNEL",        "15",                "颜色闭运算核大小"),
    ("COLOR_MIN_AREA",            "500",               "颜色检测最小面积(像素)"),
    ("COLOR_FINE_BOX_PADDING",    "0.35",              "color_fine框扩展比例"),
    ("COLOR_FINE_NUM_POINTS",     "25",                "正点采样数量 (15~30)"),
    ("COLOR_FINE_MASK_INTERSECT", "False",             "是否用颜色mask约束边缘 True/False"),
    ("COLOR_SELECT_BY",           "center",            "食物选择: center / area"),
    ("COLOR_MERGE_ALL",           "True",              "True=合并所有食材区域"),
    ("APPLY_MORPHOLOGY",          "True",              "形态学开运算去噪点 True/False"),
    ("MORPH_KERNEL_SIZE",         "3",                 "形态学核大小"),
    ("KEEP_LARGEST_COMPONENT",    "True",              "只保留最大连通区域 True/False"),
]

COLOR_RANGE_DOCS = [
    "黄色/金黄色(炒饭、煎蛋、土豆)", "红色/红棕色(生肉、番茄)", "红色2(另一区间)",
    "棕色/褐色(熟肉、酱油色)", "绿色(蔬菜、葱花)", "白色(米饭、豆腐、鱼片)",
]


# =============================================================================
# ===================== xlsx 配置读取 / 模板生成 ===============================
# =============================================================================

def parse_value(key, raw):
    default = DEFAULT_CONFIG[key]
    if isinstance(default, bool):
        return str(raw).strip().lower() in ("true", "1", "yes", "y", "是")
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        parts = str(raw).replace("，", ",").split(",")
        return [int(float(x.strip())) for x in parts if x.strip()]
    return str(raw).strip()


def load_config(xlsx_path):
    cfg = {k: (v.copy() if isinstance(v, list) else v) for k, v in DEFAULT_CONFIG.items()}
    xl = pd.ExcelFile(xlsx_path)

    if "基础参数" in xl.sheet_names:
        df = xl.parse("基础参数")
        for _, row in df.iterrows():
            key = str(row.iloc[0]).strip()
            if key in ("参数名", "nan", ""):
                continue
            if key in cfg and key != "COLOR_RANGES" and not pd.isna(row.iloc[1]):
                cfg[key] = parse_value(key, row.iloc[1])

    if "颜色范围" in xl.sheet_names:
        df = xl.parse("颜色范围")
        ranges = []
        for _, row in df.iterrows():
            vals = [row.iloc[i] for i in range(min(6, len(row)))]
            if any(pd.isna(v) for v in vals):
                continue
            try:
                ranges.append([int(float(v)) for v in vals])
            except (ValueError, TypeError):
                continue
        if ranges:
            cfg["COLOR_RANGES"] = ranges

    return cfg


def create_template():
    df_basic = pd.DataFrame(PARAM_DOCS, columns=["参数名", "参数值", "说明"])
    df_color = pd.DataFrame(
        [r + [COLOR_RANGE_DOCS[i]] for i, r in enumerate(DEFAULT_CONFIG["COLOR_RANGES"])],
        columns=["H_min", "S_min", "V_min", "H_max", "S_max", "V_max", "说明"],
    )
    with pd.ExcelWriter(TEMPLATE_XLSX, engine="openpyxl") as w:
        df_basic.to_excel(w, sheet_name="基础参数", index=False)
        df_color.to_excel(w, sheet_name="颜色范围", index=False)
    print(f"[信息] 已生成配置模板: {TEMPLATE_XLSX}")


# =============================================================================
# ===================== 绘图 / 保存工具函数 ====================================
# =============================================================================

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1, 1, -1))


def show_anns(anns, ax):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    for ann in sorted_anns:
        m = ann["segmentation"]
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        ax.imshow(m.reshape(m.shape[0], m.shape[1], 1) * color.reshape(1, 1, -1))


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*",
               s=marker_size, edgecolor="white", linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*",
               s=marker_size, edgecolor="white", linewidth=1.25)


def show_box(box, ax, color="green"):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2))


def save_figure(fig, out_dir, filename):
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[保存] {filepath}")


def save_mask_image(image, mask, out_dir, filename, alpha=0.5):
    filepath = os.path.join(out_dir, filename)
    h, w = mask.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = mask.astype(np.float32) * alpha

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.imshow(overlay)
    ax.axis("off")
    fig.savefig(filepath, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[保存] {filepath}")


def postprocess_mask(mask, cfg):
    mask_uint8 = mask.astype(np.uint8)

    if cfg["APPLY_MORPHOLOGY"]:
        k = cfg["MORPH_KERNEL_SIZE"]
        kernel = np.ones((k, k), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

    if cfg["KEEP_LARGEST_COMPONENT"]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        if num_labels > 2:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask_uint8 = (labels == largest_label).astype(np.uint8)

    return mask_uint8


# =============================================================================
# ===================== 三种识别模式 ==========================================
# =============================================================================

def _auto_find_food(sam, image, cfg):
    """内部函数：SAM2 自动分割器找搜索区域内最佳食材，返回 best 字典或 None"""
    mask_generator = SAM2AutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=cfg["AUTO_STABILITY_THRESHOLD"],
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )

    print("[auto] 正在自动检测所有物体...")
    masks = mask_generator.generate(image)
    print(f"[auto] 共检测到 {len(masks)} 个物体")

    if len(masks) == 0:
        return None, []

    roi_x1, roi_y1, roi_x2, roi_y2 = cfg["AUTO_SEARCH_ROI"]
    roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)

    candidates = []
    for i, ann in enumerate(masks):
        m = ann["segmentation"]
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        cx, cy = xs.mean(), ys.mean()

        if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
            continue

        area = ann["area"]
        area_ratio = area / roi_area
        if area_ratio < cfg["AUTO_MIN_AREA_RATIO"] or area_ratio > cfg["AUTO_MAX_AREA_RATIO"]:
            continue

        candidates.append({
            "index": i, "area": area, "area_ratio": area_ratio,
            "stability": ann["stability_score"], "center": (cx, cy),
            "segmentation": m, "bbox": ann["bbox"],
        })

    print(f"[auto] 搜索区域内符合条件的物体: {len(candidates)} 个")

    if len(candidates) == 0:
        return None, masks

    if cfg["AUTO_SELECT_BY"] == "area":
        candidates.sort(key=lambda x: x["area"], reverse=True)
    else:
        candidates.sort(key=lambda x: x["stability"], reverse=True)

    return candidates[0], masks


def _sam2_predict(sam, image, points, labels, box, multimask=True):
    """统一的 SAM2 预测调用（含 autocast）"""
    predictor = SAM2ImagePredictor(sam)
    predictor.set_image(image)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else nullcontext()
    with torch.inference_mode(), ctx:
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=np.array(box) if box is not None else None,
            multimask_output=multimask,
        )
    return masks, scores


def mode_auto(sam, image, cfg, out_dir):
    """全自动模式：自动检测 → 筛选 → 输出"""
    best, masks = _auto_find_food(sam, image, cfg)

    # 保存所有检测结果（调试用）
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    show_anns(masks, ax)
    show_box(cfg["AUTO_SEARCH_ROI"], ax, color="red")
    ax.set_title(f"All detected: {len(masks)} masks", fontsize=14)
    ax.axis("off")
    save_figure(fig, out_dir, "auto_all_masks.png")

    if best is None:
        print("[错误] 搜索区域内没有找到符合条件的物体，请调整 AUTO_SEARCH_ROI 或面积阈值")
        return None

    print(f"[auto] 选中最佳物体: 面积={best['area']} ({best['area_ratio']:.1%}), 稳定性={best['stability']:.3f}")

    final_mask = postprocess_mask(best["segmentation"], cfg)
    save_mask_image(image, final_mask, out_dir, "auto_result.png")
    return "auto_result.png"


def mode_auto_fine(sam, image, cfg, out_dir):
    """自动+精细模式：自动粗定位 → point_box 精细分割"""
    best, _ = _auto_find_food(sam, image, cfg)

    if best is None:
        print("[错误] 没有找到符合条件的物体，请调整 AUTO_SEARCH_ROI 或面积阈值")
        return None

    bx, by, bw, bh = best["bbox"]
    cx, cy = best["center"]
    print(f"[auto_fine] 粗定位完成: 中心=({cx:.0f},{cy:.0f}), 面积={best['area']}")

    pad_w = bw * cfg["AUTO_FINE_BOX_PADDING"]
    pad_h = bh * cfg["AUTO_FINE_BOX_PADDING"]
    fine_box = [
        max(0, int(bx - pad_w)), max(0, int(by - pad_h)),
        min(image.shape[1], int(bx + bw + pad_w)),
        min(image.shape[0], int(by + bh + pad_h)),
    ]
    fine_points = np.array([[int(cx), int(cy)]])
    fine_labels = np.array([1])

    print(f"[auto_fine] 自动生成 box: {fine_box}, point: ({int(cx)},{int(cy)})")

    # 粗定位结果图
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    show_mask(best["segmentation"], ax)
    show_box(fine_box, ax, color="yellow")
    show_points(fine_points, fine_labels, ax)
    ax.set_title(f"Auto detect -> fine refine\n(center: {int(cx)},{int(cy)})", fontsize=14)
    ax.axis("off")
    save_figure(fig, out_dir, "auto_fine_step1.png")

    # SAM2 精细分割
    print("[auto_fine] SAM2 精细分割中...")
    masks, scores = _sam2_predict(sam, image, fine_points, fine_labels, fine_box)
    print(f"[auto_fine] 精细分割完成, scores: {scores}")

    best_idx = int(scores.argmax())
    print(f"[auto_fine] 选中 mask #{best_idx + 1}, score: {scores[best_idx]:.3f}")

    # 候选对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        axes[i].imshow(image)
        show_mask(masks[i], axes[i])
        show_box(fine_box, axes[i], color="yellow")
        axes[i].set_title(f"Mask {i+1}, score={scores[i]:.3f}", fontsize=12)
        axes[i].axis("off")
    plt.tight_layout()
    save_figure(fig, out_dir, "auto_fine_candidates.png")

    final_mask = postprocess_mask(masks[best_idx], cfg)
    save_mask_image(image, final_mask, out_dir, "auto_fine_result.png")
    return "auto_fine_result.png"


def mode_color_fine(sam, image, cfg, out_dir):
    """颜色引导+精细模式：HSV颜色粗定位 → 多点引导SAM2精细分割"""
    roi_x1, roi_y1, roi_x2, roi_y2 = cfg["AUTO_SEARCH_ROI"]
    roi_img = image[roi_y1:roi_y2, roi_x1:roi_x2]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_RGB2HSV)

    print("[color_fine] 步骤1/3：用食物颜色检测食材区域...")

    mask_all = np.zeros((roi_img.shape[0], roi_img.shape[1]), dtype=np.uint8)
    for rng in cfg["COLOR_RANGES"]:
        lower = np.array(rng[:3])
        upper = np.array(rng[3:])
        m = cv2.inRange(hsv, lower, upper)
        mask_all = cv2.bitwise_or(mask_all, m)

    k = cfg["COLOR_CLOSE_KERNEL"]
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_all, connectivity=8)
    if num_labels <= 1:
        print("[错误] 颜色检测没有找到任何食物色区域，请调整颜色范围或搜索区域")
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_idx = np.where(areas >= cfg["COLOR_MIN_AREA"])[0]
    if len(valid_idx) == 0:
        print("[错误] 颜色检测到的区域都太小，请调小 COLOR_MIN_AREA")
        return None

    # 按策略选择最佳候选
    if cfg.get("COLOR_MERGE_ALL", True):
        # 合并所有有效的食物色区域
        food_mask_roi = np.zeros_like(mask_all)
        for idx in valid_idx:
            food_mask_roi |= (labels == (idx + 1)).astype(np.uint8)
        food_area = int(food_mask_roi.sum())
        ys_t, xs_t = np.where(food_mask_roi > 0)
        cx_roi, cy_roi = xs_t.mean(), ys_t.mean()
        print(f"[color_fine] 合并 {len(valid_idx)} 个食物区域, 总面积={food_area}")
    elif cfg["COLOR_SELECT_BY"] == "center":
        roi_cx = (roi_x2 - roi_x1) / 2
        roi_cy = (roi_y2 - roi_y1) / 2
        dists = []
        for idx in valid_idx:
            cx_i, cy_i = centroids[idx + 1]
            dists.append(np.sqrt((cx_i - roi_cx) ** 2 + (cy_i - roi_cy) ** 2))
        best_label = valid_idx[int(np.argmin(dists))] + 1
        food_mask_roi = (labels == best_label).astype(np.uint8)
        food_area = stats[best_label, cv2.CC_STAT_AREA]
        cx_roi, cy_roi = centroids[best_label]
    else:
        best_label = valid_idx[int(np.argmax(areas[valid_idx]))] + 1
        food_mask_roi = (labels == best_label).astype(np.uint8)
        food_area = stats[best_label, cv2.CC_STAT_AREA]
        cx_roi, cy_roi = centroids[best_label]

    cx, cy = cx_roi + roi_x1, cy_roi + roi_y1

    ys, xs = np.where(food_mask_roi > 0)
    bx, by = xs.min() + roi_x1, ys.min() + roi_y1
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()

    print(f"[color_fine] 颜色检测完成: 面积={food_area}, 中心=({cx:.0f},{cy:.0f})")

    pad_w = bw * cfg["COLOR_FINE_BOX_PADDING"]
    pad_h = bh * cfg["COLOR_FINE_BOX_PADDING"]
    fine_box = [
        max(0, int(bx - pad_w)), max(0, int(by - pad_h)),
        min(image.shape[1], int(bx + bw + pad_w)), min(image.shape[0], int(by + bh + pad_h)),
    ]
    print(f"[color_fine] 自动生成 box: {fine_box}")

    # 从颜色mask均匀采样多个正点
    n_points = cfg["COLOR_FINE_NUM_POINTS"]
    print(f"[color_fine] 步骤2/3：从颜色mask采样 {n_points} 个正点...")
    food_ys, food_xs = np.where(food_mask_roi > 0)
    if len(food_xs) > n_points:
        step = max(1, len(food_xs) // n_points)
        sample_idx = np.arange(0, len(food_xs), step)[:n_points]
    else:
        sample_idx = np.arange(len(food_xs))
    fine_points = np.column_stack([food_xs[sample_idx] + roi_x1, food_ys[sample_idx] + roi_y1]).astype(np.float32)
    fine_labels = np.ones(len(fine_points), dtype=np.int32)

    # 调试图：颜色检测结果 + 采样点
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(roi_img)
    axes[0].set_title("ROI (search area)", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(mask_all, cmap="gray")
    axes[1].set_title(f"Color mask ({(mask_all > 0).sum()} px)", fontsize=12)
    axes[1].axis("off")
    axes[2].imshow(image)
    show_box(fine_box, axes[2], color="yellow")
    show_n = min(20, len(fine_points))
    show_pt_idx = np.linspace(0, len(fine_points) - 1, show_n, dtype=int)
    axes[2].scatter(fine_points[show_pt_idx, 0], fine_points[show_pt_idx, 1],
                    color="lime", s=30, marker="o", edgecolors="white", linewidths=0.5)
    axes[2].set_title(f"Sampled points ({len(fine_points)} total)", fontsize=12)
    axes[2].axis("off")
    plt.tight_layout()
    save_figure(fig, out_dir, "color_fine_step1.png")

    # SAM2 精细分割
    print("[color_fine] 步骤3/3：SAM2 精细分割中...")
    masks, scores = _sam2_predict(sam, image, fine_points, fine_labels, fine_box)
    print(f"[color_fine] 精细分割完成, scores: {scores}")

    # 选覆盖食物色最多的 mask
    color_mask_full = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    color_mask_full[roi_y1:roi_y2, roi_x1:roi_x2] = food_mask_roi

    best_idx = int(scores.argmax())
    best_coverage = 0
    for i in range(len(masks)):
        m = masks[i].astype(np.uint8)
        overlap = cv2.bitwise_and(m, color_mask_full)
        coverage = overlap.sum()
        food_ratio = coverage / max(food_area, 1)
        combined = scores[i] * 0.3 + food_ratio * 0.7
        print(f"[color_fine] mask#{i+1}: score={scores[i]:.3f}, 食物色覆盖={coverage}px ({food_ratio:.1%}), 综合={combined:.3f}")
        if combined > best_coverage:
            best_coverage = combined
            best_idx = i
    print(f"[color_fine] 选中 mask #{best_idx + 1}, score: {scores[best_idx]:.3f}")

    final_mask = postprocess_mask(masks[best_idx], cfg)

    # 可选：颜色mask约束边缘
    if cfg["COLOR_FINE_MASK_INTERSECT"]:
        color_mask_full = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        color_mask_full[roi_y1:roi_y2, roi_x1:roi_x2] = food_mask_roi
        color_mask_dilated = cv2.dilate(color_mask_full, np.ones((5, 5), np.uint8), iterations=1)
        final_mask = cv2.bitwise_and(final_mask, color_mask_dilated)
        print("[color_fine] 已用颜色mask约束边缘")

    # 候选对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        axes[i].imshow(image)
        show_mask(masks[i], axes[i])
        show_box(fine_box, axes[i], color="yellow")
        axes[i].set_title(f"Mask {i+1}, score={scores[i]:.3f}", fontsize=12)
        axes[i].axis("off")
    plt.tight_layout()
    save_figure(fig, out_dir, "color_fine_candidates.png")

    save_mask_image(image, final_mask, out_dir, "color_fine_result.png")
    return "color_fine_result.png"


# =============================================================================
# ===================== 模型加载（全局只加载一次）==============================
# =============================================================================

_sam_cache = None

def get_sam():
    """加载 SAM2 模型（只加载一次，后续识别复用）"""
    global _sam_cache
    if _sam_cache is None:
        print(f"[信息] 首次加载 SAM2 模型: {SAM_CHECKPOINT_PATH} -> {DEVICE} ...")
        print(f"[信息] SAM2 配置: {SAM2_CONFIG}")
        sam = build_sam2(SAM2_CONFIG, SAM_CHECKPOINT_PATH, device=DEVICE)
        _sam_cache = sam
        print("[信息] 模型加载完成")
    return _sam_cache


# =============================================================================
# ===================== 主处理流程（Gradio 调用）===============================
# =============================================================================

RESULT_FILE_BY_MODE = {
    "auto": "auto_result.png",
    "auto_fine": "auto_fine_result.png",
    "color_fine": "color_fine_result.png",
}


def run_recognition(image_rgb, config_file):
    """
    Gradio 主处理函数
    输入: 图片(numpy RGB), xlsx配置文件(可选)
    输出: 输入图片, 识别结果图, 运行日志, zip文件路径, 输出文件夹路径
    """
    if image_rgb is None:
        return None, None, "[错误] 请先上传图片", None

    # 创建时间戳输出目录
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, ts)
    os.makedirs(out_dir, exist_ok=True)

    buf = io.StringIO()
    result_png = None

    with contextlib.redirect_stdout(buf):
        try:
            print("=" * 50)
            print("SAM2 食材识别系统")
            print(f"输出目录: {out_dir}")
            print("=" * 50)

            # 1. 读取配置
            if config_file is not None:
                cfg_path = config_file if isinstance(config_file, str) else config_file.name
                print(f"[信息] 使用配置文件: {os.path.basename(cfg_path)}")
                cfg = load_config(cfg_path)
            else:
                print("[信息] 未上传配置文件，使用默认参数")
                cfg = {k: (v.copy() if isinstance(v, list) else v) for k, v in DEFAULT_CONFIG.items()}

            print(f"[信息] 识别模式: {cfg['MODE']}")
            print(f"[信息] 搜索区域: {cfg['AUTO_SEARCH_ROI']}")

            # 2. 保存原图副本
            input_save = os.path.join(out_dir, "input.png")
            cv2.imwrite(input_save, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            print(f"[保存] {input_save}")
            print(f"[信息] 图片尺寸: {image_rgb.shape}")

            # 3. 加载模型
            sam = get_sam()

            # 4. 执行识别
            mode = cfg["MODE"]
            if mode == "auto":
                result_png = mode_auto(sam, image_rgb, cfg, out_dir)
            elif mode == "auto_fine":
                result_png = mode_auto_fine(sam, image_rgb, cfg, out_dir)
            elif mode == "color_fine":
                result_png = mode_color_fine(sam, image_rgb, cfg, out_dir)
            else:
                print(f"[错误] 未知模式: {mode}（支持: auto / auto_fine / color_fine）")

            if result_png:
                print("=" * 50)
                print(f"识别完成！结果: {result_png}")
                print("=" * 50)

        except Exception as e:
            import traceback
            print(f"[错误] 处理失败: {e}")
            traceback.print_exc(file=buf)

    log_text = buf.getvalue()

    if result_png is None:
        return image_rgb, None, log_text, None

    # 5. 打包 zip（排除 zip 自身）
    zip_path = os.path.join(out_dir, f"识别结果_{ts}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(out_dir)):
            if f.endswith(".png"):
                zf.write(os.path.join(out_dir, f), f)
    log_text += f"\n[保存] {zip_path}"
    log_text += f"\n[信息] 所有文件已保存到: {out_dir}"

    result_path = os.path.join(out_dir, result_png)
    return image_rgb, result_path, log_text, zip_path


# =============================================================================
# ===================== Gradio 网页界面 =======================================
# =============================================================================

def build_ui():
    with gr.Blocks(title="SAM2 食材识别系统") as demo:
        gr.Markdown("# SAM2 食材识别系统")
        gr.Markdown("上传图片和配置文件（可选），自动识别锅中的食材。不上传配置文件则使用默认参数。")

        with gr.Row():
            with gr.Column(scale=1):
                inp_img = gr.Image(label="① 上传食材图片", type="numpy", height=350)
                inp_cfg = gr.File(label="② 上传配置文件 .xlsx（可选）", file_types=[".xlsx"])
                btn = gr.Button("开始识别", variant="primary", size="lg")
                tpl_file = gr.File(label="没有配置文件？下载模板", value=TEMPLATE_XLSX)

            with gr.Column(scale=1):
                out_orig = gr.Image(label="输入图片", height=340)
                out_result = gr.Image(label="识别结果", height=340)

        out_log = gr.Textbox(label="运行日志", lines=12, max_lines=20)
        out_zip = gr.File(label="下载全部结果 (zip)")

        btn.click(
            fn=run_recognition,
            inputs=[inp_img, inp_cfg],
            outputs=[out_orig, out_result, out_log, out_zip],
        )

        gr.Markdown(
            "输出文件保存在 `output/时间戳/` 文件夹中。"
            "识别模式: auto(全自动) / auto_fine(自动+精细) / color_fine(颜色引导,推荐)"
        )

    return demo


if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    if not os.path.exists(TEMPLATE_XLSX):
        create_template()

    demo = build_ui()
    print(f"启动网页服务: http://127.0.0.1:{SERVER_PORT}")
    demo.launch(
        server_name="127.0.0.1",
        server_port=SERVER_PORT,
        inbrowser=True,
        allowed_paths=[SCRIPT_DIR],
    )