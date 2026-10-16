import io
import os
import time
import json
import base64
import tempfile
import traceback
import zipfile
import contextlib
import threading
import asyncio
import sys
import mimetypes
import random
import urllib.error
import urllib.request
from functools import wraps
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

if getattr(sys, "frozen", False):
    # PyInstaller 6 stores bundled resources in the onedir _internal folder.
    APP_DIR = os.environ.get(
        "SAM2_APP_DIR",
        os.path.dirname(os.path.abspath(sys.executable)),
    )
    _internal_dir = os.path.join(APP_DIR, "_internal")
    SCRIPT_DIR = _internal_dir if os.path.isdir(_internal_dir) else APP_DIR
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPT_DIR = APP_DIR

# The repository folder is named sam3, while the importable package is its
# nested sam3 subfolder. Put the repository root first so package resources
# (notably the BPE vocabulary) resolve correctly.
_SAM3_SOURCE_ROOT = os.path.join(APP_DIR, "sam3")
if os.path.isdir(os.path.join(_SAM3_SOURCE_ROOT, "sam3")) and _SAM3_SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SAM3_SOURCE_ROOT)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Remove interrupted YOLO-World downloads before the server can select them.
# A valid Ultralytics checkpoint is a torch zip archive.
for _world_weight in ("yolov8s-world.pt", "yolov8m-world.pt"):
    _world_path = os.path.join(SCRIPT_DIR, _world_weight)
    if (
        os.path.isfile(_world_path)
        and os.path.getsize(_world_path) > 0
        and not zipfile.is_zipfile(_world_path)
    ):
        try:
            os.remove(_world_path)
            print(f"[YOLO-World] removed incomplete checkpoint: {_world_path}")
        except OSError as _cleanup_error:
            print(f"[YOLO-World] could not remove incomplete checkpoint: {_cleanup_error}")

# ============================================================
# FastAPI
# ============================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", include_in_schema=False)
def health_check():
    """Lightweight liveness endpoint for local launchers and monitors."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def serve_workspace():
    return FileResponse(
        os.path.join(SCRIPT_DIR, "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/concept-tracking", include_in_schema=False)
def serve_concept_tracking():
    return FileResponse(
        os.path.join(SCRIPT_DIR, "concept_tracking.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/sam3-tracking", include_in_schema=False)
def serve_sam3_tracking():
    return FileResponse(
        os.path.join(SCRIPT_DIR, "sam3_tracking.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )

# ============================================================
# SAM2
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
SAM2_CHECKPOINT = os.path.join(SCRIPT_DIR, "checkpoints", "sam2.1_hiera_large.pt")
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"

print(f"[SAM2] loading on {device} ...")
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
predictor = SAM2ImagePredictor(sam2_model)
mask_generator = SAM2AutomaticMaskGenerator(
    sam2_model, points_per_side=32,
    pred_iou_thresh=0.85, stability_score_thresh=0.90,
    min_mask_region_area=200,
)
world_model = None
MODEL_STATUS = {
    "phase": "idle",
    "model": None,
    "message": "模型尚未加载",
}
MODEL_STATUS_LOCK = threading.Lock()
INFERENCE_LOCK = asyncio.Lock()


def serialized_inference(func):
    @wraps(func)
    async def wrapped(*args, **kwargs):
        async with INFERENCE_LOCK:
            return await func(*args, **kwargs)
    return wrapped
print("[SAM2] ready.")

# ============================================================
# 路径
# ============================================================
# Keep generated annotations outside PyInstaller's bundled resource directory.
OUTPUT_ROOT   = os.path.join(APP_DIR, "output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
TEMPLATE_XLSX = os.path.join(SCRIPT_DIR, "配置模板.xlsx")

# ============================================================
# 全局状态
# ============================================================
STATE = {
    "image_bgr": None, "image_rgb": None, "h": 0, "w": 0,
    "instances": {}, "next_id": 1, "current_id": None,
    "predictor_ready": False,
    "output_dir": None,
    "last_log": "",
}

# 独立的概念分割追踪会话。视频模型按需加载，避免普通图片标注启动时额外占用显存。
TRACK_STATE = {
    "video_path": None,
    "frames_dir": None,
    "preview_path": None,
    "width": 0,
    "height": 0,
    "fps": 0.0,
    "frame_count": 0,
    "predictor": None,
    "inference_state": None,
    "prompted": False,
    "progress": 0,
    "current_frame": 0,
    "current_time": 0.0,
    "sample_interval": 1.0,
    "run_mode": None,
    "prompt_specs": {},
    "timeline": [],
    "status": "idle",
    "message": "请上传视频",
    "error": None,
}
TRACK_LOCK = threading.RLock()
TRACK_INFERENCE_LOCK = threading.Lock()
TRACK_MODEL = None
SAM2_BATCH_STATE = {
    "root": None, "frames_dir": None, "files": [], "width": 0, "height": 0,
    "frame_count": 0, "inference_state": None, "prompted_container": False,
    "prompted_target": False, "status": "idle", "progress": 0,
    "message": "请上传批量图片", "error": None, "batch_id": None,
    "targets": {}, "container_points": [], "container_labels": [],
    "container_box": None, "seed_method": None, "gpt_analysis": None,
}
SAM2_BATCH_LOCK = threading.RLock()
SAM3_MODEL = None
SAM3_IMAGE_MODEL = None
SAM3_IMAGE_PROCESSOR = None
SAM3_AUTOCAST_CTX = None  # 保存 autocast 上下文管理器，防止被垃圾回收
SAM3_CHECKPOINT = os.path.join(APP_DIR, "sam3.pt")
SAM3_STATE = {
    "frames_dir": None, "root": None, "session_id": None, "predictor": None,
    "frame_count": 0, "width": 0, "height": 0, "fps": 0.0,
    "preview_path": None, "prompted": False, "status": "idle", "progress": 0,
    "current_frame": 0, "current_time": 0.0, "sample_interval": 1.0,
    "video_prompt": None,
    "run_mode": None,
    "timeline": [],
    "message": "请上传视频", "error": None,
}
SAM3_LOCK = threading.RLock()

MASK_ALPHA     = 0.50
MASK_ALPHA_NEW = 0.82
BASE_FADE      = 0.28

COLORS = [
    [255,  84,  84], [ 84, 178, 255], [124, 220, 124], [255, 196,  84],
    [196, 124, 255], [ 84, 220, 220], [255, 124, 196], [255, 240, 124],
    [168, 220,  84], [124, 148, 255], [255, 160, 100], [100, 255, 200],
]

def get_color(iid: int):
    return COLORS[(iid - 1) % len(COLORS)]

# ============================================================
# mask 强制转 bool
# ============================================================
def ensure_bool_mask(mask):
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.dtype == np.bool_:
        return mask
    if mask.dtype.kind == "f":
        return mask > 0.5
    return mask != 0


def _sam3_masks_from_output(output):
    """Return SAM3 masks as a list of HxW bool arrays."""
    if not isinstance(output, dict) or output.get("masks") is None:
        return []
    masks = output["masks"]
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    result = []
    for mask in masks:
        if np.asarray(mask).ndim != 2:
            continue
        bool_mask = ensure_bool_mask(mask)
        if bool_mask is not None and bool_mask.any():
            result.append(bool_mask)
    return result


def _batch_mask_bbox(mask):
    mask = ensure_bool_mask(mask)
    if mask is None or not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _choose_container_mask(masks, width, height, prompt_points=None,
                           prompt_labels=None, prompt_box=None,
                           reference_point=None):
    """Choose the container mask using the first-image prompt when available."""
    if not masks:
        return None
    points = prompt_points or []
    labels = prompt_labels or []
    positive_points = [
        (float(point[0]) / max(width, 1), float(point[1]) / max(height, 1))
        for index, point in enumerate(points)
        if index >= len(labels) or bool(labels[index])
    ]
    negative_points = [
        (float(point[0]) / max(width, 1), float(point[1]) / max(height, 1))
        for index, point in enumerate(points)
        if index < len(labels) and not bool(labels[index])
    ]
    box_center = None
    if prompt_box and len(prompt_box) == 4:
        x1, y1, x2, y2 = [float(value) for value in prompt_box]
        box_center = (
            ((x1 + x2) / 2) / max(width, 1),
            ((y1 + y2) / 2) / max(height, 1),
        )

    scored = []
    for index, mask in enumerate(masks):
        mask = ensure_bool_mask(mask)
        area = int(mask.sum())
        if not area:
            continue
        score = 0.0
        for px, py in positive_points:
            x = min(max(int(px * width), 0), width - 1)
            y = min(max(int(py * height), 0), height - 1)
            score += 100.0 if mask[y, x] else 0.0
        for px, py in negative_points:
            x = min(max(int(px * width), 0), width - 1)
            y = min(max(int(py * height), 0), height - 1)
            score -= 100.0 if mask[y, x] else 0.0
        if box_center:
            x = min(max(int(box_center[0] * width), 0), width - 1)
            y = min(max(int(box_center[1] * height), 0), height - 1)
            score += 100.0 if mask[y, x] else 0.0
        if reference_point:
            ys, xs = np.where(mask)
            if len(xs):
                center_x = float(np.mean(xs)) / max(width, 1)
                center_y = float(np.mean(ys)) / max(height, 1)
                distance = float(np.hypot(
                    center_x - float(reference_point[0]),
                    center_y - float(reference_point[1]),
                ))
                # Preserve the first-image spatial choice when several
                # instances share the same text concept.
                score += 120.0 * max(0.0, 1.0 - min(distance / 0.75, 1.0))
        # Area is only a tie-breaker. A prompt match must win over a larger
        # unrelated object returned for the same container concept.
        scored.append((score, area, index, mask))
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], item[1]))[3]


def _batch_prompt_reference_point(points=None, labels=None, box=None, width=1, height=1):
    positive = [
        point for index, point in enumerate(points or [])
        if index >= len(labels or []) or bool((labels or [])[index])
    ]
    if positive:
        point = positive[0]
        return (float(point[0]) / max(width, 1), float(point[1]) / max(height, 1))
    if box and len(box) == 4:
        return (
            (float(box[0]) + float(box[2])) / 2 / max(width, 1),
            (float(box[1]) + float(box[3])) / 2 / max(height, 1),
        )
    return None


def _map_crop_mask(mask, crop_x, crop_y, image_height, image_width):
    """Place a crop-sized mask back into full-image coordinates."""
    mask = ensure_bool_mask(mask)
    full = np.zeros((image_height, image_width), dtype=bool)
    crop_h, crop_w = mask.shape[:2]
    x2 = min(crop_x + crop_w, image_width)
    y2 = min(crop_y + crop_h, image_height)
    if x2 > crop_x and y2 > crop_y:
        full[crop_y:y2, crop_x:x2] = mask[:y2 - crop_y, :x2 - crop_x]
    return full


_BATCH_PROMPT_ALIASES = {
    "锅": "frying pan",
    "平底锅": "frying pan",
    "煎锅": "frying pan",
    "炒锅": "wok",
    "盘子": "plate",
    "碗": "bowl",
    "托盘": "tray",
    "鸡蛋": "egg",
    "煎鸡蛋": "fried egg",
    "荷包蛋": "fried egg",
    "炒鸡蛋": "scrambled egg",
    "炒蛋": "scrambled egg",
    "蛋": "egg",
    "蛋白": "egg white",
    "蛋清": "egg white",
    "鸡蛋白": "egg white",
    "鸡蛋清": "egg white",
    "蛋黄": "egg yolk",
    "鸡蛋黄": "egg yolk",
    "番茄": "tomato",
    "西红柿": "tomato",
    # SAM3 understands the English food name more reliably than the Chinese
    # token; include the characteristic slice shape/holes in the prompt.
    "藕片": "lotus root slices",
    "莲藕": "lotus root slices",
    "藕": "lotus root slices",
    "人物": "person",
}


def _batch_model_prompt(prompt):
    """Use stable English aliases for common Chinese batch concepts."""
    value = (prompt or "").strip()
    if value in {
        "蛋白", "蛋清", "鸡蛋白", "鸡蛋清",
        "蛋黄", "鸡蛋黄", "egg white", "egg whites", "egg yolk",
    }:
        # SAM3 grounds the parent object more reliably than egg sub-parts.
        # Sub-part extraction is applied after the egg mask is found.
        return "egg"
    if value in _BATCH_PROMPT_ALIASES:
        return _BATCH_PROMPT_ALIASES[value]
    for chinese, english in _BATCH_PROMPT_ALIASES.items():
        if chinese in value and len(value) <= len(chinese) + 4:
            return english
    return value


def _dedupe_batch_masks(masks, overlap_threshold=0.80):
    """Remove duplicate SAM3 candidates while keeping separate objects."""
    kept = []
    for candidate in sorted(
        [ensure_bool_mask(mask) for mask in masks if ensure_bool_mask(mask) is not None],
        key=lambda mask: int(mask.sum()),
        reverse=True,
    ):
        candidate_area = int(candidate.sum())
        if not candidate_area:
            continue
        duplicate = False
        for existing in kept:
            shared = int((candidate & existing).sum())
            smaller = min(candidate_area, int(existing.sum()))
            if shared / max(smaller, 1) >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _refine_egg_subpart_mask(mask, rgb, concept):
    """Refine egg-white/yolk requests inside an already grounded egg mask.

    SAM3 commonly returns the whole fried egg for a sub-part phrase. For
    egg-white requests, remove the saturated orange/yellow yolk pixels from
    that grounded mask while preserving the irregular white perimeter.
    """
    value = (concept or "").strip()
    white_request = value in {
        "蛋白", "蛋清", "鸡蛋白", "鸡蛋清",
        "egg white", "egg whites",
    }
    yolk_request = value in {"蛋黄", "鸡蛋黄", "egg yolk"}
    if not white_request and not yolk_request:
        return ensure_bool_mask(mask)

    base = ensure_bool_mask(mask)
    if base is None or not base.any():
        return base

    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    hue, saturation, brightness = cv2.split(hsv)
    yolk = (
        base
        & (hue >= 4) & (hue <= 38)
        & (saturation >= 70)
        & (brightness >= 75)
    )
    # Close small color gaps so the yolk is removed as one region.
    yolk_u8 = (yolk.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    yolk_u8 = cv2.morphologyEx(yolk_u8, cv2.MORPH_CLOSE, kernel)
    yolk = yolk_u8 > 0
    refined = base & ~yolk if white_request else yolk

    # Do not replace a valid SAM result with an over-aggressive color cut.
    if int(refined.sum()) < max(64, int(base.sum() * 0.10)):
        return base
    return refined

# ============================================================
# 默认配置
# ============================================================
DEFAULT_CONFIG = {
    "MODE": "color_fine",
    "AUTO_SEARCH_ROI": [0, 0, 0, 0],
    "AUTO_MIN_AREA_RATIO": 0.08,
    "AUTO_MAX_AREA_RATIO": 0.6,
    "AUTO_STABILITY_THRESHOLD": 0.7,
    "AUTO_SELECT_BY": "area",
    "AUTO_FINE_BOX_PADDING": 0.2,
    "COLOR_CLOSE_KERNEL": 15,
    "COLOR_MIN_AREA": 500,
    "COLOR_FINE_BOX_PADDING": 0.35,
    "COLOR_FINE_NUM_POINTS": 25,
    "COLOR_FINE_MASK_INTERSECT": False,
    "COLOR_SELECT_BY": "center",
    "COLOR_MERGE_ALL": True,
    "APPLY_MORPHOLOGY": True,
    "MORPH_KERNEL_SIZE": 3,
    "KEEP_LARGEST_COMPONENT": True,
    "COLOR_RANGES": [
        [10,  80,  60,  40, 255, 255],
        [0,   60,  40,  15, 255, 255],
        [165, 60,  40, 179, 255, 255],
        [5,   40,  40,  25, 180, 180],
        [40,  30,  40,  80, 200, 200],
        [0,    0, 120, 179,  40, 255],
    ],
}

PARAM_DOCS = [
    ("MODE",                      "color_fine",        "识别模式: auto / auto_fine / color_fine(推荐)"),
    ("AUTO_SEARCH_ROI",           "0,0,0,0",            "搜索区域 x1,y1,x2,y2；全为0时使用整张图片"),
    ("AUTO_MIN_AREA_RATIO",       "0.08",              "最小面积占搜索区比例"),
    ("AUTO_MAX_AREA_RATIO",       "0.6",               "最大面积占搜索区比例"),
    ("AUTO_STABILITY_THRESHOLD",  "0.7",               "稳定性阈值 (0~1)"),
    ("AUTO_SELECT_BY",            "area",              "auto模式筛选: area / stability"),
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

# ============================================================
# 配置读取
# ============================================================
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
                try:
                    cfg[key] = parse_value(key, row.iloc[1])
                except Exception:
                    pass

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

# ============================================================
# 渲染
# ============================================================
def _draw_chinese_text(img_bgr, text, position, color_bgr, font_size=20):
    """使用 PIL 在图像上绘制中文文字"""
    # 将 OpenCV 图像转换为 PIL 图像
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # 尝试加载中文字体
    font = None
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
        "/System/Library/Fonts/PingFang.ttc",  # macOS
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
    
    # 如果没有找到中文字体，使用默认字体
    if font is None:
        font = ImageFont.load_default()
    
    # 绘制文字（带黑色边框以提高可读性）
    x, y = position
    # 先绘制黑色边框
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    # 再绘制彩色文字
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))  # BGR to RGB
    draw.text((x, y), text, font=font, fill=color_rgb)
    
    # 将 PIL 图像转换回 OpenCV 图像
    img_rgb = np.array(pil_img)
    img_bgr_result = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    return img_bgr_result


def render_composite_to_b64(highlight_id=None, draw_points=True,
                             highlight_new=None, base_fade=BASE_FADE):
    if STATE["image_rgb"] is None:
        return None
    h, w = STATE["h"], STATE["w"]
    if h == 0 or w == 0:
        return None

    base = STATE["image_rgb"].astype(np.float32).copy()
    if base_fade > 0:
        base = base * (1.0 - base_fade) + 255.0 * base_fade

    items = [(iid, inst) for iid, inst in STATE["instances"].items()
             if inst.get("mask") is not None]
    if highlight_id is not None:
        items = [(iid, inst) for iid, inst in items if int(iid) == int(highlight_id)]

    if not items:
        img_rgb = np.clip(base, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")

    items.sort(key=lambda kv: int(kv[1]["mask"].sum()))

    color_map = np.zeros((h, w, 3), dtype=np.float32)
    alpha_map = np.zeros((h, w),    dtype=np.float32)
    covered   = np.zeros((h, w),    dtype=bool)

    for iid, inst in items:
        inst_mask = ensure_bool_mask(inst["mask"])
        m = inst_mask & ~covered
        if not m.any():
            continue
        color_map[m] = get_color(iid)
        a = MASK_ALPHA_NEW if (highlight_new is not None and iid == highlight_new) else MASK_ALPHA
        alpha_map[m] = a
        covered |= inst_mask

    if covered.any():
        a = alpha_map[covered][:, None]
        base[covered] = base[covered] * (1.0 - a) + color_map[covered] * a

    img_rgb = np.clip(base, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 检测任务绘制边界框（不显示文字标签）
    for iid, inst in items:
        if inst.get("task") == "detection" and inst.get("box"):
            c = get_color(iid)
            box_x1, box_y1, box_x2, box_y2 = [int(round(v)) for v in inst["box"]]
            cv2.rectangle(img_bgr, (box_x1, box_y1), (box_x2, box_y2), (int(c[2]), int(c[1]), int(c[0])), 3)

    if highlight_new is not None and highlight_new in STATE["instances"]:
        inst = STATE["instances"][highlight_new]
        if inst.get("mask") is not None:
            mask_bool = ensure_bool_mask(inst["mask"])
            mask_u8 = mask_bool.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            thickness = max(2, min(h, w) // 400)
            cv2.drawContours(img_bgr, contours, -1, (0, 0, 0),     thickness + 3)
            cv2.drawContours(img_bgr, contours, -1, (255, 255, 255), thickness)

    if draw_points:
        for iid, inst in items:
            for (x, y), lab in zip(inst["points"], inst["labels"]):
                px, py = int(round(x)), int(round(y))
                if not (0 <= px < w and 0 <= py < h):
                    continue
                pt_color = (0, 0, 255) if lab == 1 else (255, 0, 0)
                cv2.circle(img_bgr, (px, py), 9, (255, 255, 255), -1)
                cv2.circle(img_bgr, (px, py), 6, pt_color, -1)

    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def render_composite_bgr(base_fade=BASE_FADE, highlight_id=None):
    if STATE["image_rgb"] is None:
        return None
    h, w = STATE["h"], STATE["w"]
    base = STATE["image_rgb"].astype(np.float32).copy()
    if base_fade > 0:
        base = base * (1.0 - base_fade) + 255.0 * base_fade

    items = [(iid, inst) for iid, inst in STATE["instances"].items()
             if inst.get("mask") is not None]
    if highlight_id is not None:
        items = [(iid, inst) for iid, inst in items if int(iid) == int(highlight_id)]
    items.sort(key=lambda kv: int(kv[1]["mask"].sum()))

    color_map = np.zeros((h, w, 3), dtype=np.float32)
    covered   = np.zeros((h, w),    dtype=bool)

    for iid, inst in items:
        inst_mask = ensure_bool_mask(inst["mask"])
        m = inst_mask & ~covered
        if not m.any():
            continue
        color_map[m] = get_color(iid)
        covered |= inst_mask

    if covered.any():
        base[covered] = base[covered] * (1 - MASK_ALPHA) + color_map[covered] * MASK_ALPHA

    rgb = np.clip(base, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def render_composite_only_alpha():
    if STATE["image_rgb"] is None:
        return None
    h, w = STATE["h"], STATE["w"]

    items = [(iid, inst) for iid, inst in STATE["instances"].items()
             if inst.get("mask") is not None]
    items.sort(key=lambda kv: int(kv[1]["mask"].sum()))

    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    covered = np.zeros((h, w), dtype=bool)

    for iid, inst in items:
        inst_mask = ensure_bool_mask(inst["mask"])
        m = inst_mask & ~covered
        if not m.any():
            continue
        c = get_color(iid)
        bgra[m] = [int(c[2]), int(c[1]), int(c[0]), 220]
        covered |= inst_mask

    return bgra


def mask_to_binary_png_bytes(mask_bool):
    mask_bool = ensure_bool_mask(mask_bool)
    mask_u8 = mask_bool.astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", mask_u8)
    return buf.tobytes() if ok else None


def mask_to_color_png_bytes(mask_bool, color_rgb):
    mask_bool = ensure_bool_mask(mask_bool)
    h, w = mask_bool.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_bool] = [int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]), 255]
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    ok, buf = cv2.imencode(".png", bgra)
    return buf.tobytes() if ok else None


def mask_to_png_b64(mask_bool):
    mask_bool = ensure_bool_mask(mask_bool)
    mask_u8 = mask_bool.astype(np.uint8) * 255
    ok, buf = cv2.imencode(".png", mask_u8)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def image_to_jpeg_b64(image_bgr, quality=90):
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else None


def build_contour_image_bgr(mask_bool, color_rgb, image_bgr=None):
    mask_bool = ensure_bool_mask(mask_bool)
    source = image_bgr if image_bgr is not None else STATE["image_bgr"]
    if source is None:
        raise ValueError("缺少轮廓底图")
    img = source.copy()
    h, w = img.shape[:2]
    thickness = max(2, min(h, w) // 400)

    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.drawContours(img, contours, -1, (0, 0, 0), thickness + 4)
    cv2.drawContours(img, contours, -1, bgr,        thickness)
    return img


def run_predict(points=None, labels=None, box=None):
    if not STATE.get("predictor_ready"):
        predictor.set_image(STATE["image_rgb"])
        STATE["predictor_ready"] = True
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with torch.inference_mode(), ctx:
        masks, _, _ = predictor.predict(
            point_coords=points, point_labels=labels,
            box=box, multimask_output=False,
        )
    return ensure_bool_mask(masks[0])


def postprocess_mask(mask, cfg):
    mask = ensure_bool_mask(mask)
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
    return mask_uint8.astype(bool)


def add_title_bar(img_bgr, title):
    h, w = img_bgr.shape[:2]
    bar_h = max(30, h // 40)
    canvas = np.ones((h + bar_h, w, 3), dtype=np.uint8) * 240
    canvas[bar_h:] = img_bgr
    cv2.putText(canvas, title, (10, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
    return canvas


def save_debug_image(img_bgr, title, out_path):
    canvas = add_title_bar(img_bgr, title)
    cv2.imwrite(out_path, canvas)


def pad_to_height(img, target_h, bg_color=(240, 240, 240)):
    h, w = img.shape[:2]
    if h == target_h:
        return img
    if h > target_h:
        scale = target_h / h
        new_w = max(1, int(w * scale))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    canvas = np.ones((target_h, w, 3), dtype=np.uint8)
    canvas[:] = bg_color
    top = (target_h - h) // 2
    canvas[top:top + h] = img
    return canvas


def save_candidates_grid(image_rgb, masks, scores, out_path, box=None):
    tiles = []
    for i, (m, s) in enumerate(zip(masks, scores)):
        m = ensure_bool_mask(m)
        tile = image_rgb.copy()
        c = np.array([255, 100, 100], dtype=np.uint8)
        tile[m] = (tile[m].astype(np.float32) * 0.5 + c * 0.5).astype(np.uint8)
        tb = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        cv2.rectangle(tb, (0, 0), (240, 32), (0, 0, 0), -1)
        cv2.putText(tb, f"Mask {i+1}  score={s:.3f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(tb, (x1, y1), (x2, y2), (0, 255, 255), 2)
        tiles.append(tb)

    if tiles:
        target_h = max(t.shape[0] for t in tiles)
        tiles = [pad_to_height(t, target_h) for t in tiles]
        grid = np.hstack(tiles)
        cv2.imwrite(out_path, grid)


def resolve_roi(image_rgb, cfg, roi=None):
    h, w = image_rgb.shape[:2]
    values = roi if roi is not None else cfg.get("AUTO_SEARCH_ROI")
    if not values or len(values) != 4:
        return [0, 0, w, h]
    x1, y1, x2, y2 = [int(float(v)) for v in values]
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return [0, 0, w, h]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return [x1, y1, x2, y2]


def _mask_bbox(mask, padding, image_shape):
    ys, xs = np.where(ensure_bool_mask(mask))
    if len(xs) == 0:
        return None
    h, w = image_shape[:2]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad_x = max(4, int((x2 - x1 + 1) * float(padding)))
    pad_y = max(4, int((y2 - y1 + 1) * float(padding)))
    return [max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(w, x2 + pad_x + 1), min(h, y2 + pad_y + 1)]


def _sample_positive_points(mask, max_points=25):
    ys, xs = np.where(ensure_bool_mask(mask))
    if len(xs) == 0:
        return None, None
    count = min(int(max_points), len(xs))
    indices = np.linspace(0, len(xs) - 1, count, dtype=np.int64)
    points = np.column_stack([xs[indices], ys[indices]]).astype(np.float32)
    return points, np.ones(len(points), dtype=np.int32)


def _predict_from_seed(image_rgb, seed_mask, box, cfg, out_path=None):
    points, labels = _sample_positive_points(
        seed_mask, cfg.get("COLOR_FINE_NUM_POINTS", 25))
    if points is None or box is None:
        return None
    predictor.set_image(image_rgb)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) \
        if device == "cuda" else nullcontext()
    with torch.inference_mode(), ctx:
        masks, scores, _ = predictor.predict(
            point_coords=points, point_labels=labels,
            box=np.array(box, dtype=np.float32), multimask_output=False,
        )
    result_mask = ensure_bool_mask(masks[0])
    if cfg.get("COLOR_FINE_MASK_INTERSECT", False):
        result_mask &= ensure_bool_mask(seed_mask)
    result_mask = postprocess_mask(result_mask, cfg)
    if out_path:
        save_debug_image(build_contour_image_bgr(
            result_mask, [255, 120, 40], cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)),
                         f"result area={int(result_mask.sum())}", out_path)
    print(f"[refine] SAM 精修面积={int(result_mask.sum())}, "
          f"score={float(scores[0]):.3f}")
    return result_mask


def run_mode_color_fine(image_rgb, cfg, out_dir):
    x1, y1, x2, y2 = resolve_roi(image_rgb, cfg)
    roi = image_rgb[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    color_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    for value_range in cfg["COLOR_RANGES"]:
        color_mask |= cv2.inRange(
            hsv, np.array(value_range[:3], dtype=np.uint8),
            np.array(value_range[3:], dtype=np.uint8))

    color_px = int((color_mask > 0).sum())
    print(f"[color_fine] ROI: [{x1},{y1},{x2},{y2}], 颜色命中: {color_px} px")
    if color_px == 0:
        print("颜色检测没有找到任何食物色区域")
        return None

    kernel_size = max(1, int(cfg["COLOR_CLOSE_KERNEL"]))
    if kernel_size > 1:
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_CLOSE,
            np.ones((kernel_size, kernel_size), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(color_mask, 8)
    components = []
    full_image_roi = x1 == 0 and y1 == 0 and x2 == image_rgb.shape[1] and y2 == image_rgb.shape[0]
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < int(cfg["COLOR_MIN_AREA"]):
            continue
        touches_border = (
            stats[label, cv2.CC_STAT_LEFT] == 0
            or stats[label, cv2.CC_STAT_TOP] == 0
            or stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH] >= roi.shape[1]
            or stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT] >= roi.shape[0]
        )
        if full_image_roi and touches_border:
            continue
        components.append(label)
    print(f"[color_fine] 有效颜色区域: {len(components)}")
    if not components:
        print("颜色检测到的区域都太小")
        return None

    selected = np.zeros_like(color_mask, dtype=bool)
    if cfg.get("COLOR_MERGE_ALL", True):
        for label in components:
            selected |= labels == label
    else:
        if cfg.get("COLOR_SELECT_BY", "center") == "area":
            label = max(components, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        else:
            cx, cy = roi.shape[1] / 2, roi.shape[0] / 2
            label = min(components, key=lambda i: (
                (stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2 - cx) ** 2
                + (stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] / 2 - cy) ** 2))
        selected = labels == label

    full_seed = np.zeros(image_rgb.shape[:2], dtype=bool)
    full_seed[y1:y2, x1:x2] = selected
    box = _mask_bbox(full_seed, cfg["COLOR_FINE_BOX_PADDING"], image_rgb.shape)
    result_mask = _predict_from_seed(
        image_rgb, full_seed, box, cfg,
        os.path.join(out_dir, "color_fine_result.png"))
    return {"result_mask": result_mask, "source": "color_fine"} \
        if result_mask is not None else None


def _select_amg_result(image_rgb, cfg):
    x1, y1, x2, y2 = resolve_roi(image_rgb, cfg)
    roi_area = max(1, (x2 - x1) * (y2 - y1))
    candidates = []
    for item in mask_generator.generate(image_rgb):
        mask = ensure_bool_mask(item["segmentation"])
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        inside = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
        ratio = int(inside.sum()) / roi_area
        if cfg["AUTO_MIN_AREA_RATIO"] <= ratio <= cfg["AUTO_MAX_AREA_RATIO"]:
            candidates.append((item, mask))
    if not candidates:
        print("搜索区域内没有找到符合条件的物体")
        return None
    key = "predicted_iou" if cfg["AUTO_SELECT_BY"] == "stability" else "area"
    candidates.sort(key=lambda pair: pair[0].get(key, 0), reverse=True)
    print(f"[auto] 共检测到 {len(candidates)} 个物体")
    return candidates[0][1]


def run_mode_auto(image_rgb, cfg, out_dir):
    result_mask = _select_amg_result(image_rgb, cfg)
    if result_mask is None:
        return None
    result_mask = postprocess_mask(result_mask, cfg)
    save_debug_image(build_contour_image_bgr(
        result_mask, [255, 120, 40], cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)),
                     f"auto area={int(result_mask.sum())}",
                     os.path.join(out_dir, "auto_result.png"))
    return {"result_mask": result_mask, "source": "auto"}


def run_mode_auto_fine(image_rgb, cfg, out_dir):
    seed_mask = _select_amg_result(image_rgb, cfg)
    if seed_mask is None:
        return None
    box = _mask_bbox(seed_mask, cfg["AUTO_FINE_BOX_PADDING"], image_rgb.shape)
    result_mask = _predict_from_seed(
        image_rgb, seed_mask, box, cfg,
        os.path.join(out_dir, "auto_fine_result.png"))
    return {"result_mask": result_mask, "source": "auto_fine"} \
        if result_mask is not None else None


# ============================================================
# ⭐ 新增：碎丁合并核心函数
# ============================================================
def merge_pile_to_mask(image_rgb, cfg,
                        roi=None,
                        close_kernel=25,
                        dilate_iter=2,
                        erode_iter=2,
                        min_component_area=50,
                        use_sam_refine=True,
                        log_fn=print):
    """
    把一盘碎丁合并成一张 mask：
      ① HSV 颜色掩码
      ② 形态学闭运算 + 膨胀 → 近处丁连成片
      ③ 连通域分析 → 合并所有有效区域
      ④ 再用 SAM 精修边缘（可选）

    返回 (final_mask, debug_info_dict) 或 (None, info)
    """
    H, W = image_rgb.shape[:2]
    log_fn(f"[pile] 图片尺寸: {W} x {H}")

    # ---- ROI ----
    x1, y1, x2, y2 = resolve_roi(image_rgb, cfg, roi)
    log_fn(f"[pile] ROI: [{x1},{y1},{x2},{y2}] (size {x2-x1}x{y2-y1})")

    roi_img = image_rgb[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi_img, cv2.COLOR_RGB2HSV)

    # ---- 颜色掩码 ----
    mask_all = np.zeros((roi_img.shape[0], roi_img.shape[1]), dtype=np.uint8)
    for rng in cfg["COLOR_RANGES"]:
        lower = np.array(rng[:3], dtype=np.uint8)
        upper = np.array(rng[3:], dtype=np.uint8)
        mask_all |= cv2.inRange(hsv, lower, upper)

    color_px = int((mask_all > 0).sum())
    log_fn(f"[pile] 颜色命中: {color_px} px")
    if color_px == 0:
        return None, {"stage": "color", "msg": "颜色掩码为空，检查 ROI 或 HSV"}

    # ---- 形态学 ----
    k = int(close_kernel)
    if k > 1:
        mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_CLOSE,
                                     np.ones((k, k), np.uint8))
    if dilate_iter > 0:
        mask_all = cv2.dilate(mask_all, np.ones((5, 5), np.uint8),
                               iterations=int(dilate_iter))
    if erode_iter > 0:
        mask_all = cv2.erode(mask_all, np.ones((5, 5), np.uint8),
                              iterations=int(erode_iter))

    morph_px = int((mask_all > 0).sum())
    log_fn(f"[pile] 形态学后: {morph_px} px")

    # ---- 连通域 ----
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_all, connectivity=8)
    log_fn(f"[pile] 连通域数: {num - 1}")

    if num <= 1:
        return None, {"stage": "morph", "msg": "形态学后无连通域"}

    comps = []
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_component_area:
            continue
        comps.append({"label": i, "area": a})

    log_fn(f"[pile] 有效连通域: {len(comps)} (面积 ≥ {min_component_area})")
    if not comps:
        return None, {"stage": "comp", "msg": "没有面积达标的小丁"}

    # ---- 合并所有有效区域 ----
    food_mask_roi = np.zeros_like(mask_all)
    for c in comps:
        food_mask_roi |= (labels == c["label"]).astype(np.uint8)

    # 再做一次闭运算，补漏
    if k > 1:
        food_mask_roi = cv2.morphologyEx(food_mask_roi, cv2.MORPH_CLOSE,
                                          np.ones((k * 2, k * 2), np.uint8))

    total_area = int(food_mask_roi.sum())
    log_fn(f"[pile] 合并后总面积: {total_area} px")

    if total_area == 0:
        return None, {"stage": "merge", "msg": "合并后为空"}

    # ---- 全图坐标 ----
    full_mask = np.zeros((H, W), dtype=bool)
    full_mask[y1:y2, x1:x2] = food_mask_roi.astype(bool)

    # ---- SAM 精修（可选）----
    final_mask = full_mask
    sam_area = 0
    if use_sam_refine:
        ys, xs = np.where(full_mask)
        if len(xs) > 0:
            bx, by = int(xs.min()), int(ys.min())
            bx2, by2 = int(xs.max()), int(ys.max())
            pad = 20
            box = [max(0, bx - pad), max(0, by - pad),
                   min(W, bx2 + pad), min(H, by2 + pad)]

            n_pts = min(30, len(xs))
            idx = np.random.choice(len(xs), n_pts, replace=False)
            points = np.column_stack([xs[idx], ys[idx]]).astype(np.float32)
            labels_pts = np.ones(len(points), dtype=np.int32)

            log_fn(f"[pile] SAM 精修：采样 {len(points)} 个正点，box={box}")
            predictor.set_image(image_rgb)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) \
                  if device == "cuda" else nullcontext()
            try:
                with torch.inference_mode(), ctx:
                    sam_masks, scores, _ = predictor.predict(
                        point_coords=points, point_labels=labels_pts,
                        box=np.array(box), multimask_output=False,
                    )
                sam_mask = ensure_bool_mask(sam_masks[0])
                sam_area = int(sam_mask.sum())

                mask_dilated = cv2.dilate(full_mask.astype(np.uint8),
                                            np.ones((7, 7), np.uint8),
                                            iterations=1).astype(bool)
                candidate = sam_mask & mask_dilated

                if candidate.sum() < total_area * 0.3:
                    log_fn(f"[pile] SAM 面积 {sam_area} 太小，回退到纯合并 mask")
                    final_mask = full_mask
                else:
                    log_fn(f"[pile] 使用 SAM 精修 mask，面积={int(candidate.sum())}, "
                           f"score={float(scores[0]):.3f}")
                    final_mask = candidate
            except Exception as e:
                log_fn(f"[pile] SAM 精修失败: {e}，回退纯合并 mask")
                final_mask = full_mask

    return ensure_bool_mask(final_mask), {
        "roi": [x1, y1, x2, y2],
        "color_px": color_px,
        "morph_px": morph_px,
        "n_components": len(comps),
        "total_area": total_area,
        "sam_area": sam_area,
    }


# ============================================================
# 请求模型
# ============================================================
class PointRequest(BaseModel):
    x: float
    y: float
    label: int = 1

class BoxRequest(BaseModel):
    x1: float; y1: float; x2: float; y2: float

class SelectInstanceReq(BaseModel):
    id: int

class UpdateInstanceReq(BaseModel):
    id: int
    name: Optional[str] = None

class MorphInstanceReq(BaseModel):
    id: int
    amount: int = 0
    commit: bool = False
    reset: bool = False

class PointPrompt(BaseModel):
    x: float; y: float; label: int

class PromptRequest(BaseModel):
    points: Optional[List[PointPrompt]] = None
    box:    Optional[List[float]]       = None
    name:   str = "object"

class TextDetectRequest(BaseModel):
    names: List[str]
    confidence: float = 0.25
    iou: float = 0.45
    model: str = "yolov8s-world.pt"
    task: str = "segmentation"


class Sam3ImagePromptRequest(BaseModel):
    name: str
    confidence: float = 0.35

class PileRequest(BaseModel):
    """碎丁合并参数"""
    roi: Optional[List[int]] = None
    close_kernel: int = 25
    dilate_iter: int = 2
    erode_iter: int = 2
    min_component_area: int = 50
    use_sam_refine: bool = True


# ============================================================
# POST /upload
# ============================================================
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    t0 = time.time()
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"status": "error", "msg": "无法解码图片"}

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    print(f"[upload] 收到图片 {w}x{h}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUTPUT_ROOT, ts)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "original.png"), bgr)

    STATE.update({
        "image_bgr": bgr, "image_rgb": rgb, "h": h, "w": w,
        "instances": {}, "next_id": 1, "current_id": None,
        "predictor_ready": False,
        "output_dir": out_dir,
        "last_log": "",
    })

    print(f"[upload] 图片已就绪，等待用户选择检测或分割 "
          f"({time.time()-t0:.2f}s)")

    return {"status": "ok", "image_h": h, "image_w": w,
            "count": len(STATE["instances"])}

# ============================================================
# POST /recognize
# ============================================================
@app.post("/recognize")
@serialized_inference
async def recognize(
    mode: str = Query(default="color_fine"),
    config_file: Optional[UploadFile] = File(default=None),
):
    if STATE["image_rgb"] is None:
        return {"status": "error", "msg": "请先上传图片", "log": ""}

    if mode not in ("auto", "auto_fine", "color_fine"):
        return {"status": "error", "msg": f"未知模式: {mode}", "log": ""}

    cfg = {k: (v.copy() if isinstance(v, list) else v) for k, v in DEFAULT_CONFIG.items()}
    if config_file is not None and config_file.filename:
        try:
            content = await config_file.read()
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            cfg = load_config(tmp_path)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            print(f"[recognize] 加载配置文件: {config_file.filename}")
        except Exception as e:
            print(f"[recognize] 配置读取失败: {e}，使用默认参数")

    cfg["MODE"] = mode
    out_dir = STATE.get("output_dir") or os.path.join(OUTPUT_ROOT, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    log_buf = io.StringIO()
    result = None
    err_text = None

    with contextlib.redirect_stdout(log_buf):
        print(f"[recognize] 模式={mode}")
        print(f"[recognize] 图片尺寸: {STATE['w']} x {STATE['h']}")
        print(f"[recognize] ROI: {resolve_roi(STATE['image_rgb'], cfg)}")
        print(f"[recognize] 颜色范围组数: {len(cfg['COLOR_RANGES'])}")
        try:
            if mode == "auto":
                result = run_mode_auto(STATE["image_rgb"], cfg, out_dir)
            elif mode == "auto_fine":
                result = run_mode_auto_fine(STATE["image_rgb"], cfg, out_dir)
            else:
                result = run_mode_color_fine(STATE["image_rgb"], cfg, out_dir)
        except Exception as e:
            err_text = f"{type(e).__name__}: {e}"
            print(f"[recognize] EXCEPTION: {err_text}")
            traceback.print_exc(file=log_buf)

    log_text = log_buf.getvalue()
    STATE["last_log"] = log_text

    print("=" * 60)
    print(log_text)
    print("=" * 60)

    if err_text is not None:
        return {"status": "error", "msg": f"识别异常: {err_text}", "log": log_text}

    if result is None or result.get("result_mask") is None:
        hint = _diagnose_failure(mode, cfg, log_text)
        return {"status": "error", "msg": hint, "log": log_text}

    result_mask = ensure_bool_mask(result["result_mask"])
    iid = STATE["next_id"]; STATE["next_id"] += 1
    STATE["instances"][iid] = {
        "points": [], "labels": [], "box": None,
        "mask": result_mask,
        "amg": False,
        "source": result.get("source", mode),
    }
    STATE["current_id"] = iid

    color = get_color(iid)
    print(f"[recognize] 结果已加入实例 id={iid}")

    return {
        "status":     "ok",
        "id":         iid,
        "color":      [int(c) for c in color],
        "source":     result.get("source", mode),
        "area":       int(result_mask.sum()),
        "output_dir": out_dir,
        "log":        log_text,
        "rendered":   render_composite_to_b64(highlight_id=None, highlight_new=iid),
    }


def _diagnose_failure(mode, cfg, log_text):
    if mode == "color_fine":
        if "颜色检测没有找到任何食物色区域" in log_text:
            return ("颜色检测失败：ROI 区域内没有找到符合 HSV 范围的颜色。\n"
                    "→ 可能原因：\n"
                    "  1. ROI 位置不对（食物不在搜索区域内）\n"
                    "  2. 食物颜色不在 COLOR_RANGES 的 6 组 HSV 范围内\n"
                    "  3. 图片太暗/太亮导致颜色偏差\n"
                    "建议：下载配置模板，调整 AUTO_SEARCH_ROI 和 COLOR_RANGES 后重新上传")
        if "颜色检测到的区域都太小" in log_text:
            return (f"颜色检测到的区域都太小\n"
                    f"→ 请调小 COLOR_MIN_AREA（当前值: {cfg['COLOR_MIN_AREA']}）")
        if "ROI 为空" in log_text:
            return "ROI 越界 → AUTO_SEARCH_ROI 超出图片尺寸，请检查配置"
    if mode in ("auto", "auto_fine"):
        if "搜索区域内没有找到符合条件的物体" in log_text:
            return ("自动检测未在 ROI 内找到符合条件的物体\n"
                    "→ 可能原因：\n"
                    "  1. AUTO_SEARCH_ROI 位置不对\n"
                    "  2. 面积阈值 AUTO_MIN/MAX_AREA_RATIO 太严格\n"
                    "  3. AUTO_STABILITY_THRESHOLD 太高")
        if "共检测到 0 个物体" in log_text:
            return "SAM 未检测到任何物体 → 图片可能太小或太模糊"
    return "识别失败，请查看下方日志了解详情"


# ============================================================
# ⭐ 新增：POST /recognize_pile  —— 碎丁合并
# ============================================================
@app.post("/recognize_pile")
@serialized_inference
async def recognize_pile(req: PileRequest):
    """
    把一盘碎丁合并成一个 mask：
      HSV 颜色 → 形态学连接 → 合并所有丁 → SAM 精修 → 加入实例
    """
    if STATE["image_rgb"] is None:
        return {"status": "error", "msg": "请先上传图片", "log": ""}

    cfg = {k: (v.copy() if isinstance(v, list) else v) for k, v in DEFAULT_CONFIG.items()}
    out_dir = STATE.get("output_dir") or os.path.join(OUTPUT_ROOT, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    log_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_buf):
            print(f"[recognize_pile] 参数: close_kernel={req.close_kernel}, "
                  f"dilate_iter={req.dilate_iter}, erode_iter={req.erode_iter}, "
                  f"min_component_area={req.min_component_area}, "
                  f"use_sam_refine={req.use_sam_refine}")

            final_mask, info = merge_pile_to_mask(
                STATE["image_rgb"], cfg,
                roi=req.roi,
                close_kernel=req.close_kernel,
                dilate_iter=req.dilate_iter,
                erode_iter=req.erode_iter,
                min_component_area=req.min_component_area,
                use_sam_refine=req.use_sam_refine,
                log_fn=print,
            )

            if final_mask is None:
                return {
                    "status": "error",
                    "msg": f"合并失败（{info.get('stage','?')}）：{info.get('msg','')}",
                    "log": log_buf.getvalue(),
                }

            try:
                ov = STATE["image_rgb"].copy().astype(np.float32)
                c = np.array([30, 144, 255], dtype=np.float32)
                ov[final_mask] = ov[final_mask] * 0.5 + c * 0.5
                ov = np.clip(ov, 0, 255).astype(np.uint8)
                save_debug_image(
                    cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
                    f"pile_result (area={int(final_mask.sum())})",
                    os.path.join(out_dir, "pile_result.png"),
                )
            except Exception as e:
                print(f"[recognize_pile] 保存过程图失败: {e}")

    except Exception as e:
        print("[recognize_pile] EXCEPTION:")
        traceback.print_exc()
        return {"status": "error",
                "msg": f"{type(e).__name__}: {e}",
                "log": log_buf.getvalue()}

    log_text = log_buf.getvalue()
    STATE["last_log"] = log_text

    # ---- 加入实例 ----
    iid = STATE["next_id"]; STATE["next_id"] += 1
    STATE["instances"][iid] = {
        "points": [], "labels": [], "box": None,
        "mask": final_mask,
        "amg": False,
        "source": "pile",
    }
    STATE["current_id"] = iid

    color = get_color(iid)
    print(f"[recognize_pile] ✅ 已加入实例 id={iid}, 面积={int(final_mask.sum())}")

    return {
        "status":     "ok",
        "id":         iid,
        "color":      [int(c) for c in color],
        "source":     "pile",
        "area":       int(final_mask.sum()),
        "info":       info,
        "output_dir": out_dir,
        "log":        log_text,
        "rendered":   render_composite_to_b64(highlight_id=None, highlight_new=iid),
    }


# ============================================================
# ⭐ 新增：POST /recognize_batch  —— 批量识别
# ============================================================
@app.post("/recognize_batch")
async def recognize_batch(
    files: List[UploadFile] = File(...),
    mode: Optional[str] = Form(default=None),
    concept_name: Optional[str] = Form(default=None),
    container_concept: Optional[str] = Form(default="锅"),
    prompt_box: Optional[str] = Form(default=None),
    prompt_points: Optional[str] = Form(default=None),
    prompt_labels: Optional[str] = Form(default=None),
    target_prompt_points: Optional[str] = Form(default=None),
    target_prompt_labels: Optional[str] = Form(default=None),
    mode_query: Optional[str] = Query(default=None, alias="mode"),
    config_file: Optional[UploadFile] = File(default=None),
):
    """
    批量识别：上传多张图片 → 逐张使用 SAM3 图像模型 → 打包 zip 下载

    每张图片的处理流程：
      ① 内存中解码
      ② 先用 SAM3 识别限制区域（默认"锅"）
      ③ 在限制区域裁剪图像内用 SAM3 识别目标文字（支持多个概念，逗号分隔）
      ④ 使用首张图提示选择正确的限制区域，后续图片不复用首图坐标
      ⑤ 保存原图 + 结果图 + JSON 到临时文件夹
    最后把所有这些文件打包成 zip，返回给前端。
    """
    if not files:
        return {"status": "error", "msg": "没有上传文件", "log": ""}

    mode = (mode or mode_query or "sam3").strip().lower()
    if mode != "sam3":
        return {"status": "error", "msg": "批量处理已固定使用 SAM3", "log": ""}
    try:
        prompt_box_data = json.loads(prompt_box) if prompt_box else None
        prompt_points_data = json.loads(prompt_points) if prompt_points else []
        prompt_labels_data = json.loads(prompt_labels) if prompt_labels else []
        target_prompt_points_data = json.loads(target_prompt_points) if target_prompt_points else []
        target_prompt_labels_data = json.loads(target_prompt_labels) if target_prompt_labels else []
    except json.JSONDecodeError:
        return {"status": "error", "msg": "首张图片提示格式无效", "log": ""}
    
    # 支持多个概念（逗号分隔，支持中英文逗号）
    concept_str = (concept_name or "").strip()
    # 将中文逗号替换为英文逗号
    concept_str = concept_str.replace("，", ",")
    concepts = [c.strip() for c in concept_str.split(",") if c.strip()] if concept_str else []
    container = (container_concept or "").strip()
    
    # 映射所有概念
    model_concepts = [_batch_model_prompt(c) for c in concepts]
    model_container = _batch_model_prompt(container)
    
    if concepts:
        print(f"[batch] 提示映射：目标={concepts!r}->{model_concepts!r}，限制区域={container!r}->{model_container!r}")
    else:
        print(f"[batch] 提示映射：限制区域={container!r}->{model_container!r}")
    
    if len(files) > 1 and not concepts:
        return {
            "status": "error",
            "msg": "多图批量必须填写目标文字（例如：煎鸡蛋,西红柿）；首图点击只能帮助选择锅，不能单独传递到后续图片",
            "log": "",
        }
    if container and not concepts and (
        prompt_box_data or prompt_points_data or target_prompt_points_data
    ):
        return {
            "status": "error",
            "msg": "已使用锅区域限制，请填写目标文字；否则结果只会是锅本身",
            "log": "",
        }
    if not concepts and not prompt_box_data and not prompt_points_data and not container:
        return {"status": "error", "msg": "请输入目标文字、限制区域，或先在第一张图片上点击/框选目标", "log": ""}

    log_buf = io.StringIO()
    results_summary = []

    async with INFERENCE_LOCK:
        with contextlib.redirect_stdout(log_buf):
            print(f"[batch] 收到 {len(files)} 张图片，模式={mode}")
            if concepts:
                print(f"[batch] 将检测 {len(concepts)} 个目标: {concepts}")

        # 临时输出目录
        ts = time.strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(OUTPUT_ROOT, f"batch_{ts}")
        os.makedirs(batch_dir, exist_ok=True)

        if config_file is not None and config_file.filename:
            print(f"[batch] 忽略旧版批量配置文件：{config_file.filename}（批量处理固定使用 SAM3）")

        for idx, f in enumerate(files):
            name = f.filename or f"image_{idx:03d}.png"
            base_name = os.path.splitext(os.path.basename(name))[0] or f"image_{idx:03d}"
            print(f"\n[batch] ({idx+1}/{len(files)}) 处理: {name}")

            try:
                raw = await f.read()
                arr = np.frombuffer(raw, np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    print(f"[batch] 跳过：无法解码 {name}")
                    results_summary.append({"file": name, "status": "error",
                                             "msg": "无法解码"})
                    continue

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                H, W = rgb.shape[:2]

                # 保存原图
                img_dir = os.path.join(batch_dir, f"{idx:03d}_{base_name}")
                os.makedirs(img_dir, exist_ok=True)
                cv2.imwrite(os.path.join(img_dir, "original.png"), bgr)

                image_log = io.StringIO()
                with contextlib.redirect_stdout(image_log):
                    t0 = time.time()
                    _, sam3_processor = _sam3_load_image_model()
                    from PIL import Image
                    sam3_processor.set_confidence_threshold(0.35)
                    first_prompt = idx == 0
                    first_container_points = prompt_points_data if first_prompt else []
                    first_container_labels = prompt_labels_data if first_prompt else []
                    first_target_points = target_prompt_points_data if first_prompt else []
                    first_target_labels = target_prompt_labels_data if first_prompt else []
                    first_box = prompt_box_data if first_prompt else None
                    container_reference_point = _batch_prompt_reference_point(
                        prompt_points_data, prompt_labels_data, prompt_box_data, W, H
                    )

                    if container:
                        # Stage 1: independently ground the container on every
                        # image. The first-image point/box only selects which
                        # container candidate to use; it is never copied to
                        # later images.
                        print(f"[batch] 正在识别限制区域: {container}")
                        region_state = sam3_processor.set_image(Image.fromarray(rgb))
                        region_output = sam3_processor.set_text_prompt(
                            prompt=model_container, state=region_state
                        )
                        region_masks = _sam3_masks_from_output(region_output)
                        print(f"[batch] 限制区域 '{container}' 识别到 {len(region_masks)} 个候选 mask")
                        if not region_masks:
                            # Container wording can be less confident than
                            # the target wording. Retry the same text prompt
                            # before reporting that the ROI is missing.
                            sam3_processor.set_confidence_threshold(0.15)
                            region_output = sam3_processor.set_confidence_threshold(
                                0.15, state=region_state
                            )
                            region_masks = _sam3_masks_from_output(region_output)
                            print(f"[batch] 降低置信度后识别到 {len(region_masks)} 个候选 mask")
                        container_mask = _choose_container_mask(
                            region_masks, W, H,
                            first_container_points, first_container_labels, first_box,
                            reference_point=container_reference_point
                        )
                        if container_mask is None:
                            result = None
                            print(f"[batch] ❌ 未识别到限制区域: {container}")
                            print(f"[batch] 提示：请确保限制区域输入正确，或在第一张图上点击/框选限制区域")
                        elif not concepts:
                            # Without a target concept, the explicit result is
                            # the selected container itself.
                            result = {
                                "result_mask": container_mask,
                                "source": "sam3_container",
                                "instances": np.asarray([container_mask]),
                                "container_mask": container_mask,
                                "concept_instances": [],
                            }
                        else:
                            # Stage 2: crop to the detected container before
                            # grounding the target concepts. This gives SAM3 a
                            # real spatial restriction instead of merely
                            # filtering a full-image text result afterwards.
                            region_bbox = _batch_mask_bbox(container_mask)
                            x0, y0, x1, y1 = region_bbox
                            print(f"[batch] ✅ 限制区域识别成功: bbox=[{x0},{y0},{x1},{y1}], 面积={int(container_mask.sum())} 像素")
                            pad_x = max(8, int((x1 - x0) * 0.08))
                            pad_y = max(8, int((y1 - y0) * 0.08))
                            crop_x = max(0, x0 - pad_x)
                            crop_y = max(0, y0 - pad_y)
                            crop_x2 = min(W, x1 + pad_x)
                            crop_y2 = min(H, y1 + pad_y)
                            crop = rgb[crop_y:crop_y2, crop_x:crop_x2]
                            crop_h, crop_w = crop.shape[:2]
                            print(f"[batch] 裁剪区域: [{crop_x},{crop_y},{crop_x2},{crop_y2}], 尺寸={crop_w}x{crop_h}")
                            
                            # 对每个概念分别检测
                            all_concept_instances = []
                            for concept_idx, (concept, model_concept) in enumerate(zip(concepts, model_concepts)):
                                print(f"[batch] 检测概念 {concept_idx+1}/{len(concepts)}: {concept}")
                                target_state = sam3_processor.set_image(
                                    Image.fromarray(crop)
                                )
                                target_output = sam3_processor.set_text_prompt(
                                    prompt=model_concept, state=target_state
                                )

                                if first_target_points and concept_idx == 0:
                                    for point_idx, point in enumerate(first_target_points):
                                        point_label = bool(
                                            first_target_labels[point_idx]
                                            if point_idx < len(first_target_labels)
                                            else True
                                        )
                                        crop_point = [
                                            (float(point[0]) - crop_x) / max(crop_w, 1),
                                            (float(point[1]) - crop_y) / max(crop_h, 1),
                                        ]
                                        target_output = sam3_processor.add_point_prompt(
                                            crop_point, point_label, target_state
                                        )
                                elif first_box and len(first_box) == 4 and concept_idx == 0:
                                    bx1, by1, bx2, by2 = [float(v) for v in first_box]
                                    crop_box = [
                                        ((bx1 + bx2) / 2 - crop_x) / max(crop_w, 1),
                                        ((by1 + by2) / 2 - crop_y) / max(crop_h, 1),
                                        max(1.0, bx2 - bx1) / max(crop_w, 1),
                                        max(1.0, by2 - by1) / max(crop_h, 1),
                                    ]
                                    target_output = sam3_processor.add_geometric_prompt(
                                        crop_box, True, target_state
                                    )

                                target_masks = _sam3_masks_from_output(target_output)
                                if not target_masks:
                                    sam3_processor.set_confidence_threshold(0.15)
                                    target_output = sam3_processor.set_confidence_threshold(
                                        0.15, state=target_state
                                    )
                                    target_masks = _sam3_masks_from_output(target_output)

                                full_target_masks = [
                                    _map_crop_mask(mask, crop_x, crop_y, H, W)
                                    for mask in target_masks
                                ]
                                full_target_masks = [
                                    _refine_egg_subpart_mask(mask, rgb, concept)
                                    for mask in full_target_masks
                                ]
                                filtered_masks = []
                                for target_mask in full_target_masks:
                                    target_area = int(target_mask.sum())
                                    if target_area == 0:
                                        continue
                                    overlap = int((target_mask & container_mask).sum())
                                    overlap_ratio = overlap / max(target_area, 1)
                                    roi_area = int(target_mask[y0:y1, x0:x1].sum())
                                    roi_ratio = roi_area / max(target_area, 1)
                                    ys, xs = np.where(target_mask)
                                    center_inside = bool(
                                        container_mask[
                                            int(np.mean(ys)), int(np.mean(xs))
                                        ]
                                    ) if len(xs) else False
                                    roi_center_inside = bool(
                                        len(xs) and x0 <= int(np.mean(xs)) < x1
                                        and y0 <= int(np.mean(ys)) < y1
                                    )
                                    if overlap_ratio >= 0.30 or (
                                        center_inside and overlap_ratio >= 0.12
                                    ) or (
                                        roi_center_inside and roi_ratio >= 0.30
                                    ):
                                        filtered_masks.append(target_mask)
                                
                                if not filtered_masks:
                                    # Some concepts are grounded more reliably
                                    # on the original image than on a tight crop.
                                    # Keep the two-stage spatial restriction by
                                    # running the target again on the full image
                                    # and applying the same container-mask test.
                                    full_target_state = sam3_processor.set_image(
                                        Image.fromarray(rgb)
                                    )
                                    sam3_processor.set_confidence_threshold(0.15)
                                    full_target_output = sam3_processor.set_text_prompt(
                                        prompt=model_concept, state=full_target_state
                                    )
                                    for target_mask in _sam3_masks_from_output(
                                        full_target_output
                                    ):
                                        target_mask = _refine_egg_subpart_mask(
                                            target_mask, rgb, concept
                                        )
                                        target_area = int(target_mask.sum())
                                        if target_area == 0:
                                            continue
                                        overlap = int((target_mask & container_mask).sum())
                                        overlap_ratio = overlap / max(target_area, 1)
                                        roi_area = int(target_mask[y0:y1, x0:x1].sum())
                                        roi_ratio = roi_area / max(target_area, 1)
                                        ys, xs = np.where(target_mask)
                                        center_inside = bool(
                                            container_mask[
                                                int(np.mean(ys)), int(np.mean(xs))
                                            ]
                                        ) if len(xs) else False
                                        roi_center_inside = bool(
                                            len(xs) and x0 <= int(np.mean(xs)) < x1
                                            and y0 <= int(np.mean(ys)) < y1
                                        )
                                        if overlap_ratio >= 0.30 or (
                                            center_inside and overlap_ratio >= 0.12
                                        ) or (
                                            roi_center_inside and roi_ratio >= 0.30
                                        ):
                                            filtered_masks.append(target_mask)
                                
                                filtered_masks = _dedupe_batch_masks(filtered_masks)
                                
                                # 过滤掉面积过大的 mask（可能是模型错误识别了整个容器区域）
                                container_area = int(container_mask.sum())
                                max_allowed_ratio = 0.50  # 最大允许占容器面积的比例
                                final_masks = []
                                for mask in filtered_masks:
                                    mask_area = int(mask.sum())
                                    ratio = mask_area / max(container_area, 1)
                                    if ratio > max_allowed_ratio:
                                        print(f"[batch] 概念 '{concept}' 的 mask 面积过大 ({ratio:.1%} of container)，已过滤")
                                    else:
                                        final_masks.append(mask)
                                
                                print(f"[batch] 概念 '{concept}' 检测到 {len(final_masks)} 个实例")
                                
                                # 保存这个概念的实例
                                for mask in final_masks:
                                    all_concept_instances.append({
                                        "concept": concept,
                                        "mask": mask,
                                    })
                            
                            if not all_concept_instances:
                                result = None
                                print(f"[batch] 未在限制区域内识别到任何目标")
                            else:
                                # 合并所有概念的 mask
                                all_masks = [inst["mask"] for inst in all_concept_instances]
                                result = {
                                    "result_mask": np.any(all_masks, axis=0),
                                    "source": "sam3_container_text_multi",
                                    "instances": np.asarray(all_masks),
                                    "container_mask": container_mask,
                                    "concept_instances": all_concept_instances,
                                }
                                print(
                                    f"[batch] SAM3 多概念两阶段完成：区域={container}，"
                                    f"目标={concepts}，共 {len(all_concept_instances)} 个实例 "
                                    f"({time.time()-t0:.2f}s)"
                                )
                    else:
                        # Compatibility path: SAM3-only full-image prompting
                        # when the user clears the container field.
                        print(f"[batch] ⚠️ 未设置限制区域，将在全图上进行检测")
                        print(f"[batch] 提示：如需限制识别区域，请在「限制区域」输入框填写容器名称（如：锅、盘子）")
                        all_concept_instances = []
                        for concept_idx, (concept, model_concept) in enumerate(zip(concepts, model_concepts)):
                            sam3_state = sam3_processor.set_image(Image.fromarray(rgb))
                            sam3_output = sam3_state
                            if concept:
                                sam3_output = sam3_processor.set_text_prompt(
                                    prompt=model_concept, state=sam3_state
                                )
                            if first_target_points and concept_idx == 0:
                                for point_idx, point in enumerate(first_target_points):
                                    point_label = bool(
                                        target_prompt_labels_data[point_idx]
                                        if point_idx < len(target_prompt_labels_data)
                                        else True
                                    )
                                    sam3_output = sam3_processor.add_point_prompt(
                                        [float(point[0]) / W, float(point[1]) / H],
                                        point_label, sam3_state
                                    )
                            elif first_box and len(first_box) == 4 and concept_idx == 0:
                                x1, y1, x2, y2 = [float(v) for v in first_box]
                                box_prompt = [
                                    ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H,
                                    max(1.0, x2 - x1) / W, max(1.0, y2 - y1) / H,
                                ]
                                sam3_output = sam3_processor.add_geometric_prompt(
                                    box_prompt, True, sam3_state
                                )
                            sam3_masks = _sam3_masks_from_output(sam3_output)
                            if not sam3_masks and first_target_points:
                                sam3_processor.set_confidence_threshold(0.15)
                                sam3_output = sam3_processor.set_confidence_threshold(
                                    0.15, state=sam3_state
                                )
                                sam3_masks = _sam3_masks_from_output(sam3_output)
                            
                            for mask in sam3_masks:
                                all_concept_instances.append({
                                    "concept": concept,
                                    "mask": mask,
                                })
                        
                        if not all_concept_instances:
                            result = None
                        else:
                            all_masks = [inst["mask"] for inst in all_concept_instances]
                            result = {
                                "result_mask": np.any(all_masks, axis=0),
                                "source": "sam3_prompted_multi",
                                "instances": np.asarray(all_masks),
                                "concept_instances": all_concept_instances,
                            }
                        print(f"[batch] SAM3 set_image+prompt 完成 ({time.time()-t0:.2f}s)")
                log_buf.write(image_log.getvalue())

                if result is None or result.get("result_mask") is None:
                    print(f"[batch] 识别失败: {name}")
                    failure_hint = (
                        f"SAM3 未在「{container}」区域内识别到「{concepts}」"
                        if container and concepts
                        else (
                            f"SAM3 未识别到限制区域「{container}」"
                            if container else "SAM3 未返回有效 mask"
                        )
                    )
                    results_summary.append({"file": name, "status": "error",
                                             "msg": failure_hint})
                    continue

                mask = ensure_bool_mask(result["result_mask"])
                color = COLORS[idx % len(COLORS)]

                # SAM3 may return multiple concept instances. Keep each mask
                # separately in the archive while retaining a combined preview.
                concept_instances = result.get("concept_instances", [])
                sam3_instances = result.get("instances")
                if sam3_instances is not None:
                    instance_dir = os.path.join(img_dir, "instances")
                    os.makedirs(instance_dir, exist_ok=True)
                    for inst_idx, inst_mask in enumerate(sam3_instances, start=1):
                        inst_bytes = mask_to_binary_png_bytes(ensure_bool_mask(inst_mask))
                        if inst_bytes:
                            # 获取这个实例的概念名称
                            concept_name_for_inst = None
                            if inst_idx - 1 < len(concept_instances):
                                concept_name_for_inst = concept_instances[inst_idx - 1]["concept"]
                            filename = f"instance_{inst_idx:03d}"
                            if concept_name_for_inst:
                                filename += f"_{concept_name_for_inst}"
                            filename += ".png"
                            with open(os.path.join(instance_dir, filename), "wb") as fp:
                                fp.write(inst_bytes)

                # 保存 mask / overlay / contour
                b = mask_to_binary_png_bytes(mask)
                if b:
                    with open(os.path.join(img_dir, "mask_binary.png"), "wb") as fp:
                        fp.write(b)

                b = mask_to_color_png_bytes(mask, color)
                if b:
                    with open(os.path.join(img_dir, "mask_color.png"), "wb") as fp:
                        fp.write(b)

                # 生成多概念彩色 overlay
                ov = bgr.copy()
                if concept_instances:
                    # 为每个概念使用不同颜色
                    for inst_idx, inst_data in enumerate(concept_instances):
                        inst_mask = inst_data["mask"]
                        concept_name = inst_data["concept"]
                        inst_color = np.array(COLORS[inst_idx % len(COLORS)], dtype=np.uint8)
                        
                        # 叠加颜色
                        ov[inst_mask] = (
                            ov[inst_mask].astype(np.float32) * 0.38
                            + inst_color.astype(np.float32) * 0.62
                        ).astype(np.uint8)
                        
                        # 添加轮廓线
                        inst_mask_u8 = inst_mask.astype(np.uint8) * 255
                        inst_contours, _ = cv2.findContours(inst_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(ov, inst_contours, -1, tuple(int(v) for v in inst_color), 2)
                        
                        # 添加文本标签（使用 PIL 绘制中文）
                        if inst_contours:
                            x, y, w, h = cv2.boundingRect(max(inst_contours, key=cv2.contourArea))
                            ov = _draw_chinese_text(ov, concept_name, (x, max(25, y - 25)), inst_color, font_size=20)
                else:
                    # 单概念模式（兼容旧逻辑）
                    bgr_color = np.array([color[2], color[1], color[0]], dtype=np.float32)
                    ov[mask] = ov[mask].astype(np.float32) * (1 - MASK_ALPHA) + bgr_color * MASK_ALPHA
                    ov = np.clip(ov, 0, 255).astype(np.uint8)
                
                cv2.imwrite(os.path.join(img_dir, "overlay.png"), ov)
                preview_b64 = image_to_jpeg_b64(ov)

                # contour
                mask_u8 = mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)
                cimg = bgr.copy()
                th = max(2, min(H, W) // 400)
                cv2.drawContours(cimg, contours, -1, (0, 0, 0), th + 4)
                cv2.drawContours(cimg, contours, -1, (int(color[2]), int(color[1]),
                                                       int(color[0])), th)
                cv2.imwrite(os.path.join(img_dir, "contour.png"), cimg)

                # JSON
                contour_list = []
                for c in contours:
                    pts = c.reshape(-1, 2).tolist()
                    if len(pts) >= 3:
                        contour_list.append(pts)
                ys, xs = np.where(mask)
                bbox = [int(xs.min()), int(ys.min()),
                        int(xs.max()), int(ys.max())] if len(xs) else None

                # 构建概念实例列表
                concept_instances_data = []
                for inst_idx, inst_data in enumerate(concept_instances, start=1):
                    inst_mask = inst_data["mask"]
                    ys_inst, xs_inst = np.where(inst_mask)
                    bbox_inst = [int(xs_inst.min()), int(ys_inst.min()),
                                 int(xs_inst.max()), int(ys_inst.max())] if len(xs_inst) else None
                    concept_instances_data.append({
                        "instance_id": inst_idx,
                        "concept": inst_data["concept"],
                        "area": int(inst_mask.sum()),
                        "bbox_xyxy": bbox_inst,
                        "mask_base64_png": mask_to_png_b64(inst_mask),
                    })

                per_img_data = {
                    "file":       name,
                    "image_size": {"height": H, "width": W},
                    "mode":       mode,
                    "source":     result.get("source", mode),
                    "concepts":   concepts or None,
                    "container_concept": container or None,
                    "instance_count": len(concept_instances),
                    "concept_instances": concept_instances_data,
                    "preview": preview_b64,
                    "area":       int(mask.sum()),
                    "bbox_xyxy":  bbox,
                    "contours":   contour_list,
                    "mask_base64_png": mask_to_png_b64(mask),
                }
                with open(os.path.join(img_dir, "annotations.json"), "w",
                          encoding="utf-8") as fp:
                    json.dump(per_img_data, fp, indent=2, ensure_ascii=False)

                results_summary.append({
                    "file":    name,
                    "status":  "ok",
                    "area":    int(mask.sum()),
                    "source":  result.get("source", mode),
                    "concepts": concepts or None,
                    "container_concept": container or None,
                    "instance_count": len(concept_instances),
                    "preview": preview_b64,
                    "batch_index": idx,
                })
                print(f"[batch] ✅ 完成: {name}, 面积={int(mask.sum())}, 实例数={len(concept_instances)}")

            except Exception as e:
                print(f"[batch] 异常: {name} -> {type(e).__name__}: {e}")
                traceback.print_exc(file=log_buf)
                results_summary.append({"file": name, "status": "error",
                                         "msg": f"{type(e).__name__}: {e}"})

        # ---- 打包 zip ----
        zip_path = os.path.join(OUTPUT_ROOT, f"batch_{ts}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, fnames in os.walk(batch_dir):
                for fn in fnames:
                    fp = os.path.join(root, fn)
                    rel = os.path.relpath(fp, batch_dir)
                    zf.write(fp, rel)

            # 顶层 summary
            summary = {
                "batch_time":    ts,
                "mode":          mode,
                "total_files":   len(files),
                "success_count": sum(1 for r in results_summary if r["status"] == "ok"),
                "results":       results_summary,
            }
            zf.writestr("summary.json",
                        json.dumps(summary, indent=2, ensure_ascii=False))

        print(f"\n[batch] 打包完成: {zip_path}")

    log_text = log_buf.getvalue()
    STATE["last_log"] = log_text

    if not os.path.exists(zip_path):
        return {"status": "error", "msg": "打包失败", "log": log_text}

    return {
        "status":      "ok",
        "total":       len(files),
        "success":     sum(1 for r in results_summary if r["status"] == "ok"),
        "results":     results_summary,
        "batch_id":    f"batch_{ts}",
        "log":         log_text,
        "zip_url":     f"/batch_download/{os.path.basename(zip_path)}",
        "zip_base64":  None,
        "zip_name":    f"batch_{ts}.zip",
    }


@app.get("/batch_result/{batch_id}/{index}")
def load_batch_result(batch_id: str, index: int):
    """Load one completed batch image and its masks into the workbench."""
    if os.path.basename(batch_id) != batch_id or not batch_id.startswith("batch_"):
        return {"status": "error", "msg": "无效的批量任务"}
    batch_dir = os.path.join(OUTPUT_ROOT, batch_id)
    if not os.path.isdir(batch_dir) or index < 0:
        return {"status": "error", "msg": "批量任务不存在"}

    folders = sorted(
        name for name in os.listdir(batch_dir)
        if os.path.isdir(os.path.join(batch_dir, name))
    )
    if index >= len(folders):
        return {"status": "error", "msg": "批量图片索引不存在"}
    image_dir = os.path.join(batch_dir, folders[index])
    original_path = os.path.join(image_dir, "original.png")
    annotation_path = os.path.join(image_dir, "annotations.json")
    if not os.path.isfile(original_path) or not os.path.isfile(annotation_path):
        return {"status": "error", "msg": "该图片没有有效的批量实例结果"}

    bgr = cv2.imread(original_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return {"status": "error", "msg": "无法读取批量结果原图"}
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with open(annotation_path, "r", encoding="utf-8") as fp:
        annotation = json.load(fp)

    instances = {}
    workbench_path = os.path.join(image_dir, "workbench.json")
    if os.path.isfile(workbench_path):
        try:
            with open(workbench_path, "r", encoding="utf-8") as fp:
                saved = json.load(fp)
            
            # 构建 concept_instances 映射（如果有）
            # SAM2 批量跟踪保存为 "instances"，普通批量识别保存为 "concept_instances"
            concept_instances = annotation.get("concept_instances") or annotation.get("instances", [])
            concept_map = {}
            for idx, ci in enumerate(concept_instances, start=1):
                # SAM2 批量跟踪使用 "name" 字段，普通批量识别使用 "concept" 字段
                concept_map[idx] = ci.get("name") or ci.get("concept", "batch object")
            
            for enum_idx, item in enumerate(saved.get("instances", []), start=1):
                mask_name = item.get("mask_file")
                if not mask_name:
                    continue
                mask_path = os.path.join(image_dir, "instances", os.path.basename(mask_name))
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                iid = int(item.get("id", len(instances) + 1))
                if mask.shape[:2] != rgb.shape[:2]:
                    mask = cv2.resize(
                        mask, (rgb.shape[1], rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                
                # 从 concept_instances 获取类别名称（优先使用枚举索引查找）
                # 如果名称是默认的 "batch object" 或 "object"，也尝试从 concept_map 获取
                saved_name = item.get("name")
                if saved_name and saved_name not in ("batch object", "object"):
                    instance_name = saved_name
                else:
                    instance_name = concept_map.get(enum_idx) or concept_map.get(iid) or annotation.get("concept") or saved_name or "batch object"
                
                instances[iid] = {
                    "points": item.get("points", []),
                    "labels": item.get("labels", []),
                    "box": item.get("box"),
                    "mask": np.squeeze(mask) > 0,
                    "amg": bool(item.get("amg", False)),
                    "source": item.get("source", "sam3_batch"),
                    "name": instance_name,
                    "confidence": item.get("confidence"),
                    "task": item.get("task", "segmentation"),
                }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            instances = {}

    if not instances:
        instance_files = sorted(
            name for name in os.listdir(os.path.join(image_dir, "instances"))
            if name.lower().endswith(".png")
        ) if os.path.isdir(os.path.join(image_dir, "instances")) else []
        
        # 构建 concept_instances 映射（如果有）
        # SAM2 批量跟踪保存为 "instances"，普通批量识别保存为 "concept_instances"
        concept_instances = annotation.get("concept_instances") or annotation.get("instances", [])
        concept_map = {}
        for idx, ci in enumerate(concept_instances, start=1):
            # SAM2 批量跟踪使用 "name" 字段，普通批量识别使用 "concept" 字段
            concept_map[idx] = ci.get("name") or ci.get("concept", "batch object")
        
        for instance_index, mask_name in enumerate(instance_files, start=1):
            mask = cv2.imread(
                os.path.join(image_dir, "instances", mask_name),
                cv2.IMREAD_GRAYSCALE,
            )
            if mask is None:
                continue
            mask = np.squeeze(mask)
            if mask.ndim != 2:
                continue
            if mask.shape != rgb.shape[:2]:
                mask = cv2.resize(
                    mask, (rgb.shape[1], rgb.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            
            # 从 concept_instances 获取类别名称
            instance_name = concept_map.get(instance_index) or annotation.get("concept") or "batch object"
            
            instances[instance_index] = {
                "points": [], "labels": [], "box": None,
                "mask": mask > 0, "amg": False,
                "source": "sam3_batch", "name": instance_name,
                "confidence": None, "task": "segmentation",
            }

    STATE.update({
        "image_bgr": bgr, "image_rgb": rgb, "h": rgb.shape[0], "w": rgb.shape[1],
        "instances": instances, "next_id": (max(instances) + 1) if instances else 1,
        "current_id": next(iter(instances), None), "predictor_ready": False,
        "output_dir": image_dir, "last_log": "",
    })
    rendered = render_composite_to_b64(highlight_id=None)
    return {
        "status": "ok",
        "image_h": int(rgb.shape[0]),
        "image_w": int(rgb.shape[1]),
        "count": len(instances),
        "instances": [
            {
                "id": int(iid), "name": inst["name"],
                "source": inst["source"], "task": inst["task"],
                "area": int(inst["mask"].sum()),
                "color": [int(c) for c in get_color(iid)],
            }
            for iid, inst in instances.items()
        ],
        "rendered": rendered,
    }


def _persist_current_batch_state():
    """Persist edits made in the workbench for the currently loaded batch image."""
    image_dir = STATE.get("output_dir")
    if not image_dir or os.path.basename(os.path.dirname(image_dir)).startswith("sam2_batch_tracking"):
        return
    batch_parent = os.path.basename(os.path.dirname(image_dir))
    if not batch_parent.startswith("batch_") or not os.path.isdir(image_dir):
        return

    instances_dir = os.path.join(image_dir, "instances")
    os.makedirs(instances_dir, exist_ok=True)
    for name in os.listdir(instances_dir):
        if name.lower().endswith(".png"):
            try:
                os.remove(os.path.join(instances_dir, name))
            except OSError:
                pass

    saved_instances = []
    for iid, inst in sorted(STATE["instances"].items()):
        mask = ensure_bool_mask(inst.get("mask"))
        if mask is None:
            continue
        mask_name = f"instance_{int(iid):03d}.png"
        mask_bytes = mask_to_binary_png_bytes(mask)
        if mask_bytes:
            with open(os.path.join(instances_dir, mask_name), "wb") as fp:
                fp.write(mask_bytes)
        saved_instances.append({
            "id": int(iid),
            "mask_file": mask_name,
            "points": inst.get("points", []),
            "labels": inst.get("labels", []),
            "box": inst.get("box"),
            "amg": bool(inst.get("amg", False)),
            "source": inst.get("source", "sam3_batch"),
            "name": inst.get("name", "batch object"),
            "confidence": inst.get("confidence"),
            "task": inst.get("task", "segmentation"),
        })

    with open(os.path.join(image_dir, "workbench.json"), "w", encoding="utf-8") as fp:
        json.dump({
            "image_size": {"height": STATE["h"], "width": STATE["w"]},
            "instances": saved_instances,
        }, fp, indent=2, ensure_ascii=False)


@app.get("/batch_download/{filename}")
def batch_download(filename: str):
    safe_name = os.path.basename(filename)
    if safe_name != filename or not safe_name.startswith("batch_") or not safe_name.endswith(".zip"):
        return {"status": "error", "msg": "无效的批量结果文件"}
    path = os.path.join(OUTPUT_ROOT, safe_name)
    if not os.path.isfile(path):
        return {"status": "error", "msg": "批量结果文件不存在"}
    return FileResponse(path, media_type="application/zip", filename=safe_name)


@app.post("/batch_export_zip/{batch_id}")
def batch_export_zip(batch_id: str):
    """Export every image and its latest edited workbench instances."""
    if os.path.basename(batch_id) != batch_id or not batch_id.startswith("batch_"):
        return {"status": "error", "msg": "无效的批量任务"}
    batch_dir = os.path.join(OUTPUT_ROOT, batch_id)
    if not os.path.isdir(batch_dir):
        return {"status": "error", "msg": "批量任务不存在"}

    folders = sorted(
        name for name in os.listdir(batch_dir)
        if os.path.isdir(os.path.join(batch_dir, name))
    )
    if not folders:
        return {"status": "error", "msg": "批量任务没有图片结果"}

    zip_name = f"{batch_id}_edited.zip"
    zip_path = os.path.join(OUTPUT_ROOT, zip_name)
    summary = {
        "batch_id": batch_id,
        "total_images": len(folders),
        "images_with_instances": 0,
        "total_instances": 0,
        "source": "latest_workbench_state",
    }
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in folders:
            image_dir = os.path.join(batch_dir, folder)
            workbench_path = os.path.join(image_dir, "workbench.json")
            instance_count = 0
            if os.path.isfile(workbench_path):
                try:
                    with open(workbench_path, "r", encoding="utf-8") as fp:
                        instance_count = len(json.load(fp).get("instances", []))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    instance_count = 0
            else:
                instances_dir = os.path.join(image_dir, "instances")
                if os.path.isdir(instances_dir):
                    instance_count = sum(
                        1 for name in os.listdir(instances_dir)
                        if name.lower().endswith(".png")
                    )
            if instance_count:
                summary["images_with_instances"] += 1
                summary["total_instances"] += instance_count

            for root, _dirs, files in os.walk(image_dir):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    rel = os.path.relpath(filepath, batch_dir)
                    zf.write(filepath, rel)

        zf.writestr(
            "edited_summary.json",
            json.dumps(summary, indent=2, ensure_ascii=False),
        )

    return {
        "status": "ok",
        "batch_id": batch_id,
        "total": summary["total_images"],
        "success": summary["images_with_instances"],
        "instances": summary["total_instances"],
        "zip_url": f"/batch_download/{zip_name}",
        "zip_name": zip_name,
    }


# ============================================================
# GET /last_log
# ============================================================
@app.get("/last_log")
def last_log():
    return {"log": STATE.get("last_log", "(暂无日志)")}


# ============================================================
# GET /image
# ============================================================
@app.get("/image")
def get_image():
    if STATE["image_bgr"] is None:
        return {"image": None}
    ok, buf = cv2.imencode(".jpg", STATE["image_bgr"], [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return {"image": None}
    return {"image": base64.b64encode(buf.tobytes()).decode("ascii")}


# ============================================================
# GET /render
# ============================================================
@app.get("/render")
def render_view(highlight: Optional[str] = Query(default=None)):
    hid = None
    if highlight not in (None, "", "null", "undefined", "None"):
        try:
            hid = int(highlight)
        except (ValueError, TypeError):
            hid = None
    try:
        b64 = render_composite_to_b64(highlight_id=hid)
    except Exception as e:
        print("[render] EXCEPTION:")
        traceback.print_exc()
        return {"image": None, "highlight": hid, "error": f"{type(e).__name__}: {e}"}
    return {"image": b64, "highlight": hid}


# ============================================================
# GET /masks
# ============================================================
@app.get("/masks")
def get_masks():
    out = []
    for iid, inst in STATE["instances"].items():
        if inst.get("mask") is None:
            continue
        out.append({
            "id":     int(iid),
            "color":  [int(c) for c in get_color(iid)],
            "amg":    bool(inst["amg"]),
            "source": inst.get("source", "amg" if inst["amg"] else "prompt"),
            "name":   inst.get("name", "object"),
            "confidence": inst.get("confidence"),
            "task":   inst.get("task", "segmentation"),
            "area":   int(ensure_bool_mask(inst["mask"]).sum()),
        })
    out.sort(key=lambda x: x["id"])
    return {"masks": out, "count": len(out)}


# ============================================================
# POST /hit_test
# ============================================================
@app.post("/hit_test")
def hit_test(req: PointRequest):
    if STATE["image_rgb"] is None:
        return {"status": "error", "msg": "请先上传图片"}
    x, y = int(round(req.x)), int(round(req.y))
    h, w = STATE["h"], STATE["w"]
    if not (0 <= x < w and 0 <= y < h):
        return {"status": "ok", "hit": None, "candidates": []}
    hits = []
    for iid, inst in STATE["instances"].items():
        if inst.get("mask") is None:
            continue
        m = ensure_bool_mask(inst["mask"])
        if m[y, x]:
            hits.append(int(iid))
    if not hits:
        return {"status": "ok", "hit": None, "candidates": []}
    return {"status": "ok", "hit": max(hits), "candidates": sorted(hits)}


# ============================================================
def update_mask(iid: int):
    inst = STATE["instances"][iid]
    if STATE["image_rgb"] is None:
        return
    pts = labs = box = None
    if inst["points"]:
        pts  = np.array(inst["points"], dtype=np.float32)
        labs = np.array(inst["labels"], dtype=np.int32)
    if inst["box"] is not None:
        box = np.array(inst["box"], dtype=np.float32)
    if pts is None and box is None:
        return
    inst["mask"] = run_predict(pts, labs, box)


# ============================================================
# POST /predict_prompt
# ============================================================
@app.post("/predict_prompt")
def predict_prompt(req: PromptRequest):
    if STATE["image_rgb"] is None:
        return {"status": "error", "msg": "请先上传图片"}
    pts = labs = box = None
    if req.points:
        pts  = np.array([[p.x, p.y] for p in req.points], dtype=np.float32)
        labs = np.array([p.label for p in req.points], dtype=np.int32)
    if req.box and len(req.box) == 4:
        box = np.array(req.box, dtype=np.float32)
    if pts is None and box is None:
        return {"status": "error", "msg": "缺少提示"}

    mask = run_predict(pts, labs, box)
    iid = STATE["next_id"]; STATE["next_id"] += 1
    STATE["instances"][iid] = {
        "points": [[p.x, p.y] for p in (req.points or [])],
        "labels": [p.label for p in (req.points or [])],
        "box":    req.box, "mask": mask,
        "amg": False, "source": "prompt", "name": req.name or "object", "confidence": None,
    }
    STATE["current_id"] = iid

    _persist_current_batch_state()
    return {
        "status":   "ok",
        "new_id":   iid, "id": iid,
        "color":    [int(c) for c in get_color(iid)],
        "amg":      False,
        "source":   "prompt",
        "area":     int(mask.sum()),
        "name":     req.name or "object", "confidence": None,
        "rendered": render_composite_to_b64(highlight_id=None, highlight_new=iid),
    }


# ============================================================
# POST /detect_text
# YOLO-World: 文本类别 -> 检测框 -> SAM2 精细 mask
# ============================================================
@app.post("/detect_text")
def detect_text(req: TextDetectRequest):
    global world_model
    if STATE["image_rgb"] is None:
        return {"status": "error", "msg": "请先上传图片"}
    if YOLO is None:
        return {"status": "error", "msg": "缺少 ultralytics，请在 sam2 环境执行 pip install ultralytics"}

    names = [str(x).strip() for x in req.names if str(x).strip()]
    if not names:
        return {"status": "error", "msg": "至少输入一个类别名称"}
    model_name = str(req.model or "yolov8s-world.pt").strip()
    # Ultralytics YOLO-World weights are published as yolov8*-world.pt.
    # Keep accepting the old UI aliases so existing clients do not break.
    model_aliases = {
        "yolo11s-world.pt": "yolov8s-world.pt",
        "yolo11m-world.pt": "yolov8m-world.pt",
    }
    model_name = model_aliases.get(model_name, model_name)
    if model_name not in {
        "yolov8s-world.pt", "yolov8m-world.pt",
        "yolov8l-world.pt", "yolov8x-world.pt",
    }:
        return {"status": "error", "msg": f"不支持的 YOLO-World 模型: {model_name}"}
    if world_model is None or getattr(world_model, "model_name", None) != model_name:
        print(f"[YOLO-World] loading {model_name} ...")
        with MODEL_STATUS_LOCK:
            MODEL_STATUS.update({
                "phase": "loading",
                "model": model_name,
                "message": "正在检查或下载模型及文本编码器",
        })
        try:
            local_weight = os.path.join(SCRIPT_DIR, model_name)
            model_source = local_weight if os.path.isfile(local_weight) else model_name
            try:
                world_model = YOLO(model_source)
            except Exception as first_error:
                # Interrupted downloads leave a .pt file that exists but is
                # not a valid torch archive. Remove only that known model
                # file and let Ultralytics download a clean copy once.
                weight_path = os.path.join(SCRIPT_DIR, model_name)
                error_text = str(first_error)
                if (
                    os.path.isfile(weight_path)
                    and (
                        "failed finding central directory" in error_text
                        or "PytorchStreamReader" in error_text
                    )
                ):
                    print(f"[YOLO-World] removing corrupted checkpoint {weight_path}")
                    os.remove(weight_path)
                    world_model = YOLO(model_source)
                else:
                    raise
            world_model.model_name = model_name
            with MODEL_STATUS_LOCK:
                MODEL_STATUS.update({
                    "phase": "ready",
                    "model": model_name,
                    "message": "模型已加载，可直接识别",
                })
        except Exception as e:
            traceback.print_exc()
            with MODEL_STATUS_LOCK:
                MODEL_STATUS.update({
                    "phase": "error",
                    "model": model_name,
                    "message": str(e),
                })
            return {
                "status": "error",
                "msg": (
                    f"YOLO-World 模型加载失败: {e}\n"
                    "首次使用需要联网下载模型，请稍后重试。"
                ),
            }
    # Some Ultralytics/CLIP combinations tokenize on CPU while the cached
    # text model is on CUDA. Build the text features on CPU, then move only
    # the resulting embeddings back to the YOLO model device.
    model_device = next(world_model.model.parameters()).device
    try:
        if model_device.type == "cuda":
            world_model.model.to("cpu")
            world_model.set_classes(names)
            world_model.model.to(model_device)
            if getattr(world_model.model, "txt_feats", None) is not None:
                world_model.model.txt_feats = world_model.model.txt_feats.to(model_device)
        else:
            world_model.set_classes(names)
    except Exception as e:
        traceback.print_exc()
        with MODEL_STATUS_LOCK:
            MODEL_STATUS.update({
                "phase": "error",
                "model": model_name,
                "message": str(e),
            })
        return {
            "status": "error",
            "msg": (
                f"类别名称编码失败: {e}\n"
                "请尝试使用英文物体名称，或重新启动后端后再试。"
            ),
        }

    try:
        result = world_model.predict(
            source=STATE["image_rgb"],
            conf=max(0.01, min(float(req.confidence), 0.99)),
            iou=max(0.01, min(float(req.iou), 0.99)),
            device=0 if device == "cuda" else "cpu",
            verbose=False,
        )[0]
    except Exception as e:
        traceback.print_exc()
        with MODEL_STATUS_LOCK:
            MODEL_STATUS.update({
                "phase": "error",
                "model": model_name,
                "message": str(e),
            })
        return {"status": "error", "msg": f"目标检测失败: {e}"}
    boxes = result.boxes
    created = []
    for box_tensor, conf_tensor, cls_tensor in zip(boxes.xyxy, boxes.conf, boxes.cls):
        box = [float(v) for v in box_tensor.detach().cpu().tolist()]
        confidence = float(conf_tensor.detach().cpu().item())
        class_index = int(cls_tensor.detach().cpu().item())
        name = names[class_index] if class_index < len(names) else str(class_index)
        task = req.task if req.task in ("detection", "segmentation") else "segmentation"
        if task == "detection":
            mask = np.zeros((STATE["h"], STATE["w"]), dtype=bool)
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            mask[max(0, y1):min(STATE["h"], y2), max(0, x1):min(STATE["w"], x2)] = True
        else:
            mask = run_predict(box=np.array(box, dtype=np.float32))
        iid = STATE["next_id"]
        STATE["next_id"] += 1
        STATE["instances"][iid] = {
            "points": [], "labels": [], "box": box, "mask": mask,
            "amg": False, "source": "yolo_world", "name": name,
            "confidence": confidence, "task": task,
        }
        created.append({
            "id": iid, "name": name, "confidence": confidence,
            "color": [int(c) for c in get_color(iid)], "source": "yolo_world",
            "task": task,
            "area": int(mask.sum()),
        })
    if created:
        STATE["current_id"] = created[-1]["id"]
    return {
        "status": "ok", "count": len(created), "instances": created,
        "rendered": render_composite_to_b64(highlight_id=None),
    }


@app.post("/sam3_image_prompt")
@serialized_inference
async def sam3_image_prompt(req: Sam3ImagePromptRequest):
    """主页右侧的 SAM3 图像概念提示，不经过 YOLO-World。"""
    if STATE["image_rgb"] is None:
        return {"status": "error", "msg": "请先上传图片"}
    name = req.name.strip()
    if not name:
        return {"status": "error", "msg": "请输入要识别的物体名称"}
    try:
        global SAM3_IMAGE_PROCESSOR
        _, processor = _sam3_load_image_model()
        processor.set_confidence_threshold(max(0.05, min(float(req.confidence), 0.99)))
        from PIL import Image
        state = processor.set_image(Image.fromarray(STATE["image_rgb"]))
        # Keep the Chinese label for the UI, but use the stable English alias
        # for SAM3 grounding (e.g. 藕片 -> lotus root slices).
        output = processor.set_text_prompt(prompt=_batch_model_prompt(name), state=state)
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None or len(masks) == 0:
            return {"status": "ok", "count": 0, "msg": f"SAM3 未找到：{name}", "rendered": render_composite_to_b64()}
        created = []
        masks_np = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
        boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
        scores_np = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
        for idx, mask in enumerate(masks_np):
            mask = ensure_bool_mask(mask)
            if mask.ndim != 2:
                mask = np.squeeze(mask)
            if mask.ndim != 2:
                return {"status": "error", "msg": f"SAM3 返回了不支持的 mask 形状: {mask.shape}"}
            iid = STATE["next_id"]
            STATE["next_id"] += 1
            box = [float(v) for v in boxes_np[idx].tolist()] if idx < len(boxes_np) else None
            confidence = float(scores_np[idx]) if idx < len(scores_np) else None
            STATE["instances"][iid] = {
                "points": [], "labels": [], "box": box, "mask": mask,
                "amg": False, "source": "sam3_text", "name": name,
                "confidence": confidence, "task": "segmentation",
            }
            created.append({"id": iid, "name": name, "confidence": confidence,
                            "color": [int(c) for c in get_color(iid)],
                            "source": "sam3_text", "task": "segmentation", "area": int(mask.sum())})
        if created:
            STATE["current_id"] = created[-1]["id"]
        return {"status": "ok", "count": len(created), "instances": created,
                "rendered": render_composite_to_b64(highlight_id=None)}
    except Exception as exc:
        traceback.print_exc()
        return {"status": "error", "msg": str(exc)}


@app.get("/model_status")
def model_status(model: Optional[str] = Query(default=None)):
    requested = str(model or "yolov8s-world.pt").strip()
    with MODEL_STATUS_LOCK:
        status = dict(MODEL_STATUS)
    if status["model"] == requested and status["phase"] in {"loading", "ready", "error"}:
        return status
    weight_path = os.path.join(SCRIPT_DIR, requested)
    if (
        os.path.isfile(weight_path)
        and os.path.getsize(weight_path) > 0
        and zipfile.is_zipfile(weight_path)
    ):
        return {
            "phase": "cached",
            "model": requested,
            "message": "模型文件已缓存，首次执行时会加载到内存",
        }
    if os.path.isfile(weight_path) and os.path.getsize(weight_path) > 0:
        return {
            "phase": "error",
            "model": requested,
            "message": "模型文件损坏，将在下次使用时自动重新下载",
        }
    return {
        "phase": "idle",
        "model": requested,
        "message": "模型尚未下载或缓存文件不完整",
    }


# ============================================================
# POST /add_point
# ============================================================
@app.post("/add_point")
def add_point(req: PointRequest):
    if STATE["image_rgb"] is None:
        return {"status": "error", "msg": "请先上传图片"}
    iid = STATE["current_id"]
    if iid is None or iid not in STATE["instances"]:
        return {"status": "error", "msg": "没有当前实例"}
    inst = STATE["instances"][iid]
    inst["points"].append([req.x, req.y])
    inst["labels"].append(req.label)
    update_mask(iid)
    _persist_current_batch_state()
    return {
        "status":   "ok", "id": iid,
        "rendered": render_composite_to_b64(highlight_id=None, highlight_new=iid),
    }


# ============================================================
# POST /add_box
# ============================================================
@app.post("/add_box")
def add_box(req: BoxRequest):
    if STATE["image_rgb"] is None:
        return {"status": "error", "msg": "请先上传图片"}
    iid = STATE["current_id"]
    if iid is None or iid not in STATE["instances"]:
        return {"status": "error", "msg": "没有当前实例"}
    inst = STATE["instances"][iid]
    inst["box"] = [req.x1, req.y1, req.x2, req.y2]
    update_mask(iid)
    _persist_current_batch_state()
    return {
        "status":   "ok", "id": iid,
        "rendered": render_composite_to_b64(highlight_id=None, highlight_new=iid),
    }


@app.post("/update_instance")
def update_instance(req: UpdateInstanceReq):
    if req.id not in STATE["instances"]:
        return {"status": "error", "msg": "实例不存在"}
    inst = STATE["instances"][req.id]
    if req.name is not None:
        name = str(req.name).strip()
        if not name:
            return {"status": "error", "msg": "类别名称不能为空"}
        inst["name"] = name
    STATE["current_id"] = req.id
    _persist_current_batch_state()
    return {
        "status": "ok",
        "id": req.id,
        "name": inst.get("name", "object"),
        "rendered": render_composite_to_b64(highlight_id=req.id),
    }


@app.post("/morph_instance")
def morph_instance(req: MorphInstanceReq):
    if STATE["image_rgb"] is None:
        return {"status": "error", "msg": "请先上传图片"}
    if req.id not in STATE["instances"]:
        return {"status": "error", "msg": "实例不存在"}
    inst = STATE["instances"][req.id]
    if inst.get("mask") is None:
        return {"status": "error", "msg": "实例没有可调整的 mask"}

    if req.reset:
        base = inst.pop("morph_base", None)
        if base is not None:
            inst["mask"] = ensure_bool_mask(base)
        _persist_current_batch_state()
        return {
            "status": "ok", "id": req.id,
            "area": int(ensure_bool_mask(inst["mask"]).sum()),
            "rendered": render_composite_to_b64(highlight_id=req.id),
        }

    if "morph_base" not in inst:
        inst["morph_base"] = ensure_bool_mask(inst["mask"]).copy()
    base = ensure_bool_mask(inst["morph_base"])
    amount = max(-10, min(10, int(req.amount)))
    mask_u8 = (base.astype(np.uint8) * 255)
    if amount > 0:
        kernel = np.ones((3, 3), np.uint8)
        adjusted = cv2.dilate(mask_u8, kernel, iterations=amount)
    elif amount < 0:
        kernel = np.ones((3, 3), np.uint8)
        adjusted = cv2.erode(mask_u8, kernel, iterations=abs(amount))
    else:
        adjusted = mask_u8
    inst["mask"] = adjusted > 0
    if req.commit:
        inst.pop("morph_base", None)
    _persist_current_batch_state()
    return {
        "status": "ok", "id": req.id,
        "area": int(ensure_bool_mask(inst["mask"]).sum()),
        "rendered": render_composite_to_b64(highlight_id=req.id),
    }


# ============================================================
@app.post("/select_instance")
def select_instance(req: SelectInstanceReq):
    if req.id in STATE["instances"]:
        STATE["current_id"] = req.id
    return {"status": "ok", "current_id": STATE["current_id"]}

@app.post("/new_instance")
def new_instance():
    iid = STATE["next_id"]; STATE["next_id"] += 1
    STATE["instances"][iid] = {
        "points": [], "labels": [], "box": None,
        "mask": None, "amg": False, "source": "prompt",
    }
    STATE["current_id"] = iid
    _persist_current_batch_state()
    return {"status": "ok", "current_id": iid}

@app.post("/delete_instance")
def delete_instance(req: SelectInstanceReq):
    if req.id in STATE["instances"]:
        del STATE["instances"][req.id]
    if STATE["current_id"] == req.id:
        STATE["current_id"] = next(iter(STATE["instances"]), None)
    _persist_current_batch_state()
    return {
        "status":   "ok",
        "rendered": render_composite_to_b64(highlight_id=None),
    }

@app.post("/reset")
def reset():
    STATE.update({
        "image_bgr": None, "image_rgb": None, "h": 0, "w": 0,
        "instances": {}, "next_id": 1, "current_id": None,
        "predictor_ready": False,
        "output_dir": None, "last_log": "",
    })
    return {"status": "ok"}


# ============================================================
# 导出
# ============================================================
def build_instance_export(iid, inst):
    mask_bool = ensure_bool_mask(inst["mask"])
    mask_u8   = mask_bool.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour_list = []
    for c in contours:
        pts = c.reshape(-1, 2).tolist()
        if len(pts) >= 3:
            contour_list.append(pts)

    main_contour = max(contour_list, key=lambda pts: len(pts)) if contour_list else []
    total_pts = sum(len(c) for c in contour_list)

    ys, xs = np.where(mask_bool)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None

    return {
        "instance_id":       int(iid),
        "name":              inst.get("name", "object"),
        "confidence":        inst.get("confidence"),
        "color":             [int(c) for c in get_color(iid)],
        "is_amg":            bool(inst["amg"]),
        "source":            inst.get("source", "amg" if inst["amg"] else "prompt"),
        "prompt_points":     inst["points"],
        "labels":            inst["labels"],
        "box":               inst["box"],
        "area":              int(mask_bool.sum()),
        "bbox_xyxy":         bbox,
        "contours":          contour_list,
        "polygon":           main_contour,
        "mask_points_count": total_pts,
        "mask_base64_png":   mask_to_png_b64(mask_bool),
    }


@app.post("/export")
def export():
    if STATE["image_bgr"] is None:
        return {"status": "error", "msg": "未上传图片"}
    total = len(STATE["instances"])
    with_mask = sum(1 for i in STATE["instances"].values() if i.get("mask") is not None)
    print(f"[export] 实例数={total}, 有 mask={with_mask}")

    if total == 0:
        return {"status": "error", "msg": "后端没有任何实例。"}
    if with_mask == 0:
        return {"status": "error", "msg": "所有实例都还没有 mask。"}

    instances = []
    for iid, inst in STATE["instances"].items():
        if inst.get("mask") is None:
            continue
        instances.append(build_instance_export(iid, inst))
    instances.sort(key=lambda x: x["instance_id"])

    return {
        "status":     "ok",
        "count":      len(instances),
        "image_size": {"height": STATE["h"], "width": STATE["w"]},
        "instances":  instances,
    }


@app.post("/export_one")
def export_one(req: SelectInstanceReq):
    if STATE["image_bgr"] is None:
        return {"status": "error", "msg": "未上传图片"}
    if req.id not in STATE["instances"]:
        return {"status": "error", "msg": f"实例 {req.id} 不存在"}
    inst = STATE["instances"][req.id]
    if inst.get("mask") is None:
        return {"status": "error", "msg": f"实例 {req.id} 还没有 mask"}
    return {
        "status":     "ok",
        "count":      1,
        "image_size": {"height": STATE["h"], "width": STATE["w"]},
        "instances":  [build_instance_export(req.id, inst)],
    }


def collect_class_names(instances_data):
    """按出现顺序收集所有类别名称。"""
    names = []
    for item in instances_data:
        name = item.get("name", "object") or "object"
        if name not in names:
            names.append(name)
    return names


def build_yolo_segmentation(instances_data, class_names=None):
    """生成 YOLO 分割标签：class x1 y1 x2 y2 ... （归一化多边形坐标）。"""
    names = class_names if class_names is not None else collect_class_names(instances_data)
    lines = []
    width, height = STATE["w"], STATE["h"]
    for item in instances_data:
        if item.get("task") == "detection":
            continue
        polygon = item.get("polygon") or []
        if len(polygon) < 3:
            continue
        class_id = names.index(item.get("name", "object") or "object")
        coords = []
        for x, y in polygon:
            coords.extend((max(0.0, min(1.0, float(x) / width)),
                           max(0.0, min(1.0, float(y) / height))))
        lines.append(str(class_id) + " " + " ".join(f"{v:.6f}" for v in coords))
    return names, "\n".join(lines) + ("\n" if lines else "")


def build_yolo_detection(instances_data, class_names=None):
    """生成 YOLO 检测标签：class cx cy w h （归一化中心点与宽高）。"""
    names = class_names if class_names is not None else collect_class_names(instances_data)
    lines = []
    width, height = STATE["w"], STATE["h"]
    for item in instances_data:
        if item.get("task") != "detection":
            continue
        box = item.get("box")
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        cx = max(0.0, min(1.0, ((x1 + x2) / 2.0) / width))
        cy = max(0.0, min(1.0, ((y1 + y2) / 2.0) / height))
        bw = max(0.0, min(1.0, (x2 - x1) / width))
        bh = max(0.0, min(1.0, (y2 - y1) / height))
        if bw <= 0 or bh <= 0:
            continue
        class_id = names.index(item.get("name", "object") or "object")
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return names, "\n".join(lines) + ("\n" if lines else "")


@app.post("/export_zip")
def export_zip():
    if STATE["image_bgr"] is None:
        return {"status": "error", "msg": "未上传图片"}

    total = len(STATE["instances"])
    with_mask = sum(1 for i in STATE["instances"].values() if i.get("mask") is not None)
    if total == 0 or with_mask == 0:
        return {"status": "error", "msg": "没有可导出的实例"}

    print(f"[export_zip] 打包 {with_mask} 个实例")

    buf = io.BytesIO()
    instances_data = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for iid, inst in sorted(STATE["instances"].items()):
            if inst.get("mask") is None:
                continue
            instances_data.append(build_instance_export(iid, inst))
        json_data = {
            "status":     "ok",
            "count":      len(instances_data),
            "image_size": {"height": STATE["h"], "width": STATE["w"]},
            "instances":  instances_data,
        }
        zf.writestr("annotations.json",
                    json.dumps(json_data, indent=2, ensure_ascii=False))

        class_names = collect_class_names(instances_data)
        _, yolo_seg = build_yolo_segmentation(instances_data, class_names)
        _, yolo_det = build_yolo_detection(instances_data, class_names)
        has_det = any(i.get("task") == "detection" for i in instances_data)
        has_seg = any(i.get("task") != "detection" for i in instances_data)
        if has_det and not has_seg:
            zf.writestr("labels/original.txt", yolo_det)
        elif has_seg and not has_det:
            zf.writestr("labels/original.txt", yolo_seg)
        if has_det:
            zf.writestr("labels/detection.txt", yolo_det)
        if has_seg:
            zf.writestr("labels/segmentation.txt", yolo_seg)
        zf.writestr("classes.txt", "\n".join(class_names) + "\n")
        yaml_lines = ["path: .", "train: original.png", "val: original.png", "names:"]
        yaml_lines.extend(f"  {i}: {name}" for i, name in enumerate(class_names))
        zf.writestr("data.yaml", "\n".join(yaml_lines) + "\n")

        ok, b = cv2.imencode(".png", STATE["image_bgr"])
        if ok:
            zf.writestr("original.png", b.tobytes())

        comp_bgr = render_composite_bgr(base_fade=BASE_FADE)
        if comp_bgr is not None:
            ok, b = cv2.imencode(".png", comp_bgr)
            if ok:
                zf.writestr("composite.png", b.tobytes())

        comp_alpha = render_composite_only_alpha()
        if comp_alpha is not None:
            ok, b = cv2.imencode(".png", comp_alpha)
            if ok:
                zf.writestr("composite_alpha.png", b.tobytes())

        for iid, inst in sorted(STATE["instances"].items()):
            if inst.get("mask") is None:
                continue
            mask = ensure_bool_mask(inst["mask"])
            color = get_color(iid)
            pad = f"{iid:04d}"

            b = mask_to_binary_png_bytes(mask)
            if b:
                zf.writestr(f"masks/instance_{pad}_binary.png", b)

            b = mask_to_color_png_bytes(mask, color)
            if b:
                zf.writestr(f"masks/instance_{pad}_color.png", b)

            overlay = STATE["image_bgr"].astype(np.float32)
            bgr_color = np.array([color[2], color[1], color[0]], dtype=np.float32)
            overlay[mask] = overlay[mask] * (1 - MASK_ALPHA) + bgr_color * MASK_ALPHA
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            ok, b = cv2.imencode(".png", overlay)
            if ok:
                zf.writestr(f"overlays/instance_{pad}.png", b.tobytes())

            cimg = build_contour_image_bgr(mask, color)
            ok, b = cv2.imencode(".png", cimg)
            if ok:
                zf.writestr(f"contours/instance_{pad}.png", b.tobytes())

        out_dir = STATE.get("output_dir")
        if out_dir and os.path.isdir(out_dir):
            for fname in sorted(os.listdir(out_dir)):
                if not fname.endswith(".png"):
                    continue
                if fname == "original.png":
                    continue
                fpath = os.path.join(out_dir, fname)
                try:
                    zf.write(fpath, f"process/{fname}")
                except Exception as e:
                    print(f"[export_zip] 跳过 {fname}: {e}")

        readme = (
            "SAM2 分割结果导出\n"
            f"导出于: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"原图尺寸: {STATE['w']} x {STATE['h']}\n"
            f"实例数量: {len(instances_data)}\n\n"
            "文件说明:\n"
            "  annotations.json            所有实例的分割点/轮廓/bbox/mask\n"
            "  labels/original.txt         YOLO Segmentation 标签（归一化多边形）\n"
            "  classes.txt / data.yaml     YOLO 类别配置\n"
            "  original.png                原图\n"
            "  composite.png               原图 + 所有 mask 彩色叠加\n"
            "  composite_alpha.png         只有 mask（透明背景）\n"
            "  masks/instance_XXXX_binary.png   单实例黑白 mask\n"
            "  masks/instance_XXXX_color.png    单实例彩色 mask（透明背景）\n"
            "  overlays/instance_XXXX.png       原图 + 该实例 mask 叠加\n"
            "  contours/instance_XXXX.png       原图 + 该实例轮廓\n"
            "  process/*.png               识别过程调试图\n"
        )
        zf.writestr("README.txt", readme)

    zip_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    print(f"[export_zip] 完成，{len(zip_b64)//1024} KB")
    return {
        "status":     "ok",
        "count":      len(instances_data),
        "zip_base64": zip_b64,
    }


# ============================================================
# GET /debug_roi
# ============================================================
@app.get("/debug_roi")
def debug_roi():
    """在原图上画出 ROI 框，让用户看到 ROI 位置"""
    if STATE["image_bgr"] is None:
        return {"status": "error", "msg": "请先上传图片"}

    cfg = {k: (v.copy() if isinstance(v, list) else v) for k, v in DEFAULT_CONFIG.items()}
    x1, y1, x2, y2 = resolve_roi(STATE["image_rgb"], cfg)
    img = STATE["image_bgr"].copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(img, f"ROI [{x1},{y1},{x2},{y2}]", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return {"image": base64.b64encode(buf.tobytes()).decode("ascii")}


# ============================================================
# GET /config_template
# ============================================================
@app.get("/config_template")
def config_template():
    if not os.path.exists(TEMPLATE_XLSX):
        create_template()
    with open(TEMPLATE_XLSX, "rb") as f:
        content = f.read()
    return {
        "filename": "配置模板.xlsx",
        "base64": base64.b64encode(content).decode("ascii"),
    }


# ============================================================
# GET /debug_state
# ============================================================
@app.get("/debug_state")
def debug_state():
    return {
        "has_image":     STATE["image_bgr"] is not None,
        "image_size":    [STATE["w"], STATE["h"]],
        "num_instances": len(STATE["instances"]),
        "with_mask":     sum(1 for i in STATE["instances"].values()
                             if i.get("mask") is not None),
        "instance_ids":  sorted([int(k) for k in STATE["instances"].keys()]),
        "output_dir":    STATE.get("output_dir"),
    }


# ============================================================
# 概念分割追踪页：视频 + 概念初始化 + SAM2 视频记忆传播
# ============================================================
class TrackingPromptRequest(BaseModel):
    frame_index: int = 0
    object_id: int = 1
    points: List[List[float]] = []
    labels: List[int] = []
    box: Optional[List[float]] = None
    name: str = "object"


class TrackingConceptRequest(BaseModel):
    name: str
    confidence: float = 0.20


def _tracking_predictor():
    global TRACK_MODEL
    if TRACK_MODEL is None:
        print(f"[tracking] loading SAM2 video predictor on {device} ...")
        TRACK_MODEL = build_sam2_video_predictor(
            SAM2_CONFIG, SAM2_CHECKPOINT, device=device
        )
        print("[tracking] video predictor ready")
    return TRACK_MODEL


def _tracking_mask_preview(frame_bgr, masks, object_ids, boxes=None, box_labels=None):
    image = frame_bgr.copy()
    palette = COLORS
    if masks is not None:
        mask_array = masks.detach().float().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
        if mask_array.ndim == 4:
            mask_array = mask_array[:, 0]
        for index, mask in enumerate(mask_array):
            mask = mask > 0
            color = np.array(palette[index % len(palette)], dtype=np.uint8)
            image[mask] = (image[mask].astype(np.float32) * 0.38 + color * 0.62).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, tuple(int(v) for v in color), 2)
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                label = str(object_ids[index]) if index < len(object_ids) else str(index + 1)
                cv2.putText(image, label, (x, max(22, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tuple(int(v) for v in color), 2, cv2.LINE_AA)
    for index, box in enumerate(boxes or []):
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        x1 = max(0, min(image.shape[1] - 1, x1))
        x2 = max(0, min(image.shape[1] - 1, x2))
        y1 = max(0, min(image.shape[0] - 1, y1))
        y2 = max(0, min(image.shape[0] - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 215, 255), 3)
        label = (
            str(box_labels[index])
            if box_labels and index < len(box_labels)
            else f"bbox {index + 1}"
        )
        cv2.putText(
            image,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None


VIDEO_SAMPLE_INTERVAL = 1.0


def _video_sample_indices(
    frame_count,
    fps,
    interval=VIDEO_SAMPLE_INTERVAL,
    randomize=False,
    seed=None,
):
    """Return one frame per interval, optionally randomized within each interval."""
    if frame_count <= 0:
        return [0]

    step = max(1, int(round((fps or 25.0) * interval)))
    if not randomize:
        indices = list(range(0, frame_count, step))
    else:
        rng = random.Random(seed)
        indices = [0]
        for start in range(step, frame_count, step):
            end = min(start + step, frame_count)
            indices.append(rng.randrange(start, end))

    if not randomize and indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return sorted(set(indices))


def _mask_stats(masks):
    if masks is None:
        return 0, 0
    if torch.is_tensor(masks):
        masks = masks.detach().float().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    if masks.ndim != 3:
        return 0, 0
    mask_array = masks > 0.5
    return int(mask_array.shape[0]), int(mask_array.sum())


def _new_video_timeline(
    frames_dir,
    frame_count,
    fps,
    route_prefix,
    randomize=False,
    seed=None,
):
    return [
        {
            "frame_index": int(frame_index),
            "second": round(float(frame_index) / max(fps or 25.0, 1e-6), 2),
            "instance_count": 0,
            "area": 0,
            "status": "pending",
            "preview_url": f"{route_prefix}/timeline/{frame_index}",
        }
        for frame_index in _video_sample_indices(
            frame_count,
            fps,
            randomize=randomize,
            seed=seed,
        )
    ]


def _update_video_timeline(state, root, frame_index, frame, masks, object_ids, boxes=None, box_labels=None):
    """Save a sampled preview and update its instance statistics."""
    timeline = state.get("timeline", [])
    item = next((entry for entry in timeline if entry["frame_index"] == int(frame_index)), None)
    if item is None:
        return
    preview_dir = os.path.join(root, "timeline")
    os.makedirs(preview_dir, exist_ok=True)
    preview = _tracking_mask_preview(frame, masks, object_ids, boxes=boxes, box_labels=box_labels)
    if preview:
        preview_path = os.path.join(preview_dir, f"{int(frame_index):06d}.jpg")
        with open(preview_path, "wb") as handle:
            handle.write(base64.b64decode(preview))
    count, area = _mask_stats(masks)
    item.update({
        "instance_count": count,
        "area": area,
        "status": "done",
    })


def _update_sam3_timeline(state, root, frame_index, frame, outputs):
    timeline = state.get("timeline", [])
    item = next((entry for entry in timeline if entry["frame_index"] == int(frame_index)), None)
    if item is None:
        return
    preview_dir = os.path.join(root, "timeline")
    os.makedirs(preview_dir, exist_ok=True)
    output_image = _sam3_output_image(frame, outputs)
    ok, encoded = cv2.imencode(".jpg", output_image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if ok:
        with open(os.path.join(preview_dir, f"{int(frame_index):06d}.jpg"), "wb") as handle:
            handle.write(encoded.tobytes())
    masks = _sam3_output_masks(outputs)
    count, area = _mask_stats(masks)
    item.update({
        "instance_count": count,
        "area": area,
        "status": "done",
        "mask": masks,  # 保存 mask 用于后续帧的提示提取
    })


def _timeline_status(state):
    return {
        "timeline": state.get("timeline", []),
        "current_frame": int(state.get("current_frame", 0)),
        "current_time": round(float(state.get("current_time", 0.0)), 2),
        "sample_interval": float(state.get("sample_interval", VIDEO_SAMPLE_INTERVAL)),
    }


@app.post("/concept-tracking/upload")
async def tracking_upload(file: UploadFile = File(...)):
    data = await file.read()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = os.path.join(OUTPUT_ROOT, "concept_tracking", stamp)
    frames_dir = os.path.join(root, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    video_path = os.path.join(root, os.path.basename(file.filename or "input.mp4"))
    with open(video_path, "wb") as handle:
        handle.write(data)
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return {"status": "error", "msg": "无法打开视频，请使用 mp4/avi/mov 文件"}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    count = 0
    first = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if first is None:
            first = frame.copy()
        cv2.imwrite(os.path.join(frames_dir, f"{count:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        count += 1
    capture.release()
    if not count:
        return {"status": "error", "msg": "视频没有可读取的帧"}
    with TRACK_LOCK:
        timeline = _new_video_timeline(
            frames_dir, count, fps, "/concept-tracking",
            randomize=True,
            seed=int(time.time()),
        )
        TRACK_STATE.update({
            "video_path": video_path, "frames_dir": frames_dir,
            "preview_path": os.path.join(root, "tracked.mp4"),
            "width": width, "height": height, "fps": fps, "frame_count": count,
            "inference_state": None, "prompted": False, "progress": 0,
            "current_frame": 0, "current_time": 0.0,
            "sample_interval": VIDEO_SAMPLE_INTERVAL, "timeline": timeline,
            "run_mode": None,
            "prompt_specs": {},
            "status": "uploaded", "message": "视频已上传，请输入概念或在首帧框选目标", "error": None,
        })
    # 让概念检测复用现有图片检测器的首帧输入。
    if first is not None:
        STATE.update({"image_bgr": first, "image_rgb": cv2.cvtColor(first, cv2.COLOR_BGR2RGB), "h": height, "w": width})
    return {
        "status": "ok", "width": width, "height": height, "fps": fps,
        "frame_count": count, "sample_interval": VIDEO_SAMPLE_INTERVAL,
        "timeline": timeline,
        "video_url": f"/concept-tracking/video/{stamp}/{os.path.basename(video_path)}",
    }


@app.get("/concept-tracking/video/{stamp}/{filename}")
def tracking_video(stamp: str, filename: str):
    """Serve the uploaded source video with native browser seek controls."""
    safe_filename = os.path.basename(filename)
    video_path = os.path.join(OUTPUT_ROOT, "concept_tracking", stamp, safe_filename)
    if not os.path.isfile(video_path):
        return {"status": "error", "msg": "视频文件不存在"}
    return FileResponse(
        video_path,
        media_type=mimetypes.guess_type(video_path)[0] or "video/mp4",
        filename=safe_filename,
    )


@app.post("/concept-tracking/prompt")
def tracking_prompt(req: TrackingPromptRequest):
    with TRACK_LOCK:
        if not TRACK_STATE["frames_dir"]:
            return {"status": "error", "msg": "请先上传视频"}
        try:
            points = np.asarray(req.points, dtype=np.float32) if req.points else None
            labels = np.asarray(req.labels, dtype=np.int32) if req.labels else None
            box = np.asarray(req.box, dtype=np.float32) if req.box else None
            frame_idx = int(req.frame_index)
            frame = cv2.imread(
                os.path.join(TRACK_STATE["frames_dir"], f"{frame_idx:06d}.jpg")
            )
            if frame is None:
                raise RuntimeError("无法读取首帧")
            with TRACK_INFERENCE_LOCK:
                predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                masks, _, _ = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=box,
                    multimask_output=False,
                )
            mask_values = _sam2_batch_masks(masks)
            if not mask_values:
                raise RuntimeError("首帧提示没有生成有效 mask")
            object_ids = [int(req.object_id)]
            TRACK_STATE["prompted"] = True
            TRACK_STATE["status"] = "prompted"
            TRACK_STATE["message"] = f"已生成 {req.name} 首帧 mask，可开始逐秒抽样分割"
            TRACK_STATE["prompt_specs"][int(req.object_id)] = {
                "name": req.name or "object",
                "points": req.points or [],
                "labels": req.labels or [],
                "box": req.box,
            }
            _update_video_timeline(
                TRACK_STATE, os.path.dirname(TRACK_STATE["frames_dir"]),
                frame_idx, frame, np.asarray([mask_values[0]], dtype=np.uint8), object_ids,
            )
            return {
                "status": "ok",
                "frame_index": frame_idx,
                "object_ids": object_ids,
                "preview": _tracking_mask_preview(
                    frame, np.asarray([mask_values[0]], dtype=np.uint8), object_ids
                ),
            }
        except Exception as exc:
            traceback.print_exc()
            TRACK_STATE["status"] = "error"
            TRACK_STATE["error"] = str(exc)
            return {"status": "error", "msg": f"初始分割失败: {exc}"}


@app.post("/concept-tracking/concept")
def tracking_concept(req: TrackingConceptRequest):
    name = req.name.strip()
    if not name:
        return {"status": "error", "msg": "请输入概念名称"}
    with TRACK_LOCK:
        frames_dir = TRACK_STATE.get("frames_dir")
        width = int(TRACK_STATE.get("width") or 0)
        height = int(TRACK_STATE.get("height") or 0)
    if not frames_dir:
        return {"status": "error", "msg": "请先上传视频"}
    frame_path = os.path.join(frames_dir, "000000.jpg")
    frame = cv2.imread(frame_path)
    if frame is None:
        return {"status": "error", "msg": "无法读取首帧"}
    try:
        analysis = _gpt_analyze_bboxes_only(frame, [name])
        gpt_targets = analysis.get("targets", [])
        boxes = []
        for target in gpt_targets:
            target_name = str(target.get("name") or name).strip()
            if target_name and target_name.lower() not in {name.lower()}:
                target_name = name
            for box in target.get("instance_boxes", []) or []:
                if len(box) != 4:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                boxes.append([
                    max(0.0, min(1.0, x1)) * width,
                    max(0.0, min(1.0, y1)) * height,
                    max(0.0, min(1.0, x2)) * width,
                    max(0.0, min(1.0, y2)) * height,
                ])
        if not boxes:
            return {"status": "error", "msg": "大模型没有返回可用 bbox"}

        frame_masks = []
        object_ids = []
        with TRACK_INFERENCE_LOCK:
            predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for index, box in enumerate(boxes, start=1):
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=np.asarray(box, dtype=np.float32),
                    multimask_output=False,
                )
                values = _sam2_batch_masks(masks)
                if not values:
                    continue
                frame_masks.append(values[0])
                object_ids.append(index)
        if not frame_masks:
            return {"status": "error", "msg": "bbox 没有生成有效 mask"}

        with TRACK_LOCK:
            TRACK_STATE["prompt_specs"] = {
                int(object_id): {
                    "name": name,
                    "points": [],
                    "labels": [],
                    "box": boxes[index],
                    "gpt_prompt": name,
                }
                for index, object_id in enumerate(object_ids)
            }
            TRACK_STATE["prompted"] = True
            TRACK_STATE["status"] = "prompted"
            TRACK_STATE["message"] = (
                f"大模型 bbox 初始化完成，已建立 {len(object_ids)} 个抽样分割目标"
            )
            TRACK_STATE["gpt_analysis"] = analysis
            _update_video_timeline(
                TRACK_STATE,
                os.path.dirname(frames_dir),
                0,
                frame,
                np.asarray(frame_masks, dtype=np.uint8),
                object_ids,
                boxes=boxes,
                box_labels=[name] * len(boxes),
            )
        return {
            "status": "ok",
            "frame_index": 0,
            "object_ids": object_ids,
            "boxes": boxes,
            "analysis": analysis,
            "preview": _tracking_mask_preview(
                frame,
                np.asarray(frame_masks, dtype=np.uint8),
                object_ids,
                boxes=boxes,
                box_labels=[name] * len(boxes),
            ),
        }
    except Exception as exc:
        traceback.print_exc()
        with TRACK_LOCK:
            TRACK_STATE["status"] = "error"
            TRACK_STATE["error"] = str(exc)
        return {"status": "error", "msg": f"大模型 bbox 初始化失败: {exc}"}


def _run_tracking_job(sample_only=True):
    try:
        with TRACK_LOCK:
            frame_count = TRACK_STATE["frame_count"]
            frames_dir = TRACK_STATE["frames_dir"]
            fps, width, height = TRACK_STATE["fps"], TRACK_STATE["width"], TRACK_STATE["height"]
            timeline = list(TRACK_STATE["timeline"])
            prompt_specs = {
                int(object_id): dict(spec)
                for object_id, spec in TRACK_STATE["prompt_specs"].items()
            }
            TRACK_STATE.update({
                "status": "running",
                "progress": 0,
                "run_mode": "sampled",
                "message": "正在对每秒随机抽取帧独立分割",
            })
        if not prompt_specs:
            raise RuntimeError("没有可用于独立帧分割的首帧提示")

        sample_frames = [int(item["frame_index"]) for item in timeline]
        with TRACK_INFERENCE_LOCK:
            for done, frame_idx in enumerate(sample_frames, start=1):
                frame = cv2.imread(os.path.join(frames_dir, f"{frame_idx:06d}.jpg"))
                if frame is None:
                    raise RuntimeError(f"无法读取采样帧 {frame_idx}")
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_masks = []
                frame_ids = []
                for object_id, spec in prompt_specs.items():
                    predictor.set_image(image_rgb)
                    points = spec.get("points") or None
                    labels = spec.get("labels") or None
                    box = spec.get("box") or None
                    masks, _, _ = predictor.predict(
                        point_coords=np.asarray(points, dtype=np.float32) if points else None,
                        point_labels=np.asarray(labels, dtype=np.int32) if labels else None,
                        box=np.asarray(box, dtype=np.float32) if box else None,
                        multimask_output=False,
                    )
                    values = _sam2_batch_masks(masks)
                    if values:
                        frame_masks.append(values[0])
                        frame_ids.append(int(object_id))
                _update_video_timeline(
                    TRACK_STATE, os.path.dirname(frames_dir),
                    frame_idx, frame,
                    np.asarray(frame_masks, dtype=np.uint8) if frame_masks else None,
                    frame_ids,
                )
                with TRACK_LOCK:
                    TRACK_STATE["current_frame"] = int(frame_idx)
                    TRACK_STATE["current_time"] = float(frame_idx) / max(fps or 25.0, 1e-6)
                    TRACK_STATE["progress"] = int(done * 100 / max(len(sample_frames), 1))
        with TRACK_LOCK:
            TRACK_STATE.update({
                "status": "done",
                "progress": 100,
                "message": "每秒随机抽样独立分割完成",
            })
    except Exception as exc:
        traceback.print_exc()
        with TRACK_LOCK:
            TRACK_STATE.update({"status": "error", "error": str(exc), "message": f"追踪失败: {exc}"})


@app.post("/concept-tracking/start")
def tracking_start():
    with TRACK_LOCK:
        if not TRACK_STATE["prompted"]:
            return {"status": "error", "msg": "请先输入概念或在首帧点击/框选目标"}
        if TRACK_STATE["status"] == "running":
            return {"status": "ok"}
        threading.Thread(target=_run_tracking_job, daemon=True).start()
    return {"status": "ok", "message": "已开始每秒随机抽样独立分割"}


@app.post("/concept-tracking/start-sampled")
def tracking_start_sampled():
    """Run SAM2 propagation while displaying one random frame per second."""
    with TRACK_LOCK:
        if not TRACK_STATE["prompted"]:
            return {"status": "error", "msg": "请先输入概念或在首帧点击/框选目标"}
        if TRACK_STATE["status"] == "running":
            return {"status": "ok"}
        threading.Thread(target=_run_tracking_job, daemon=True).start()
    return {"status": "ok", "message": "已开始每秒随机抽样独立分割"}


@app.get("/concept-tracking/status")
def tracking_status():
    with TRACK_LOCK:
        response = {
            key: TRACK_STATE[key]
            for key in (
                "status", "progress", "message", "error", "frame_count",
                "width", "height", "fps",
                "run_mode",
            )
        }
        response.update(_timeline_status(TRACK_STATE))
        return response


@app.get("/concept-tracking/timeline/{frame_index}")
def tracking_timeline(frame_index: int):
    with TRACK_LOCK:
        frames_dir = TRACK_STATE.get("frames_dir")
        root = os.path.dirname(frames_dir) if frames_dir else None
    if not root or frame_index < 0:
        return {"status": "error", "msg": "采样帧不存在"}
    preview_path = os.path.join(root, "timeline", f"{frame_index:06d}.jpg")
    source_path = os.path.join(frames_dir, f"{frame_index:06d}.jpg")
    path = preview_path if os.path.isfile(preview_path) else source_path
    if not os.path.isfile(path):
        return {"status": "error", "msg": "采样帧不存在"}
    return FileResponse(path, media_type="image/jpeg")


@app.get("/concept-tracking/result")
def tracking_result():
    with TRACK_LOCK:
        path = TRACK_STATE.get("preview_path")
        if TRACK_STATE.get("status") != "done" or not path or not os.path.isfile(path):
            return {"status": "error", "msg": "追踪结果尚未生成"}
    return FileResponse(path, media_type="video/mp4", filename="concept_tracking.mp4")


# ============================================================
# SAM2 批量点击记忆传播：首图点击锅 -> 点击目标 -> 后续图片传播
# ============================================================
class Sam2BatchPromptRequest(BaseModel):
    stage: str
    points: List[List[float]] = []
    labels: List[int] = []
    box: Optional[List[float]] = None  # [x1, y1, x2, y2] 像素坐标
    name: str = "object"
    target_id: Optional[int] = None  # 目标 ID，用于多目标跟踪（从 2 开始）


class Sam2BatchGptInitRequest(BaseModel):
    targets: List[str] = []
    container_concept: str = ""
    prompt_box: Optional[List[float]] = None
    prompt_points: List[List[float]] = []
    prompt_labels: List[int] = []
    target_hints: Dict[str, Dict[str, Any]] = {}


class Sam2BatchRefineRequest(BaseModel):
    target_id: int
    points: List[List[float]] = []
    labels: List[int] = []
    name: str = "object"


# 点击点缓存文件路径
SAM2_CLICK_CACHE_FILE = os.path.join(OUTPUT_ROOT, "sam2_click_cache.json")


def _load_sam2_click_cache():
    """加载 SAM2 点击点缓存"""
    if os.path.exists(SAM2_CLICK_CACHE_FILE):
        try:
            with open(SAM2_CLICK_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save_sam2_click_cache(container_points, container_labels, container_box, targets, width, height):
    """保存 SAM2 点击点缓存（支持多目标）"""
    cache_targets = {}
    for object_id, target in (targets or {}).items():
        cache_targets[str(object_id)] = {
            "name": target.get("name", f"target_{object_id}"),
            "points": target.get("points", []),
            "labels": target.get("labels", []),
            "negative_points": target.get("negative_points", []),
            "manual_points": target.get("manual_points", []),
            "manual_labels": target.get("manual_labels", []),
            "gpt_prompt": target.get("gpt_prompt"),
        }
    cache = {
        "container_points": container_points,
        "container_labels": container_labels,
        "container_box": container_box,
        "targets": cache_targets,
        "width": width,
        "height": height,
        "timestamp": time.time(),
    }
    try:
        os.makedirs(os.path.dirname(SAM2_CLICK_CACHE_FILE), exist_ok=True)
        with open(SAM2_CLICK_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[sam2-batch] 保存点击缓存失败: {e}")


def _sam2_batch_masks(masks):
    if masks is None:
        return []
    values = masks.detach().float().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
    if values.ndim == 4:
        values = values[:, 0]
    elif values.ndim == 2:
        values = values[None, ...]
    return [np.asarray(mask > 0, dtype=bool) for mask in values]


def _sam2_public_targets(targets):
    return {
        str(object_id): {
            key: value
            for key, value in target.items()
            if key not in {"seed_mask"}
        }
        for object_id, target in (targets or {}).items()
    }


def _gpt_json_content(response):
    """Extract JSON text from a Chat Completions response."""
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("GPT 没有返回结果")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("GPT 返回内容为空")
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def _qwen_response_format_failed(detail):
    return (
        "response_format" in detail
        and (
            "invalid_parameter_error" in detail
            or "Model output became abnormal" in detail
            or "invalid JSON" in detail
        )
    )


def _gpt_fallback_targets(target_names):
    return [
        {
            "name": name,
            "prompt": _batch_model_prompt(name),
            "positive_points": [],
            "negative_points": [],
            "instance_boxes": [],
            "box": None,
        }
        for name in target_names
    ]


def _normalize_gpt_analysis(data, requested_targets):
    """Coerce common model response variants into {"targets": [...]}."""
    if isinstance(data, list):
        print("[GPT] 模型返回了顶层数组，已自动转换为 targets")
        data = {"targets": data}
    elif isinstance(data, dict):
        targets = data.get("targets")
        if not isinstance(targets, list):
            for key in ("target", "objects", "items", "results", "result"):
                value = data.get(key)
                if isinstance(value, list):
                    data = dict(data)
                    data["targets"] = value
                    print(f"[GPT] 模型返回字段 {key}，已自动转换为 targets")
                    break
                if isinstance(value, dict):
                    data = dict(data)
                    data["targets"] = [value]
                    print(f"[GPT] 模型返回单个字段 {key}，已自动转换为 targets")
                    break
    if not isinstance(data, dict) or not isinstance(data.get("targets"), list):
        raise RuntimeError(f"GPT 返回格式无效：缺少 targets。实际返回: {str(data)[:200]}")

    cleaned = [dict(item) for item in data["targets"] if isinstance(item, dict)]
    if not cleaned:
        print("[GPT] 模型返回 targets 为空，改用目标名称作为 SAM3 文本提示兜底")
        cleaned = _gpt_fallback_targets(requested_targets)
    while len(cleaned) < len(requested_targets):
        name = requested_targets[len(cleaned)]
        print(f"[GPT] 模型漏返目标 {name}，已添加兜底提示")
        cleaned.append(_gpt_fallback_targets([name])[0])

    for index, target in enumerate(cleaned):
        if not str(target.get("name") or "").strip() and index < len(requested_targets):
            target["name"] = requested_targets[index]
        if not str(target.get("prompt") or "").strip():
            target["prompt"] = _batch_model_prompt(target.get("name") or requested_targets[min(index, len(requested_targets) - 1)])
        target.setdefault("positive_points", [])
        target.setdefault("negative_points", [])
        target.setdefault("instance_boxes", [])
        target.setdefault("box", None)
    data = dict(data)
    data["targets"] = cleaned
    return data


def _openai_connection():
    """Load an OpenAI-compatible connection from process env or local config."""
    api_key = (
        os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("DASHSCOPE_API_KEY", "")
        or os.environ.get("QWEN_API_KEY", "")
    ).strip()
    base_url = (
        os.environ.get("OPENAI_BASE_URL", "")
        or os.environ.get("DASHSCOPE_BASE_URL", "")
        or os.environ.get("QWEN_BASE_URL", "")
    ).strip()
    model = (
        os.environ.get("OPENAI_VISION_MODEL", "")
        or os.environ.get("DASHSCOPE_MODEL", "")
        or os.environ.get("QWEN_MODEL", "")
    ).strip()

    # This project may be launched outside a shell, so also support a local
    # .env file without adding a dotenv dependency. Standard KEY=VALUE files
    # plus JSON/JSON-ish connector formats are accepted.
    env_path = os.path.join(APP_DIR, ".env")
    if os.path.isfile(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as env_file:
                raw = env_file.read().strip()
            if raw.startswith("{"):
                config = json.loads(raw)
                provider_name = None
                configured_model = str(config.get("model") or "").strip()
                if "/" in configured_model:
                    provider_name, configured_model = configured_model.split("/", 1)
                providers = config.get("provider") if isinstance(config.get("provider"), dict) else {}
                provider_config = (
                    providers.get(provider_name)
                    if provider_name and isinstance(providers.get(provider_name), dict)
                    else None
                )
                if provider_config is None and len(providers) == 1:
                    provider_config = next(iter(providers.values()))
                provider_options = (
                    provider_config.get("options", {})
                    if isinstance(provider_config, dict)
                    and isinstance(provider_config.get("options"), dict)
                    else {}
                )
                api_key = api_key or str(
                    config.get("key")
                    or config.get("apiKey")
                    or provider_options.get("apiKey")
                    or provider_options.get("key")
                    or ""
                ).strip()
                base_url = base_url or str(
                    config.get("url")
                    or config.get("baseURL")
                    or config.get("baseUrl")
                    or provider_options.get("baseURL")
                    or provider_options.get("baseUrl")
                    or provider_options.get("url")
                    or ""
                ).strip()
                model = (
                    configured_model
                    or model
                )
            else:
                for line in raw.splitlines():
                    line = line.strip().rstrip(",")
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                    elif ":" in line:
                        key, value = line.split(":", 1)
                    else:
                        continue
                    key = key.strip().strip("\"'")
                    value = value.strip().strip("\"'")
                    if key in {"OPENAI_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY", "apiKey", "key"} and not api_key:
                        api_key = value
                    elif key in {"OPENAI_BASE_URL", "DASHSCOPE_BASE_URL", "QWEN_BASE_URL", "baseURL", "baseUrl", "url"} and not base_url:
                        base_url = value
                    elif key in {"OPENAI_VISION_MODEL", "DASHSCOPE_MODEL", "QWEN_MODEL", "model"}:
                        model = value or model
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(f".env 配置无法读取: {exc}") from exc

    if api_key:
        cleaned_key = "".join(ch for ch in api_key if 32 <= ord(ch) <= 126).strip()
        if cleaned_key != api_key:
            print("[GPT] API key 含有非 ASCII 字符，已自动移除；请检查 .env 中的 QWEN_API_KEY")
            api_key = cleaned_key
    if not api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY，无法使用 GPT 分析首图")
    if not base_url:
        if model.startswith("qwen") or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY"):
            base_url = "https://dashscope.aliyuncs.com/compatible-mode"
        else:
            base_url = "https://api.openai.com"
    if not model:
        model = "qwen-vl-max" if "aliyuncs.com" in base_url or "dashscope" in base_url else "gpt-4o-mini"
    if "/" in model:
        model = model.split("/", 1)[1]
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return api_key, f"{base_url}/chat/completions", model


def _gpt_analyze_first_image(image_bgr, targets, roi_note=""):
    """Ask GPT Vision for normalized positive/negative seed points."""
    api_key, endpoint, model = _openai_connection()
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("首图编码失败")
    image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
    target_text = "、".join(targets)
    system_prompt = (
        "你是一个用于图像分割初始化的视觉标注助手。"
        "请只返回严格 JSON 对象，不要 Markdown。"
        "最外层必须是包含 targets 数组的对象。"
    )
    user_prompt = f"""
分析这张食物图片，为以下类别提供用于 SAM3 分割的首图提示：{target_text}。
目标是覆盖每个类别尽可能多的可见区域，而不是只找一个实例。
{roi_note}
对每个类别返回 4 到 16 个正点，正点必须落在目标食材内部，不能落在边缘、汤汁、油光、阴影、盘子或锅上。
**重要**：先判断目标是一整块还是分散的多块：
- 如果目标是一整块（如一整块炒鸡蛋、一整块肉、一整片藕片），只需返回 4-6 个正点，均匀分布在这一整块上即可，不要过度分散。
- 如果目标是分散的多块（如多块番茄、多块土豆），每个清晰可见块至少给 1 个正点，总共不超过 16 个。
返回 2 到 8 个负点，负点应落在容易混淆的其他食材、盘子/锅、汤汁、背景或非目标区域上。
如果目标类别是食材的子部位或细粒度类别，不要退化成父类别整体；例如"蛋黄"只点橙黄蛋黄区域，"蛋白"只点浅黄/白色蛋白区域，不要返回整颗鸡蛋。
必须为用户请求的每一个类别都返回一个 targets 项，禁止返回空数组 []。
如果一个类别在图中不可见或不确定，仍返回该类别，但 positive_points 为空，并在 prompt 中保持原类别语义。
如果用户目标本身是炒鸡蛋、煎鸡蛋、荷包蛋等蛋类菜品，就按对应蛋类目标返回，不要因为它是菜名而返回空数组。
坐标全部使用 0 到 1 的归一化格式 [x, y]，原点在左上角。
**重要**：为每个正点返回一个对应的实例框，框要紧密包围该实例，不要覆盖其他实例或非目标区域。
instance_boxes 数组的长度必须与 positive_points 相同，每个框对应一个正点。
同时给出覆盖该类别所有实例的归一化总框 [x1, y1, x2, y2]；总框应尽量覆盖目标的主要可见块，但不要覆盖过多非目标区域。
prompt 使用简短英文视觉短语，必须尽量具体描述外观，例如 tomato pieces、scrambled egg pieces、Chinese broccoli stems and leaves、sliced beef pieces；对于藕片使用 lotus root slices（白色/浅黄色圆片，带多个规则孔洞）。
不要把"番茄炒鸡蛋"当作一个整体类别，番茄和鸡蛋必须分别返回。

JSON 格式：
{{
  "targets": [
    {{
      "name": "番茄",
      "prompt": "tomato pieces",
      "positive_points": [[0.1, 0.2], [0.3, 0.4]],
      "negative_points": [[0.5, 0.6]],
      "instance_boxes": [[0.08, 0.18, 0.15, 0.25], [0.28, 0.38, 0.35, 0.45]],
      "box": [0.05, 0.15, 0.40, 0.50]
    }}
  ]
}}
"""
    messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "high",
                        },
                    },
                ],
            },
    ]

    model_candidates = [model]
    if "aliyuncs.com" in endpoint or "dashscope" in endpoint:
        # 添加多个备选模型，按优先级排序
        model_candidates.extend([
            "qwen-vl-max-latest",
            "qwen-vl-plus-latest",
            "qwen-vl-max",
            "qwen-vl-plus",
        ])
    model_candidates = list(dict.fromkeys([item for item in model_candidates if item]))

    result = None
    last_error = None
    for candidate in model_candidates:
        for use_response_format in (True, False):
            payload = {
                "model": candidate,
                "temperature": 0,
                "messages": messages,
            }
            if use_response_format:
                payload["response_format"] = {"type": "json_object"}
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    result = json.loads(response.read().decode(charset, errors="replace"))
                    if isinstance(result, dict):
                        result["_model_used"] = candidate
                    break
            except urllib.error.HTTPError as exc:
                raw_detail = exc.read()
                charset = exc.headers.get_content_charset() if exc.headers else None
                detail = raw_detail.decode(charset or "utf-8", errors="replace")[:800]
                last_error = f"GPT 请求失败 HTTP {exc.code}: {detail}"
                if "model_not_found" in detail or "Model not exist" in detail or "access_denied" in detail or "Access denied" in detail:
                    print(f"[GPT] 模型不可用或无权限访问，尝试下一个模型: {candidate}")
                    break
                if use_response_format and _qwen_response_format_failed(detail):
                    print("[GPT] 千问 JSON 结构化输出中断，改用普通 JSON 提示重试")
                    continue
                raise RuntimeError(last_error) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"GPT 网络请求失败: {exc.reason}") from exc
            break
        if result is not None:
            break
    if result is None:
        raise RuntimeError(last_error or "GPT 请求失败：没有可用模型")
    data = _normalize_gpt_analysis(_gpt_json_content(result), targets)
    print(f"[GPT] 解析后的数据结构: {type(data)}, keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    print(f"[GPT] 数据内容预览: {str(data)[:500]}")
    if isinstance(data, dict) and isinstance(result, dict) and result.get("_model_used"):
        data["_model_used"] = result["_model_used"]
    return data


def _normalized_points(value):
    points = []
    for point in value or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            continue
        if 0 <= x <= 1 and 0 <= y <= 1:
            points.append([x, y])
    return points


def _normalized_box(value):
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _normalized_boxes(value):
    boxes = []
    for box in value or []:
        norm_box = _normalized_box(box)
        if norm_box is not None:
            boxes.append(norm_box)
    return boxes


def _box_xyxy_to_sam3_prompt(box):
    x1, y1, x2, y2 = box
    return [
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        max(0.001, x2 - x1),
        max(0.001, y2 - y1),
    ]


def _normalize_prompt_points(points, width, height):
    normalized = []
    for point in points or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            continue
        if x > 1 or y > 1:
            x /= max(width, 1)
            y /= max(height, 1)
        if 0 <= x <= 1 and 0 <= y <= 1:
            normalized.append([x, y])
    return normalized


def _merge_manual_target_hints(target, target_hints, width, height):
    name = str(target.get("name") or "").strip()
    hint = None
    for key, value in (target_hints or {}).items():
        if str(key).strip().lower() == name.lower():
            hint = value
            break
    if not isinstance(hint, dict):
        return target, 0, 0

    merged = dict(target)
    points = _normalize_prompt_points(hint.get("points"), width, height)
    labels = hint.get("labels") or [1] * len(points)
    positives = []
    negatives = []
    for index, point in enumerate(points):
        label = labels[index] if index < len(labels) else 1
        if int(label) == 0:
            negatives.append(point)
        else:
            positives.append(point)

    merged["positive_points"] = _normalized_points(merged.get("positive_points")) + positives
    merged["negative_points"] = _normalized_points(merged.get("negative_points")) + negatives
    return merged, len(positives), len(negatives)


# HSV 颜色范围定义（用于快速预过滤）
# 格式: {"H": [(h_min, h_max), ...], "S": (s_min, s_max), "V": (v_min, v_max)}
_HSV_COLOR_RANGES = {
    # 番茄/红色食物（放宽范围以检测更多番茄区域）
    "tomato": {"H": [(0, 15), (165, 180)], "S": (30, 255), "V": (30, 255)},
    "番茄": {"H": [(0, 15), (165, 180)], "S": (30, 255), "V": (30, 255)},
    "辣椒": {"H": [(0, 10), (170, 180)], "S": (50, 255), "V": (50, 255)},
    "西瓜": {"H": [(0, 10), (170, 180)], "S": (50, 255), "V": (50, 255)},
    # 鸡蛋/黄色食物
    "egg": {"H": (20, 45), "S": (30, 200), "V": (150, 255)},
    "鸡蛋": {"H": (20, 45), "S": (30, 200), "V": (150, 255)},
    "蛋黄": {"H": (20, 40), "S": (80, 255), "V": (150, 255)},
    "蛋白": {"H": (0, 40), "S": (0, 80), "V": (200, 255)},
    # 青菜/绿色食物
    "vegetable": {"H": (60, 120), "S": (50, 255), "V": (50, 255)},
    "青菜": {"H": (60, 120), "S": (50, 255), "V": (50, 255)},
    "菜": {"H": (60, 120), "S": (50, 255), "V": (50, 255)},
    "黄瓜": {"H": (70, 130), "S": (40, 200), "V": (80, 255)},
    "青椒": {"H": (80, 140), "S": (60, 255), "V": (60, 255)},
    "生菜": {"H": (70, 130), "S": (40, 200), "V": (100, 255)},
    # 肉类/棕色食物
    "meat": {"H": (5, 25), "S": (50, 200), "V": (50, 180)},
    "牛肉": {"H": (5, 25), "S": (50, 200), "V": (50, 180)},
    "猪肉": {"H": (5, 25), "S": (50, 200), "V": (50, 180)},
    "鸡肉": {"H": (10, 30), "S": (40, 180), "V": (100, 220)},
    # 米饭/白色食物
    "rice": {"H": (0, 40), "S": (0, 50), "V": (180, 255)},
    "米饭": {"H": (0, 40), "S": (0, 50), "V": (180, 255)},
    "馒头": {"H": (20, 40), "S": (10, 60), "V": (200, 255)},
    "面包": {"H": (20, 40), "S": (20, 80), "V": (180, 255)},
    # 藕片通常接近白色，受冷光/反光影响色相不稳定；不要用窄黄色范围
    # 把低饱和度的藕片 mask 过滤掉，最终由 GPT/点提示做语义筛选。
    "lotus_root": {"H": (0, 179), "S": (0, 120), "V": (150, 255)},
    "藕片": {"H": (0, 179), "S": (0, 120), "V": (150, 255)},
    "莲藕": {"H": (0, 179), "S": (0, 120), "V": (150, 255)},
    # 胡萝卜/橙色食物
    "carrot": {"H": (10, 25), "S": (100, 255), "V": (150, 255)},
    "胡萝卜": {"H": (10, 25), "S": (100, 255), "V": (150, 255)},
    "南瓜": {"H": (15, 30), "S": (80, 255), "V": (150, 255)},
    "橙子": {"H": (15, 30), "S": (100, 255), "V": (150, 255)},
    # 土豆/浅黄色食物
    "potato": {"H": (25, 45), "S": (30, 120), "V": (150, 230)},
    "土豆": {"H": (25, 45), "S": (30, 120), "V": (150, 230)},
    "马铃薯": {"H": (25, 45), "S": (30, 120), "V": (150, 230)},
    # 玉米/金黄色食物
    "corn": {"H": (35, 55), "S": (80, 255), "V": (180, 255)},
    "玉米": {"H": (35, 55), "S": (80, 255), "V": (180, 255)},
    # 茄子/紫色食物
    "eggplant": {"H": (130, 160), "S": (50, 200), "V": (50, 200)},
    "茄子": {"H": (130, 160), "S": (50, 200), "V": (50, 200)},
    "紫甘蓝": {"H": (130, 160), "S": (50, 200), "V": (50, 200)},
    # 香菇/深棕色食物
    "mushroom": {"H": (10, 25), "S": (40, 150), "V": (50, 150)},
    "香菇": {"H": (10, 25), "S": (40, 150), "V": (50, 150)},
    "木耳": {"H": (0, 20), "S": (20, 100), "V": (30, 120)},
    "海带": {"H": (80, 120), "S": (50, 200), "V": (30, 150)},
    # 火腿/粉红色食物
    "ham": {"H": [(0, 10), (170, 180)], "S": (30, 150), "V": (180, 255)},
    "火腿": {"H": [(0, 10), (170, 180)], "S": (30, 150), "V": (180, 255)},
    "香肠": {"H": [(0, 10), (170, 180)], "S": (40, 180), "V": (150, 230)},
    "虾": {"H": [(0, 15), (165, 180)], "S": (60, 200), "V": (150, 255)},
}


def _calculate_mask_hsv(image_rgb, mask):
    """计算 mask 区域的平均 HSV 值"""
    # RGB 转 HSV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # 提取 mask 区域的像素
    pixels = image_hsv[mask]
    if len(pixels) == 0:
        return None
    
    # 计算平均值
    avg_hsv = np.mean(pixels, axis=0)
    return avg_hsv


def _match_hsv_range(avg_hsv, color_range):
    """检查 HSV 值是否在指定范围内"""
    if avg_hsv is None or color_range is None:
        return False
    
    h, s, v = avg_hsv
    
    # 检查 H（色相）
    h_range = color_range.get("H")
    if h_range:
        if isinstance(h_range[0], tuple):
            # 多个范围（如红色有两个范围）
            h_match = any(h_min <= h <= h_max for h_min, h_max in h_range)
        else:
            # 单个范围
            h_match = h_range[0] <= h <= h_range[1]
        if not h_match:
            return False
    
    # 检查 S（饱和度）
    s_range = color_range.get("S")
    if s_range and not (s_range[0] <= s <= s_range[1]):
        return False
    
    # 检查 V（亮度）
    v_range = color_range.get("V")
    if v_range and not (v_range[0] <= v <= v_range[1]):
        return False
    
    return True


def _hsv_filter_masks(image_rgb, masks, target_name, target_prompt):
    """使用 HSV 颜色空间快速过滤 mask"""
    # 查找目标的颜色范围
    color_range = None
    target_lower = target_name.lower()
    
    # 先尝试精确匹配
    if target_name in _HSV_COLOR_RANGES:
        color_range = _HSV_COLOR_RANGES[target_name]
    else:
        # 再尝试模糊匹配
        for key, value in _HSV_COLOR_RANGES.items():
            if key in target_lower or target_lower in key:
                color_range = value
                break
    
    # 如果没有定义颜色范围，返回所有 mask
    if color_range is None:
        print(f"[HSV-filter] {target_name}: 未定义颜色范围，跳过过滤")
        return masks, []
    
    print(f"[HSV-filter] {target_name}: 使用颜色范围 H={color_range.get('H')}, S={color_range.get('S')}, V={color_range.get('V')}")
    
    matched_masks = []
    matched_indices = []
    unmatched_masks = []
    
    for idx, mask in enumerate(masks):
        avg_hsv = _calculate_mask_hsv(image_rgb, mask)
        if avg_hsv is None:
            unmatched_masks.append((idx, mask))
            continue
        
        if _match_hsv_range(avg_hsv, color_range):
            matched_masks.append(mask)
            matched_indices.append(idx)
            print(f"[HSV-filter] mask {idx}: HSV=({avg_hsv[0]:.1f}, {avg_hsv[1]:.1f}, {avg_hsv[2]:.1f}) ✓ 匹配")
        else:
            unmatched_masks.append((idx, mask))
            print(f"[HSV-filter] mask {idx}: HSV=({avg_hsv[0]:.1f}, {avg_hsv[1]:.1f}, {avg_hsv[2]:.1f}) ✗ 不匹配")
    
    print(f"[HSV-filter] {target_name}: 颜色匹配 {len(matched_masks)}/{len(masks)} 个 mask")
    return matched_masks, matched_indices


def _visualize_masks_for_gpt(image_rgb, masks, max_masks=30):
    """Visualize masks with numbered overlays for GPT to review."""
    from PIL import ImageDraw, ImageFont
    height, width = image_rgb.shape[:2]
    vis_image = image_rgb.copy()
    
    # Limit number of masks to avoid overwhelming GPT
    masks_to_show = masks[:max_masks]
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]
    
    for idx, mask in enumerate(masks_to_show):
        color = colors[idx % len(colors)]
        # Create colored overlay
        mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        mask_rgb[mask] = color
        # Blend with original image
        vis_image = cv2.addWeighted(vis_image, 1.0, mask_rgb, 0.5, 0)
        
        # Add number label at mask center
        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            center_x = int(np.mean(xs))
            center_y = int(np.mean(ys))
            # Draw number
            cv2.circle(vis_image, (center_x, center_y), 15, color, -1)
            cv2.putText(vis_image, str(idx), (center_x - 8, center_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis_image, len(masks_to_show)


def _gpt_filter_masks(image_bgr, masks, target_name, target_prompt):
    """Ask GPT to filter which masks belong to the target category."""
    api_key, endpoint, model = _openai_connection()
    
    # Visualize masks
    vis_image, num_shown = _visualize_masks_for_gpt(image_bgr, masks)
    ok, encoded = cv2.imencode(".jpg", vis_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("可视化图片编码失败")
    image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
    
    system_prompt = (
        "你是一个图像分割质量审核助手。"
        "请只返回严格 JSON，不要 Markdown。"
    )
    user_prompt = f"""
分析这张图片，图中有 {num_shown} 个编号的 mask（用不同颜色和数字标记）。
判断每个 mask 是否属于目标类别：{target_name}（英文描述：{target_prompt}）。

规则：
1. 只保留真正属于 {target_name} 的 mask
2. 排除属于其他食材、盘子、锅、背景、汤汁、阴影的 mask
3. 如果 mask 边界不清晰或不确定，也排除
4. 返回保留的 mask 编号列表

JSON 格式：
{{
  "accepted_masks": [0, 2, 5, 7],
  "rejected_masks": [1, 3, 4, 6],
  "reason": "简要说明筛选理由"
}}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    
    try:
        last_error = None
        for use_response_format in (True, False):
            payload = {
                "model": model,
                "temperature": 0,
                "messages": messages,
            }
            if use_response_format:
                payload["response_format"] = {"type": "json_object"}
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    result = json.loads(response.read().decode(charset, errors="replace"))
                    filter_result = _gpt_json_content(result)
                    print(f"[GPT-filter] {target_name}: 接受 {filter_result.get('accepted_masks', [])}, 拒绝 {filter_result.get('rejected_masks', [])}")
                    print(f"[GPT-filter] 理由: {filter_result.get('reason', '无')}")
                    return filter_result
            except urllib.error.HTTPError as exc:
                raw_detail = exc.read()
                charset = exc.headers.get_content_charset() if exc.headers else None
                detail = raw_detail.decode(charset or "utf-8", errors="replace")[:800]
                last_error = f"GPT 请求失败 HTTP {exc.code}: {detail}"
                if use_response_format and _qwen_response_format_failed(detail):
                    print("[GPT-filter] 千问 JSON 结构化输出中断，改用普通 JSON 提示重试")
                    continue
                raise RuntimeError(last_error) from exc
        raise RuntimeError(last_error or "GPT 筛选请求失败")
    except Exception as exc:
        print(f"[GPT-filter] ❌ 千问筛选失败: {exc}")
        # Fallback: accept all masks
        return {"accepted_masks": list(range(num_shown)), "rejected_masks": [], "reason": f"筛选失败，保留所有: {exc}"}


def _gpt_analyze_bboxes_only(image_bgr, targets, roi_note=""):
    """直接让 GPT 返回 bbox，不需要 SAM3 候选 mask。
    
    Args:
        image_bgr: 原始图片 (BGR)
        targets: 目标名称列表，如 ["番茄", "鸡蛋"]
        roi_note: ROI 区域说明
    
    Returns:
        dict: {
            "dish_name": "番茄炒蛋",
            "targets": [
                {
                    "name": "番茄",
                    "prompt": "tomato pieces",
                    "positive_points": [...],
                    "negative_points": [...],
                    "instance_boxes": [[x1, y1, x2, y2], ...]  # 归一化坐标
                }
            ]
        }
    """
    api_key, endpoint, model = _openai_connection()
    
    height, width = image_bgr.shape[:2]
    ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("图片编码失败")
    image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
    
    target_text = "、".join(targets)
    
    system_prompt = (
        "你是一个食物识别助手，负责分析图片中的食物类别和位置。"
        "请只返回严格 JSON 对象，不要 Markdown。"
        "最外层必须包含 dish_name 和 targets 数组。"
    )
    
    user_prompt = f"""
分析这张食物图片，完成以下任务：

## 任务 1：识别菜式名称
识别这道菜的名称，例如"番茄炒蛋"、"宫保鸡丁"等。

## 任务 2：为每个目标类别提供边界框
目标类别：{target_text}
{roi_note}

对于每个目标类别：
- 识别图片中所有属于该类别的独立实例
- 为每个实例提供一个边界框 (bounding box)
- 提供正样本点（实例内部的点）和负样本点（实例外部的点）
- **只识别在锅/容器内部的实例**，排除锅外的区域

## 坐标格式
所有坐标使用 0 到 1 的归一化格式：
- 边界框: [x1, y1, x2, y2]，左上角为原点
- 点坐标: [x, y]

## JSON 格式
{{
  "dish_name": "番茄炒蛋",
  "targets": [
    {{
      "name": "番茄",
      "prompt": "tomato pieces",
      "positive_points": [[0.15, 0.25], [0.35, 0.45]],
      "negative_points": [[0.55, 0.65]],
      "instance_boxes": [[0.10, 0.20, 0.20, 0.30], [0.30, 0.40, 0.40, 0.50]]
    }},
    {{
      "name": "鸡蛋",
      "prompt": "egg pieces",
      "positive_points": [[0.25, 0.35]],
      "negative_points": [[0.45, 0.55]],
      "instance_boxes": [[0.20, 0.30, 0.30, 0.40]]
    }}
  ]
}}

## 注意事项
- 每个类别通常有 1-5 个实例
- 边界框应该紧密包围实例
- 正样本点应该在实例内部
- 负样本点应该在实例外部（其他食材或背景）
- 只识别在锅/容器内部的实例
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    
    # 尝试多个模型
    model_candidates = list(dict.fromkeys([
        model,
        os.environ.get("QWEN_MODEL", ""),
        "qwen-vl-max",
        "qwen-vl-plus",
    ]))
    model_candidates = [m for m in model_candidates if m]
    
    result = None
    last_error = None
    for candidate in model_candidates:
        for use_response_format in (True, False):
            payload = {
                "model": candidate,
                "temperature": 0,
                "messages": messages,
            }
            if use_response_format:
                payload["response_format"] = {"type": "json_object"}
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    result = json.loads(response.read().decode(charset, errors="replace"))
                    data = _gpt_json_content(result)
                    print(f"[GPT-bbox] 模型 {candidate} 返回: dish_name={data.get('dish_name')}, targets={len(data.get('targets', []))}")
                    return data
            except urllib.error.HTTPError as exc:
                raw_detail = exc.read()
                charset = exc.headers.get_content_charset() if exc.headers else None
                detail = raw_detail.decode(charset or "utf-8", errors="replace")[:800]
                last_error = f"GPT 请求失败 HTTP {exc.code}: {detail}"
                if use_response_format and _qwen_response_format_failed(detail):
                    continue
                break
            except Exception as exc:
                last_error = f"GPT 请求异常：{exc}"
                break
        print(f"[GPT-bbox] 模型 {candidate} 失败，尝试下一个")
    
    if result is None:
        raise RuntimeError(last_error or "GPT 请求失败：没有可用模型")
    
    data = _gpt_json_content(result)
    print(f"[GPT-bbox] 返回数据 keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    
    # 标准化返回数据
    normalized = {"dish_name": "", "targets": []}
    normalized["dish_name"] = str(data.get("dish_name") or "").strip()
    
    raw_targets = data.get("targets", []) if isinstance(data, dict) else []
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict):
            continue
        target_name = str(raw_target.get("name") or "").strip()
        if not target_name:
            continue
        
        normalized_target = {
            "name": target_name,
            "prompt": str(raw_target.get("prompt") or _batch_model_prompt(target_name)).strip(),
            "positive_points": _normalized_points(raw_target.get("positive_points")),
            "negative_points": _normalized_points(raw_target.get("negative_points")),
            "instance_boxes": _normalized_boxes(raw_target.get("instance_boxes")),
        }
        normalized["targets"].append(normalized_target)
    
    return normalized


def _gpt_analyze_and_filter_masks(image_bgr, targets, masks_per_target, roi_note=""):
    """合并分析+筛选为一次 GPT 调用。
    
    Args:
        image_bgr: 原始图片 (BGR)
        targets: 目标名称列表，如 ["番茄", "鸡蛋"]
        masks_per_target: dict, {target_name: [(original_idx, mask_array), ...]}
        roi_note: ROI 区域说明
    
    Returns:
        dict: {
            "targets": [
                {
                    "name": "番茄",
                    "prompt": "tomato pieces",
                    "positive_points": [...],
                    "negative_points": [...],
                    "instance_boxes": [...],
                    "box": [...],
                    "accepted_mask_indices": [0, 2, 5]  # 接受的 mask 在 masks_per_target 中的索引
                }
            ]
        }
    """
    api_key, endpoint, model = _openai_connection()
    
    # 构建可视化图片：所有目标的 mask 都标注上
    height, width = image_bgr.shape[:2]
    vis_image = image_bgr.copy()
    
    # 为每个目标的 mask 分配颜色和编号
    mask_info = []  # [(target_name, mask_idx_in_target, global_mask_id, color)]
    global_mask_id = 0
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (128, 255, 0), (0, 128, 255), (255, 0, 128),
    ]
    
    pil_image = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    try:
        font = ImageFont.truetype("/Windows/Fonts/arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    
    target_mask_summary = []  # 用于 prompt 的 mask 汇总
    
    for target_name in targets:
        masks = masks_per_target.get(target_name, [])
        target_mask_summary.append(f"- {target_name}: {len(masks)} 个候选 mask")
        
        for local_idx, (orig_idx, mask) in enumerate(masks):
            color = colors[global_mask_id % len(colors)]
            mask_info.append((target_name, local_idx, global_mask_id, color))
            
            # 绘制 mask 半透明覆盖
            mask_rgba = np.zeros((height, width, 4), dtype=np.uint8)
            mask_rgba[mask] = [*color, 80]
            mask_pil = Image.fromarray(mask_rgba, "RGBA")
            overlay = Image.alpha_composite(overlay, mask_pil)
            draw = ImageDraw.Draw(overlay)
            
            # 绘制编号
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                # 绘制圆形背景和编号
                draw.ellipse([cx-15, cy-15, cx+15, cy+15], fill=(*color, 200))
                draw.text((cx-8, cy-10), str(global_mask_id), fill=(255, 255, 255, 255), font=font)
            
            global_mask_id += 1
    
    # 合并 overlay 到原图
    vis_image = cv2.cvtColor(np.array(Image.alpha_composite(pil_image.convert("RGBA"), overlay).convert("RGB")), cv2.COLOR_RGB2BGR)
    
    ok, encoded = cv2.imencode(".jpg", vis_image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("可视化图片编码失败")
    image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
    
    # 构建 mask 编号与目标的映射说明
    mask_mapping_lines = []
    for target_name, local_idx, global_id, color in mask_info:
        mask_mapping_lines.append(f"  编号 {global_id}: 属于 {target_name} 的候选 mask #{local_idx+1}")
    mask_mapping = "\n".join(mask_mapping_lines) if mask_mapping_lines else "  (无候选 mask)"
    
    target_text = "、".join(targets)
    
    system_prompt = (
        "你是一个图像分割初始化助手，同时负责分析食材和筛选候选区域。"
        "请只返回严格 JSON 对象，不要 Markdown。"
        "最外层必须是包含 targets 数组的对象。"
    )
    
    user_prompt = f"""
分析这张食物图片，完成两个任务：

## 任务 1：为每个目标类别提供分割提示
目标类别：{target_text}
{roi_note}

对每个类别：
- 返回 4 到 16 个正点（positive_points），必须落在目标食材内部
- 返回 2 到 8 个负点（negative_points），落在容易混淆的其他食材、盘子/锅、背景上
- 返回 prompt（简短英文视觉短语，如 tomato pieces、scrambled egg pieces）
- 返回 instance_boxes（每个正点对应的实例框）
- 返回 box（覆盖该类别所有实例的总框）

**重要**：先判断目标是一整块还是分散的多块：
- 一整块：4-6 个正点均匀分布
- 分散多块：每个可见块至少 1 个正点

## 任务 2：筛选候选 mask
图中标注了 {global_mask_id} 个候选 mask（用不同颜色和数字标记）。

Mask 编号与目标的映射：
{mask_mapping}

**重要**：对每个目标类别，判断哪些编号的 mask 真正属于该类别：
- 只保留真正属于目标类别的 mask
- 排除属于其他食材、盘子、锅、背景、汤汁、阴影的 mask
- **只接受在锅/容器内部的 mask**，排除锅外的区域
- 如果 mask 边界不清晰或不确定，也排除
- 返回 accepted_mask_indices（接受的 mask 编号列表）

**注意**：不要接受所有 mask！只接受真正属于目标类别且在锅内的 mask。通常一个类别应该接受 3-10 个 mask，而不是全部 30 个。

## 坐标格式
所有坐标使用 0 到 1 的归一化格式 [x, y]，原点在左上角。

## JSON 格式
{{
  "targets": [
    {{
      "name": "番茄",
      "prompt": "tomato pieces",
      "positive_points": [[0.1, 0.2], [0.3, 0.4]],
      "negative_points": [[0.5, 0.6]],
      "instance_boxes": [[0.08, 0.18, 0.15, 0.25], [0.28, 0.38, 0.35, 0.45]],
      "box": [0.05, 0.15, 0.40, 0.50],
      "accepted_mask_indices": [0, 2, 5]
    }}
  ]
}}
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    
    # 模型选择逻辑（与 _gpt_analyze_first_image 相同）
    model_candidates = [model]
    if "aliyuncs.com" in endpoint or "dashscope" in endpoint:
        model_candidates.extend([
            "qwen-vl-max-latest",
            "qwen-vl-plus-latest",
            "qwen-vl-max",
            "qwen-vl-plus",
        ])
    model_candidates = list(dict.fromkeys([item for item in model_candidates if item]))
    
    result = None
    last_error = None
    for candidate in model_candidates:
        for use_response_format in (True, False):
            payload = {
                "model": candidate,
                "temperature": 0,
                "messages": messages,
            }
            if use_response_format:
                payload["response_format"] = {"type": "json_object"}
            request = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    result = json.loads(response.read().decode(charset, errors="replace"))
                    if isinstance(result, dict):
                        result["_model_used"] = candidate
                    break
            except urllib.error.HTTPError as exc:
                raw_detail = exc.read()
                charset = exc.headers.get_content_charset() if exc.headers else None
                detail = raw_detail.decode(charset or "utf-8", errors="replace")[:800]
                last_error = f"GPT 请求失败 HTTP {exc.code}: {detail}"
                if "model_not_found" in detail or "Model not exist" in detail or "access_denied" in detail or "Access denied" in detail:
                    print(f"[GPT-combined] 模型不可用，尝试下一个: {candidate}")
                    break
                if use_response_format and _qwen_response_format_failed(detail):
                    print("[GPT-combined] JSON 结构化输出中断，改用普通提示重试")
                    continue
                raise RuntimeError(last_error) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"GPT 网络请求失败: {exc.reason}") from exc
            break
        if result is not None:
            break
    
    if result is None:
        raise RuntimeError(last_error or "GPT 请求失败：没有可用模型")
    
    data = _gpt_json_content(result)
    print(f"[GPT-combined] 返回数据 keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    
    # 标准化返回数据
    normalized = {"targets": []}
    raw_targets = data.get("targets", []) if isinstance(data, dict) else []
    
    for raw_target in raw_targets:
        if not isinstance(raw_target, dict):
            continue
        target_name = str(raw_target.get("name") or "").strip()
        if not target_name:
            continue
        
        normalized_target = {
            "name": target_name,
            "prompt": str(raw_target.get("prompt") or _batch_model_prompt(target_name)).strip(),
            "positive_points": _normalized_points(raw_target.get("positive_points")),
            "negative_points": _normalized_points(raw_target.get("negative_points")),
            "instance_boxes": _normalized_boxes(raw_target.get("instance_boxes")),
            "box": raw_target.get("box"),
            "accepted_mask_indices": [
                int(idx) for idx in (raw_target.get("accepted_mask_indices") or [])
                if isinstance(idx, (int, float))
            ],
        }
        normalized["targets"].append(normalized_target)
        print(f"[GPT-combined] {target_name}: 正点={len(normalized_target['positive_points'])}, "
              f"负点={len(normalized_target['negative_points'])}, "
              f"接受mask={normalized_target['accepted_mask_indices']}")
    
    if isinstance(result, dict) and result.get("_model_used"):
        normalized["_model_used"] = result["_model_used"]
    
    return normalized


def _sam3_detect_candidate_masks(processor, image_rgb, target_name):
    """用 SAM3 检测候选 mask，不调用 GPT 筛选。
    
    Args:
        processor: SAM3 处理器
        image_rgb: RGB 图片
        target_name: 目标名称（如 "番茄"）
    
    Returns:
        list: mask 数组列表
    """
    name = str(target_name or "object").strip()
    prompt = _batch_model_prompt(name)
    print(f"[SAM3-detect] 检测目标: {name}, prompt={prompt!r}")
    
    height, width = image_rgb.shape[:2]
    
    # 使用低阈值检测所有候选 mask
    print(f"[SAM3-detect] 使用低阈值一次性检测所有候选 mask")
    processor.set_confidence_threshold(0.02)
    state = processor.set_image(Image.fromarray(image_rgb))
    output = processor.set_text_prompt(prompt=prompt, state=state)
    
    masks = _sam3_masks_from_output(output)
    print(f"[SAM3-detect] 检测到 {len(masks)} 个候选 mask")
    
    if not masks:
        print(f"[SAM3-detect] ⚠️ {name}: 未检测到任何 mask")
        return []
    
    # 筛选面积最大的 30 个 mask
    max_candidate_masks = 30
    if len(masks) > max_candidate_masks:
        print(f"[SAM3-detect] 筛选面积最大的 {max_candidate_masks} 个 mask")
        mask_areas = [(idx, int(mask.sum())) for idx, mask in enumerate(masks)]
        mask_areas.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in mask_areas[:max_candidate_masks]]
        top_indices.sort()
        masks = [masks[idx] for idx in top_indices]
        print(f"[SAM3-detect] 筛选后保留 {len(masks)} 个 mask")
    
    # 打印每个 mask 的信息
    for idx, mask in enumerate(masks):
        mask_area = int(mask.sum())
        bbox = _batch_mask_bbox(mask)
        print(f"[SAM3-detect] mask {idx}: 面积={mask_area}, bbox={bbox}")
    
    return masks


def _sam3_seed_masks_for_target(processor, image_rgb, target):
    """Ground one GPT-described category on the first image with SAM3."""
    name = str(target.get("name") or "object").strip()
    prompt = str(target.get("prompt") or _batch_model_prompt(name)).strip()
    positive = _normalized_points(target.get("positive_points"))
    negative = _normalized_points(target.get("negative_points"))
    instance_boxes = _normalized_boxes(target.get("instance_boxes"))
    total_box = _normalized_box(target.get("box"))
    fallback_boxes = instance_boxes or ([total_box] if total_box is not None else [])
    print(f"[SAM3-seed] 处理目标: {name}, prompt={prompt!r}, 正点={len(positive)}, 负点={len(negative)}")
    if positive:
        print(f"[SAM3-seed] 正点坐标: {positive}")
    if fallback_boxes:
        print(f"[SAM3-seed] 千问返回框: instance_boxes={len(instance_boxes)}, total_box={total_box}")
    
    height, width = image_rgb.shape[:2]

    def masks_from_boxes():
        box_masks = []
        if not fallback_boxes:
            return box_masks
        for box_index, box in enumerate(fallback_boxes):
            box_state = processor.set_image(Image.fromarray(image_rgb))
            if prompt:
                processor.set_text_prompt(prompt=prompt, state=box_state)
            box_prompt = _box_xyxy_to_sam3_prompt(box)
            box_output = processor.add_geometric_prompt(box_prompt, True, box_state)
            found = _sam3_masks_from_output(box_output)
            if found:
                print(f"[SAM3-seed] 框兜底 {box_index}: box={box}, 生成 {len(found)} 个 mask")
                box_masks.extend(found)
            else:
                print(f"[SAM3-seed] 框兜底 {box_index}: box={box}, 未生成 mask")
        return _dedupe_batch_masks(box_masks)
    
    # 改进方案：一次性检测所有 mask，然后让千问筛选
    print(f"[SAM3-seed] 使用低阈值一次性检测所有候选 mask")
    processor.set_confidence_threshold(0.02)
    state = processor.set_image(Image.fromarray(image_rgb))
    output = processor.set_text_prompt(prompt=prompt, state=state)
    
    # 如果有正点，添加正点提示
    for point in positive:
        output = processor.add_point_prompt(point, True, state)
    for point in negative:
        output = processor.add_point_prompt(point, False, state)
    
    masks = _sam3_masks_from_output(output)
    print(f"[SAM3-seed] 检测到 {len(masks)} 个候选 mask")
    
    if not masks:
        print(f"[SAM3-seed] ⚠️ {name}: 文本/点提示未检测到 mask，尝试千问框兜底")
        fallback_masks = masks_from_boxes()
        if fallback_masks:
            print(f"[SAM3-seed] ✅ {name}: 框兜底生成 {len(fallback_masks)} 个 mask")
            return fallback_masks
        print(f"[SAM3-seed] ❌ {name}: SAM3 未检测到任何 mask")
        return []
    
    # 筛选面积最大的30个mask（避免处理太多）
    max_candidate_masks = 30
    if len(masks) > max_candidate_masks:
        print(f"[SAM3-seed] 筛选面积最大的 {max_candidate_masks} 个 mask")
        # 计算每个mask的面积
        mask_areas = [(idx, int(mask.sum())) for idx, mask in enumerate(masks)]
        # 按面积排序，取最大的30个
        mask_areas.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in mask_areas[:max_candidate_masks]]
        top_indices.sort()  # 重新排序保持原始顺序
        masks = [masks[idx] for idx in top_indices]
        print(f"[SAM3-seed] 筛选后保留 {len(masks)} 个 mask")
    
    # 打印每个 mask 的信息（只打印筛选后的）
    for idx, mask in enumerate(masks):
        mask_area = int(mask.sum())
        bbox = _batch_mask_bbox(mask)
        print(f"[SAM3-seed] mask {idx}: 面积={mask_area}, bbox={bbox}")
    
    # HSV 颜色预过滤已禁用 - 直接让 GPT 筛选所有 mask
    # （HSV 过滤太严格，会把有效 mask 过滤掉）
    print(f"[SAM3-seed] 跳过 HSV 预过滤，直接发送 {len(masks)} 个 mask 给千问筛选")
    
    # 让千问筛选哪些 mask 是目标
    print(f"[SAM3-seed] 请求千问筛选 mask...")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    filter_result = _gpt_filter_masks(image_bgr, masks, name, prompt)
    
    # 确保 filter_result 是字典
    if not isinstance(filter_result, dict):
        print(f"[SAM3-seed] ⚠️ 千问返回格式错误: {type(filter_result)}，使用 fallback")
        filter_result = {"accepted_masks": list(range(len(masks))), "rejected_masks": [], "reason": "返回格式错误"}
    
    accepted_indices = filter_result.get("accepted_masks", [])
    if not accepted_indices:
        print(f"[SAM3-seed] ⚠️ {name}: 千问拒绝了所有 mask")
        # Fallback: 使用被正点击中的 mask
        selected = []
        for idx, mask in enumerate(masks):
            for point in positive:
                px = min(width - 1, max(0, int(point[0] * width)))
                py = min(height - 1, max(0, int(point[1] * height)))
                if mask[py, px]:
                    selected.append(mask)
                    break
        if selected:
            print(f"[SAM3-seed] 使用被正点击中的 {len(selected)} 个 mask 作为 fallback")
            return _dedupe_batch_masks(selected)
        fallback_masks = masks_from_boxes()
        if fallback_masks:
            print(f"[SAM3-seed] 使用千问框兜底的 {len(fallback_masks)} 个 mask")
            return fallback_masks
        return []
    
    # 保留千问确认的 mask（不去重 - GPT 已确认它们是不同物体）
    selected_masks = [masks[idx] for idx in accepted_indices if idx < len(masks)]
    # 只确保 bool 类型，不做去重
    result = [ensure_bool_mask(m) for m in selected_masks if ensure_bool_mask(m) is not None]
    result = [m for m in result if int(m.sum()) > 0]
    print(f"[SAM3-seed] ✅ {name}: 千问筛选后保留 {len(result)} 个 mask (总共 {len(masks)} 个)")
    return result


def _rect_mask_from_pixel_box(box, height, width):
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in box]
    x1, x2 = sorted((max(0, min(width, int(round(x1)))), max(0, min(width, int(round(x2))))))
    y1, y2 = sorted((max(0, min(height, int(round(y1)))), max(0, min(height, int(round(y2))))))
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _gpt_roi_note(container_concept, prompt_box, width, height):
    parts = []
    if container_concept:
        parts.append(f"只在“{container_concept}”这个盛放/烹饪器具区域内寻找目标食材。")
    if prompt_box and len(prompt_box) == 4:
        x1, y1, x2, y2 = [float(value) for value in prompt_box]
        norm_box = [
            max(0.0, min(1.0, x1 / max(width, 1))),
            max(0.0, min(1.0, y1 / max(height, 1))),
            max(0.0, min(1.0, x2 / max(width, 1))),
            max(0.0, min(1.0, y2 / max(height, 1))),
        ]
        parts.append(
            "用户手动画了限制框，请只在该框内找目标。"
            f"限制框归一化坐标为 [{norm_box[0]:.4f}, {norm_box[1]:.4f}, {norm_box[2]:.4f}, {norm_box[3]:.4f}]。"
        )
    if parts:
        return "限制条件：" + " ".join(parts)
    return ""


def _sam3_container_roi_mask(processor, image_rgb, container_concept, prompt_points,
                             prompt_labels, prompt_box):
    height, width = image_rgb.shape[:2]
    if prompt_box and len(prompt_box) == 4:
        return _rect_mask_from_pixel_box(prompt_box, height, width), "manual_box"
    if not container_concept:
        return None, None
    state = processor.set_image(Image.fromarray(image_rgb))
    output = processor.set_text_prompt(
        prompt=_batch_model_prompt(container_concept), state=state
    )
    masks = _sam3_masks_from_output(output)
    if not masks:
        processor.set_confidence_threshold(0.15)
        output = processor.set_confidence_threshold(0.15, state=state)
        masks = _sam3_masks_from_output(output)
    roi_mask = _choose_container_mask(
        masks, width, height,
        prompt_points=prompt_points,
        prompt_labels=prompt_labels,
        prompt_box=prompt_box,
    )
    if roi_mask is not None:
        # 确保 roi_mask 是 bool 类型
        roi_mask = ensure_bool_mask(roi_mask)
        # 计算 ROI 区域的 bbox
        ys, xs = np.where(roi_mask)
        if len(xs) > 0:
            roi_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            print(f"[ROI-debug] 容器 mask bbox={roi_bbox}, 图片尺寸=({height}, {width}), dtype={roi_mask.dtype}")
            # 检测 ROI 是否是环形（容器边缘）：检查 bbox 中心是否在 mask 内
            center_y = (roi_bbox[1] + roi_bbox[3]) // 2
            center_x = (roi_bbox[0] + roi_bbox[2]) // 2
            center_in_mask = roi_mask[center_y, center_x]
            print(f"[ROI-debug] bbox 中心=({center_x}, {center_y}), 在 mask 内={center_in_mask}")
            if not center_in_mask:
                # 可能是环形（只覆盖容器边缘），但需要验证
                # 如果反转后覆盖 >80% 图片，说明不是真正的环形，而是只检测到一边
                inverted_mask = ~roi_mask
                inverted_area = int(inverted_mask.sum())
                total_area = height * width
                inverted_ratio = inverted_area / total_area
                print(f"[ROI-debug] 反转后面积比例: {inverted_ratio:.2%}")
                if inverted_ratio < 0.80:
                    # 反转后覆盖 <80%，是真正的环形，使用反转
                    roi_mask = inverted_mask
                    print(f"[ROI-debug] 检测到环形 ROI，已反转（边缘→内部）")
                    # 重新计算 bbox
                    ys2, xs2 = np.where(roi_mask)
                    if len(xs2) > 0:
                        roi_bbox2 = (int(xs2.min()), int(ys2.min()), int(xs2.max()), int(ys2.max()))
                        print(f"[ROI-debug] 反转后 bbox={roi_bbox2}, 面积={int(roi_mask.sum())}")
                else:
                    # 反转后覆盖 >80%，说明只检测到一边，不反转
                    print(f"[ROI-debug] ⚠️ 反转后覆盖过大 ({inverted_ratio:.1%})，保持原 mask（可能只检测到容器一边）")
    return roi_mask, "container_text" if roi_mask is not None else None


def _clean_target_names(targets):
    cleaned = []
    seen = set()
    for item in targets or []:
        for part in str(item or "").replace("，", ",").split(","):
            name = part.strip()
            key = name.lower()
            if name and key not in seen:
                cleaned.append(name)
                seen.add(key)
    return cleaned


def _sync_gpt_seed_to_workbench(targets):
    """Expose GPT+SAM3 first-frame masks as editable workbench instances."""
    if STATE["image_rgb"] is None:
        return []
    # Replace the previous GPT seed set on repeated initialization, while
    # preserving manually created/edited instances from other sources.
    stale_ids = [
        iid for iid, inst in STATE["instances"].items()
        if inst.get("source") == "gpt_sam3_sam2_seed"
    ]
    for iid in stale_ids:
        STATE["instances"].pop(iid, None)

    added = []
    next_iid = max([0, *[int(iid) for iid in STATE["instances"].keys()]]) + 1
    for _object_id, target in sorted(targets.items()):
        mask = ensure_bool_mask(target.get("seed_mask"))
        if mask is None or not mask.any():
            continue
        iid = next_iid
        next_iid += 1
        instance_name = target.get("name", "object")
        print(f"[workbench] 添加实例 {iid}: name={instance_name!r}, object_id={_object_id}")
        positive_points = [
                [float(point[0]) * STATE["w"], float(point[1]) * STATE["h"]]
                for point in target.get("points", [])
            ]
        negative_points = [
            [float(point[0]) * STATE["w"], float(point[1]) * STATE["h"]]
            for point in target.get("negative_points", [])
        ]
        manual_points = [
            [float(point[0]), float(point[1])]
            for point in target.get("manual_points", [])
        ]
        STATE["instances"][iid] = {
            "points": positive_points + negative_points + manual_points,
            "labels": (
                [1] * len(positive_points)
                + [0] * len(negative_points)
                + list(target.get("manual_labels", []))
            ),
            "box": None,
            "mask": mask,
            "amg": False,
            "source": "gpt_sam3_sam2_seed",
            "name": instance_name,
            "confidence": None,
            "task": "segmentation",
            "gpt_prompt": target.get("gpt_prompt"),
        }
        added.append(iid)
    STATE["next_id"] = max(next_iid, STATE.get("next_id", 1))
    STATE["current_id"] = added[-1] if added else STATE.get("current_id")
    return added


def _refine_sam2_batch_target(req):
    """Apply one positive/negative first-frame correction to an existing memory."""
    if not req.points:
        raise RuntimeError("请先点击首图添加正点或负点")
    with SAM2_BATCH_LOCK:
        if SAM2_BATCH_STATE["inference_state"] is None:
            raise RuntimeError("请先完成 GPT + SAM3 首图初始化")
        if SAM2_BATCH_STATE["status"] == "running":
            raise RuntimeError("SAM2 正在传播，请等待完成后再修正首图")
        target = SAM2_BATCH_STATE.get("targets", {}).get(req.target_id)
        if not target:
            raise RuntimeError("找不到要修正的目标")
        predictor_model = _tracking_predictor()
        points = np.asarray(req.points, dtype=np.float32)
        labels = np.asarray(
            req.labels or [1] * len(req.points), dtype=np.int32
        )
        if len(labels) != len(points):
            raise RuntimeError("正负点数量不一致")
        frame_idx, object_ids, masks = predictor_model.add_new_points_or_box(
            SAM2_BATCH_STATE["inference_state"],
            0,
            int(req.target_id),
            points=points,
            labels=labels,
            box=None,
        )
        mask_values = _sam2_batch_masks(masks)
        selected_mask = None
        for index, object_id in enumerate(object_ids):
            if int(object_id) == int(req.target_id) and index < len(mask_values):
                selected_mask = mask_values[index]
                break
        if selected_mask is None or not selected_mask.any():
            raise RuntimeError("该修正点没有生成有效 mask")

        target["seed_mask"] = selected_mask
        target["manual_points"] = (
            target.get("manual_points", []) + req.points
        )
        target["manual_labels"] = (
            target.get("manual_labels", []) + [int(value) for value in labels.tolist()]
        )
        target["name"] = req.name or target.get("name", "object")
        workbench_ids = _sync_gpt_seed_to_workbench(SAM2_BATCH_STATE["targets"])
        frame = cv2.imread(
            os.path.join(SAM2_BATCH_STATE["frames_dir"], "000000.jpg")
        )
        if frame is None:
            raise RuntimeError("无法读取首图")
        preview = _tracking_mask_preview(
            frame,
            np.asarray([selected_mask], dtype=np.uint8),
            [int(req.target_id)],
        )
        SAM2_BATCH_STATE["prompted_target"] = True
        SAM2_BATCH_STATE["status"] = "prompted"
        SAM2_BATCH_STATE["message"] = (
            f"目标「{target['name']}」已应用 "
            f"{len(req.points)} 个人工{'负' if any(int(v) == 0 for v in labels) else '正'}点"
        )
        _save_sam2_click_cache(
            SAM2_BATCH_STATE.get("container_points", []),
            SAM2_BATCH_STATE.get("container_labels", []),
            SAM2_BATCH_STATE.get("container_box"),
            SAM2_BATCH_STATE.get("targets", {}),
            SAM2_BATCH_STATE["width"],
            SAM2_BATCH_STATE["height"],
        )
        return {
            "status": "ok",
            "target_id": int(req.target_id),
            "area": int(selected_mask.sum()),
            "preview": preview,
            "rendered": render_composite_to_b64(highlight_id=None),
            "workbench_instance_ids": workbench_ids,
            "message": SAM2_BATCH_STATE["message"],
        }


def _gpt_seed_sam2_batch(req):
    """Initialize SAM2 objects from GPT points refined by SAM3 masks."""
    target_names = req.targets
    target_names = _clean_target_names(target_names)
    if not target_names:
        raise RuntimeError("请先填写要识别的食材类别，例如：芥兰,牛肉")
    log_lines = [
        f"[GPT] 首图分析目标：{'、'.join(target_names)}",
        "[GPT] 正在请求视觉模型生成每类多点提示...",
    ]
    with SAM2_BATCH_LOCK:
        if not SAM2_BATCH_STATE["frames_dir"]:
            raise RuntimeError("请先选择批量图片")
        frame_path = os.path.join(SAM2_BATCH_STATE["frames_dir"], "000000.jpg")
        width = SAM2_BATCH_STATE["width"]
        height = SAM2_BATCH_STATE["height"]
        fallback_container_points = SAM2_BATCH_STATE.get("container_points", [])
        fallback_container_labels = SAM2_BATCH_STATE.get("container_labels", [])
        fallback_container_box = SAM2_BATCH_STATE.get("container_box")
    frame = cv2.imread(frame_path)
    if frame is None:
        raise RuntimeError("无法读取首图")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # SAM3 is only an optional first-frame mask generator.  If it cannot be
    # loaded (or does not find a mask), SAM2 can still be initialized directly
    # from the points returned by GPT/user hints.
    processor = None
    try:
        _, processor = _sam3_load_image_model()
        processor.set_confidence_threshold(0.35)
    except Exception as exc:
        log_lines.append(f"[SAM3] 首图模型不可用，改用 SAM2 点提示兜底：{exc}")
        print(f"[gpt-seed] SAM3 不可用，使用 SAM2 点提示兜底: {exc}")
    container = (req.container_concept or "").strip()
    prompt_points = req.prompt_points or fallback_container_points
    prompt_labels = req.prompt_labels or fallback_container_labels
    prompt_box = req.prompt_box or fallback_container_box
    roi_mask, roi_source = (None, None)
    if processor is not None:
        roi_mask, roi_source = _sam3_container_roi_mask(
            processor, rgb, container, prompt_points, prompt_labels, prompt_box
        )
    if roi_mask is not None:
        log_lines.append(
            f"[ROI] 限制区域来源={roi_source}，面积={int(roi_mask.sum())}"
        )
        print(f"[ROI] 限制区域：来源={roi_source}, 面积={int(roi_mask.sum())}, 图片尺寸={rgb.shape[:2]}")
    elif container or req.prompt_box:
        log_lines.append("[ROI] 未能生成限制区域，将仅依赖 GPT 类别点")
        print(f"[ROI] 未能生成限制区域：container={container!r}, prompt_box={req.prompt_box}")

    roi_note = _gpt_roi_note(container, prompt_box, width, height)
    
    # ===== 简化流程：直接让 GPT 返回 bbox，不需要 SAM3 =====
    log_lines.append("[GPT-bbox] 直接请求 GPT 返回边界框，跳过 SAM3 候选检测...")
    analysis = _gpt_analyze_bboxes_only(frame, target_names, roi_note=roi_note)
    gpt_targets = analysis.get("targets", [])
    dish_name = analysis.get("dish_name", "")
    
    if dish_name:
        log_lines.append(f"[GPT-bbox] 识别菜式：{dish_name}")
    
    if not gpt_targets:
        raise RuntimeError("GPT 没有找到可初始化的目标")
    log_lines.append(f"[GPT-bbox] 返回 {len(gpt_targets)} 个目标")
    
    # 清空 masks_per_target，因为新流程不使用 SAM3 mask
    masks_per_target = {}
    
    # ===== 后续处理：与旧流程相同 =====

    predictor_model = _tracking_predictor()
    with SAM2_BATCH_LOCK:
        if SAM2_BATCH_STATE["inference_state"] is None:
            SAM2_BATCH_STATE["inference_state"] = predictor_model.init_state(
                SAM2_BATCH_STATE["frames_dir"], offload_video_to_cpu=True
            )
        inference_state = SAM2_BATCH_STATE["inference_state"]

    targets = {}
    preview_masks = []
    preview_ids = []
    next_id = 2
    requested_keys = {target.lower() for target in target_names}
    print(f"[GPT] 用户请求的目标: {target_names}")
    print(f"[GPT] 千问返回的目标数量: {len(gpt_targets)}")
    for target_index, target in enumerate(gpt_targets):
        name = str(target.get("name") or "").strip()
        print(f"[GPT] 千问返回的目标 {target_index}: name={name!r}, prompt={target.get('prompt')!r}")
        if target_index < len(target_names) and (
            not name or name.lower() not in requested_keys
        ):
            # Vision models sometimes translate class names in the JSON name.
            # Keep the user-facing category stable and leave the English words
            # in gpt_prompt for SAM3.
            name = target_names[target_index]
            target = dict(target)
            target["name"] = name
            print(f"[GPT] 使用用户输入的名称: {name}")
        if not name or name in {item["name"] for item in targets.values()}:
            print(f"[GPT] 跳过空名称或重复名称: {name}")
            continue
        if name.lower() not in requested_keys:
            log_lines.append(f"[GPT] 跳过未请求类别：{name}")
            print(f"[GPT] 跳过未请求类别: {name}")
            continue
        target, manual_pos, manual_neg = _merge_manual_target_hints(
            target, req.target_hints, width, height
        )
        if manual_pos or manual_neg:
            log_lines.append(
                f"[人工修正] {name}: 合并正点={manual_pos}, 负点={manual_neg}"
            )
        positive = _normalized_points(target.get("positive_points"))
        negative = _normalized_points(target.get("negative_points"))
        log_lines.append(
            f"[GPT] {name}: prompt={target.get('prompt') or _batch_model_prompt(name)}, "
            f"正点={len(positive)}, 负点={len(negative)}"
        )
        
        # ===== 简化流程：使用 bbox 初始化 SAM2 =====
        instance_boxes = target.get("instance_boxes", [])
        
        if not instance_boxes:
            log_lines.append(f"[GPT-bbox] {name}: 没有返回边界框，跳过")
            print(f"[GPT-bbox] {name}: 没有返回边界框")
            continue
        
        print(f"[GPT-bbox] {name}: 获得 {len(instance_boxes)} 个边界框")
        
        # 对每个 bbox 创建独立实例
        for box_idx, box in enumerate(instance_boxes):
            if len(box) != 4:
                log_lines.append(f"[GPT-bbox] {name}#{box_idx+1}: 无效的 bbox 格式，跳过")
                continue
            
            # 转换归一化坐标为像素坐标 [x1, y1, x2, y2]
            x1, y1, x2, y2 = [float(v) for v in box]
            pixel_box = np.asarray([
                x1 * width, y1 * height, x2 * width, y2 * height
            ], dtype=np.float32)
            
            print(f"[GPT-bbox] {name}#{box_idx+1}: box={box}, pixel_box={pixel_box}")
            
            try:
                # 使用 bbox 初始化 SAM2
                _, box_ids, box_masks = predictor_model.add_new_points_or_box(
                    inference_state, 0, next_id,
                    points=None, labels=None, box=pixel_box,
                )
                
                box_values = _sam2_batch_masks(box_masks)
                if not box_values:
                    log_lines.append(f"[SAM2] {name}#{box_idx+1}: bbox 未生成有效 mask")
                    continue
                
                mask = box_values[0]
                
                # 保存目标信息
                targets[next_id] = {
                    "name": name,
                    "points": positive,
                    "labels": [1] * len(positive),
                    "negative_points": negative,
                    "gpt_prompt": target.get("prompt") or _batch_model_prompt(name),
                    "seed_mask": mask,
                    "seed_box": pixel_box.tolist(),
                }
                
                log_lines.append(
                    f"[SAM2] {name}#{box_idx+1}: bbox 初始化成功，"
                    f"面积={int(mask.sum())}，object_id={next_id}"
                )
                print(f"[GPT-bbox] {name}#{box_idx+1}: ✅ 已添加到 targets，object_id={next_id}")
                
                preview_masks.append(mask)
                preview_ids.append(next_id)
                next_id += 1
                
            except Exception as exc:
                log_lines.append(f"[SAM2] {name}#{box_idx+1}: bbox 初始化失败：{exc}")
                print(f"[GPT-bbox] {name}#{box_idx+1}: ❌ bbox 初始化失败：{exc}")
                continue
        
        # 跳过后续的 mask 处理逻辑
        continue

    if not targets:
        raise RuntimeError("GPT 没有返回有效的边界框来初始化 SAM2")
    log_lines.append(f"[SAM2] 首图 memory 初始化完成：{len(targets)} 个目标")
    workbench_ids = _sync_gpt_seed_to_workbench(targets)
    if workbench_ids:
        log_lines.append(
            f"[工作区] 已添加首图实例：{', '.join(str(iid) for iid in workbench_ids)}"
        )
    preview = _tracking_mask_preview(
        frame, np.asarray(preview_masks, dtype=np.uint8), preview_ids
    )
    with SAM2_BATCH_LOCK:
        SAM2_BATCH_STATE.update({
            "prompted_container": True,
            "prompted_target": True,
            "targets": targets,
            "seed_method": "gpt_bbox",
            "gpt_analysis": analysis,
            "gpt_target_names": target_names,
            "roi_source": roi_source,
            "container_concept": container,
            "status": "prompted",
            "message": f"GPT bbox 首图初始化完成，已建立 {len(targets)} 个目标 memory",
            "error": None,
        })
    cache_targets = {
        object_id: {
            "name": target["name"],
            "points": target["points"],
            "labels": target["labels"],
            "negative_points": target.get("negative_points", []),
            "gpt_prompt": target["gpt_prompt"],
        }
        for object_id, target in targets.items()
    }
    _save_sam2_click_cache([], [], None, cache_targets, width, height)
    return {
        "status": "ok",
        "message": SAM2_BATCH_STATE["message"],
        "targets": {
            str(object_id): {
                key: value for key, value in target.items() if key != "seed_mask"
            }
            for object_id, target in targets.items()
        },
        "analysis": analysis,
        "preview": preview,
        "rendered": render_composite_to_b64(highlight_id=None),
        "workbench_instance_ids": workbench_ids,
        "log": "\n".join(log_lines),
    }


@app.post("/sam2-batch-tracking/gpt-initialize")
def sam2_batch_tracking_gpt_initialize(req: Sam2BatchGptInitRequest):
    try:
        return _gpt_seed_sam2_batch(req)
    except Exception as exc:
        traceback.print_exc()
        with SAM2_BATCH_LOCK:
            SAM2_BATCH_STATE.update({"status": "error", "error": str(exc), "message": str(exc)})
        return {"status": "error", "msg": str(exc)}


@app.post("/sam2-batch-tracking/refine")
def sam2_batch_tracking_refine(req: Sam2BatchRefineRequest):
    try:
        return _refine_sam2_batch_target(req)
    except Exception as exc:
        traceback.print_exc()
        return {"status": "error", "msg": str(exc)}


@app.post("/sam2-batch-tracking/upload")
async def sam2_batch_tracking_upload(files: List[UploadFile] = File(...)):
    if not files:
        return {"status": "error", "msg": "没有上传图片"}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = os.path.join(OUTPUT_ROOT, "sam2_batch_tracking", stamp)
    frames_dir = os.path.join(root, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    names = []
    width = height = 0
    for index, upload in enumerate(files):
        raw = await upload.read()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[sam2-batch] 跳过无法解码的图片: {upload.filename or index}")
            continue
        if not width:
            height, width = image.shape[:2]
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        # Use a contiguous frame index.  The previous code used the upload
        # index, so one corrupt image left a gap and caused SAM2 propagation
        # results to be matched with the wrong file (or fail to load).
        frame_name = f"{len(names):06d}.jpg"
        cv2.imwrite(os.path.join(frames_dir, frame_name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        names.append(upload.filename or f"image_{index:03d}.jpg")
    if not names:
        return {"status": "error", "msg": "没有可解码的图片"}
    
    # 检查是否有缓存的点击点
    cached_clicks = _load_sam2_click_cache()
    has_cached_clicks = False
    if cached_clicks:
        # 检查图片尺寸是否匹配
        if cached_clicks.get("width") == width and cached_clicks.get("height") == height:
            has_cached_clicks = True
            print(f"[sam2-batch] 发现缓存点击点: container={len(cached_clicks.get('container_points', []))}, "
                  f"target={len(cached_clicks.get('target_points', []))}")
        else:
            print(f"[sam2-batch] 缓存尺寸不匹配: 缓存={cached_clicks.get('width')}x{cached_clicks.get('height')}, "
                  f"当前={width}x{height}")
    
    with SAM2_BATCH_LOCK:
        SAM2_BATCH_STATE.update({
            "root": root, "frames_dir": frames_dir, "files": names,
            "width": width, "height": height, "frame_count": len(names),
            "inference_state": None, "prompted_container": False,
            "prompted_target": False, "status": "uploaded", "progress": 0,
            "message": "请在第一张图依次点击锅和目标", "error": None, "batch_id": None,
            "targets": {}, "container_points": [], "container_labels": [],
            "container_box": None, "seed_method": None, "gpt_analysis": None,
            "cached_clicks": cached_clicks if has_cached_clicks else None,
        })
    first = cv2.imread(os.path.join(frames_dir, "000000.jpg"))
    if first is not None:
        STATE.update({
            "image_bgr": first, "image_rgb": cv2.cvtColor(first, cv2.COLOR_BGR2RGB),
            "h": height, "w": width, "instances": {}, "next_id": 1,
            "current_id": None, "predictor_ready": False, "output_dir": root,
        })
    return {
        "status": "ok", "total": len(names), "width": width, "height": height,
        "has_cached_clicks": has_cached_clicks,
        "cached_clicks": cached_clicks if has_cached_clicks else None,
    }


@app.post("/sam2-batch-tracking/prompt")
def sam2_batch_tracking_prompt(req: Sam2BatchPromptRequest):
    stage = (req.stage or "").strip().lower()
    if stage not in {"container", "target"}:
        return {"status": "error", "msg": "无效的提示阶段"}
    if not req.points and not req.box:
        return {"status": "error", "msg": "请至少点击一个正点或框选区域"}
    with SAM2_BATCH_LOCK:
        if not SAM2_BATCH_STATE["frames_dir"]:
            return {"status": "error", "msg": "请先选择批量图片"}
        if stage == "target" and not SAM2_BATCH_STATE["prompted_container"]:
            return {"status": "error", "msg": "请先点击锅，再点击目标"}
        try:
            predictor_model = _tracking_predictor()
            if SAM2_BATCH_STATE["inference_state"] is None:
                SAM2_BATCH_STATE["inference_state"] = predictor_model.init_state(
                    SAM2_BATCH_STATE["frames_dir"], offload_video_to_cpu=True
                )
            
            # 确定 object_id
            if stage == "container":
                object_id = 1
            else:
                # 对于目标，使用 target_id 或自动分配
                if req.target_id is not None:
                    object_id = req.target_id
                else:
                    # 自动分配下一个可用的 target_id
                    existing_targets = SAM2_BATCH_STATE.get("targets", {})
                    if existing_targets:
                        object_id = max(existing_targets.keys()) + 1
                    else:
                        object_id = 2
            
            # 处理框选（用于限制区域）
            if req.box and stage == "container":
                # 框选转换为点提示（框中心点）
                x1, y1, x2, y2 = req.box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                points = np.asarray([[center_x, center_y]], dtype=np.float32)
                labels = np.asarray([1], dtype=np.int32)
                print(f"[sam2-batch] 框选限制区域: [{x1:.1f},{y1:.1f}]-[{x2:.1f},{y2:.1f}], 中心点=({center_x:.1f},{center_y:.1f})")
            else:
                points = np.asarray(req.points, dtype=np.float32)
                labels = np.asarray(req.labels or [1] * len(req.points), dtype=np.int32)
            
            frame_idx, object_ids, masks = predictor_model.add_new_points_or_box(
                SAM2_BATCH_STATE["inference_state"], 0, object_id,
                points=points, labels=labels, box=None,
            )
            found = _sam2_batch_masks(masks)
            if not found:
                return {"status": "error", "msg": f"{stage} 点击没有生成有效实例"}
            
            if stage == "container":
                SAM2_BATCH_STATE["prompted_container"] = True
                SAM2_BATCH_STATE["container_points"] = req.points
                SAM2_BATCH_STATE["container_labels"] = req.labels
                SAM2_BATCH_STATE["container_box"] = req.box
                message = "锅已锁定，请点击目标（可添加多个目标）"
            else:
                # 初始化 targets 字典
                if "targets" not in SAM2_BATCH_STATE:
                    SAM2_BATCH_STATE["targets"] = {}
                
                # 保存目标信息
                target_name = req.name or f"target_{object_id}"
                SAM2_BATCH_STATE["targets"][object_id] = {
                    "name": target_name,
                    "points": req.points,
                    "labels": req.labels,
                }
                SAM2_BATCH_STATE["prompted_target"] = True
                message = f"目标 '{target_name}' (ID={object_id}) 已建立 SAM2 memory，可以继续添加更多目标或开始传播"
                
                # 保存点击缓存（支持多目标）
                _save_sam2_click_cache(
                    SAM2_BATCH_STATE.get("container_points", []),
                    SAM2_BATCH_STATE.get("container_labels", []),
                    SAM2_BATCH_STATE.get("container_box"),
                    SAM2_BATCH_STATE.get("targets", {}),
                    SAM2_BATCH_STATE["width"],
                    SAM2_BATCH_STATE["height"],
                )
                print(f"[sam2-batch] 点击缓存已保存 (目标数: {len(SAM2_BATCH_STATE['targets'])})")
            
            SAM2_BATCH_STATE["status"] = "prompted"
            SAM2_BATCH_STATE["message"] = message
            preview = cv2.imread(os.path.join(SAM2_BATCH_STATE["frames_dir"], "000000.jpg"))
            return {
                "status": "ok", "stage": stage, "message": message,
                "object_id": object_id,
                "preview": _tracking_mask_preview(preview, masks, object_ids),
            }
        except Exception as exc:
            traceback.print_exc()
            SAM2_BATCH_STATE.update({"status": "error", "error": str(exc)})
            return {"status": "error", "msg": f"SAM2 提示失败: {exc}"}


def _run_sam2_batch_tracking():
    try:
        with SAM2_BATCH_LOCK:
            state = SAM2_BATCH_STATE["inference_state"]
            root = SAM2_BATCH_STATE["root"]
            frames_dir = SAM2_BATCH_STATE["frames_dir"]
            names = list(SAM2_BATCH_STATE["files"])
            total = SAM2_BATCH_STATE["frame_count"]
            targets = SAM2_BATCH_STATE.get("targets", {})
            SAM2_BATCH_STATE.update({"status": "running", "progress": 0, "message": "SAM2 正在传播目标 memory"})
        batch_stamp = time.strftime("%Y%m%d_%H%M%S")
        batch_id = f"batch_{batch_stamp}"
        batch_dir = os.path.join(OUTPUT_ROOT, batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        predictor_model = _tracking_predictor()
        for done, (frame_idx, object_ids, masks) in enumerate(
            predictor_model.propagate_in_video(state), start=1
        ):
            frame = cv2.imread(os.path.join(frames_dir, f"{frame_idx:06d}.jpg"))
            if frame is None:
                continue
            mask_values = _sam2_batch_masks(masks)
            
            # 为每个目标保存 mask
            target_masks = {}
            for object_index, object_id in enumerate(object_ids):
                obj_id = int(object_id)
                if obj_id >= 2 and object_index < len(mask_values):  # object_id >= 2 是目标
                    target_masks[obj_id] = mask_values[object_index]
            
            image_dir = os.path.join(batch_dir, f"{frame_idx:03d}_{os.path.splitext(os.path.basename(names[frame_idx]))[0]}")
            instance_dir = os.path.join(image_dir, "instances")
            os.makedirs(instance_dir, exist_ok=True)
            cv2.imwrite(os.path.join(image_dir, "original.png"), frame)
            
            # 保存每个目标的实例
            instances_data = []
            overlay = frame.copy()
            
            for idx, (obj_id, target_mask) in enumerate(target_masks.items()):
                target_name = targets.get(obj_id, {}).get("name", f"target_{obj_id}")
                instance_filename = f"instance_{idx+1:03d}_{target_name}.png"
                cv2.imwrite(os.path.join(instance_dir, instance_filename), (target_mask.astype(np.uint8) * 255))
                
                # 使用全局 COLORS 数组（12 种颜色）
                color = np.array(COLORS[idx % len(COLORS)], dtype=np.uint8)
                
                # ===== 简化渲染：只画边界框和标签，不画面具覆盖 =====
                # 提取边界框
                contours, _ = cv2.findContours(target_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox_xyxy = None
                contour_list = []
                if contours:
                    # 边界框
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    bbox_xyxy = [int(x), int(y), int(x + w), int(y + h)]
                    
                    # 绘制边界框矩形
                    cv2.rectangle(overlay, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), 
                                 tuple(int(v) for v in color), 2)
                    
                    # 绘制标签
                    label = target_name
                    overlay = _draw_chinese_text(overlay, label, (x, max(25, y - 25)), color, font_size=20)
                    
                    # 轮廓点（转换为列表格式，用于 JSON 保存）
                    for c in contours:
                        pts = c.reshape(-1, 2).tolist()
                        if len(pts) >= 3:
                            contour_list.append(pts)
                
                instances_data.append({
                    "instance_id": idx + 1,
                    "object_id": obj_id,
                    "name": target_name,
                    "gpt_prompt": targets.get(obj_id, {}).get("gpt_prompt"),
                    "area": int(target_mask.sum()),
                    "bbox_xyxy": bbox_xyxy,
                    "contours": contour_list,
                    "mask_base64_png": mask_to_png_b64(target_mask),
                })
            
            cv2.imwrite(os.path.join(image_dir, "overlay.png"), overlay)
            
            # 保存标注信息
            annotation = {
                "file": names[frame_idx],
                "image_size": {"height": int(frame.shape[0]), "width": int(frame.shape[1])},
                "mode": "sam2_memory_tracking",
                "source": (
                    "gpt_bbox_sam2_memory"
                    if SAM2_BATCH_STATE.get("seed_method") == "gpt_bbox"
                    else "sam2_tracking"
                ),
                "seed_method": SAM2_BATCH_STATE.get("seed_method"),
                "container_concept": "container_click",
                "instance_count": len(instances_data),
                "instances": instances_data,
            }
            with open(os.path.join(image_dir, "annotations.json"), "w", encoding="utf-8") as handle:
                json.dump(annotation, handle, indent=2, ensure_ascii=False)
            
            with SAM2_BATCH_LOCK:
                SAM2_BATCH_STATE["progress"] = int(done * 100 / max(total, 1))
        
        with SAM2_BATCH_LOCK:
            SAM2_BATCH_STATE.update({
                "status": "done", "progress": 100, "message": f"SAM2 批量传播完成（{len(targets)} 个目标）",
                "batch_id": batch_id,
            })
    except Exception as exc:
        traceback.print_exc()
        with SAM2_BATCH_LOCK:
            SAM2_BATCH_STATE.update({"status": "error", "error": str(exc), "message": f"SAM2 批量传播失败: {exc}"})


@app.post("/sam2-batch-tracking/start")
def sam2_batch_tracking_start():
    with SAM2_BATCH_LOCK:
        if not SAM2_BATCH_STATE["prompted_target"]:
            return {"status": "error", "msg": "请先用 GPT + SAM3 初始化首图，或手动点击目标"}
        if SAM2_BATCH_STATE["status"] == "running":
            return {"status": "ok"}
        threading.Thread(target=_run_sam2_batch_tracking, daemon=True).start()
    return {"status": "ok", "message": "已开始 SAM2 memory 传播"}


@app.get("/sam2-batch-tracking/status")
def sam2_batch_tracking_status():
    with SAM2_BATCH_LOCK:
        status = {
            key: SAM2_BATCH_STATE[key]
            for key in ("status", "progress", "message", "error", "frame_count", "width", "height", "batch_id",
                        "container_points", "container_labels", "container_box",
                        "seed_method", "gpt_analysis")
        }
        status["targets"] = _sam2_public_targets(SAM2_BATCH_STATE.get("targets"))
        return status


@app.post("/sam2-batch-tracking/clear-cache")
def sam2_batch_tracking_clear_cache():
    """清除 SAM2 点击点缓存"""
    if os.path.exists(SAM2_CLICK_CACHE_FILE):
        try:
            os.remove(SAM2_CLICK_CACHE_FILE)
            print(f"[sam2-batch] 点击缓存已清除")
            return {"status": "ok", "message": "点击缓存已清除"}
        except Exception as e:
            return {"status": "error", "msg": f"清除缓存失败: {e}"}
    return {"status": "ok", "message": "没有缓存可清除"}


@app.get("/sam2-batch-tracking/cache")
def sam2_batch_tracking_get_cache():
    """获取 SAM2 点击点缓存"""
    cache = _load_sam2_click_cache()
    if cache:
        return {"status": "ok", "cache": cache}
    return {"status": "ok", "cache": None}


@app.post("/sam2-batch-tracking/export-json/{batch_id}")
def sam2_batch_export_json(batch_id: str):
    """导出 SAM2 批处理结果为 JSON（包含所有实例的详细掩码）"""
    if os.path.basename(batch_id) != batch_id or not batch_id.startswith("batch_"):
        return {"status": "error", "msg": "无效的批量任务"}
    batch_dir = os.path.join(OUTPUT_ROOT, batch_id)
    if not os.path.isdir(batch_dir):
        return {"status": "error", "msg": "批量任务不存在"}
    
    folders = sorted(
        name for name in os.listdir(batch_dir)
        if os.path.isdir(os.path.join(batch_dir, name))
    )
    if not folders:
        return {"status": "error", "msg": "批量任务没有图片结果"}
    
    results = []
    for folder in folders:
        image_dir = os.path.join(batch_dir, folder)
        annotation_path = os.path.join(image_dir, "annotations.json")
        if os.path.isfile(annotation_path):
            try:
                with open(annotation_path, "r", encoding="utf-8") as f:
                    annotation = json.load(f)
                # 包含完整的实例信息（包括掩码）
                results.append({
                    "folder": folder,
                    "file": annotation.get("file"),
                    "image_size": annotation.get("image_size"),
                    "mode": annotation.get("mode"),
                    "source": annotation.get("source"),
                    "container_concept": annotation.get("container_concept"),
                    "instance_count": annotation.get("instance_count", 0),
                    "instances": annotation.get("instances", []),  # 包含所有实例的详细掩码
                })
            except Exception as e:
                print(f"[sam2-export] 读取 {annotation_path} 失败: {e}")
    
    json_name = f"{batch_id}_results.json"
    json_path = os.path.join(OUTPUT_ROOT, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "batch_id": batch_id,
            "total_images": len(results),
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    
    return {
        "status": "ok",
        "json_url": f"/batch_download_json/{json_name}",
        "json_name": json_name,
        "total_images": len(results),
    }


@app.get("/batch_download_json/{filename}")
def batch_download_json(filename: str):
    """下载 JSON 结果文件"""
    safe_name = os.path.basename(filename)
    if safe_name != filename or not safe_name.endswith(".json"):
        return {"status": "error", "msg": "无效的文件名"}
    path = os.path.join(OUTPUT_ROOT, safe_name)
    if not os.path.isfile(path):
        return {"status": "error", "msg": "文件不存在"}
    return FileResponse(path, media_type="application/json", filename=safe_name)


@app.post("/sam2-batch-tracking/export-zip/{batch_id}")
def sam2_batch_export_zip(batch_id: str):
    """导出 SAM2 批处理结果为 ZIP"""
    if os.path.basename(batch_id) != batch_id or not batch_id.startswith("batch_"):
        return {"status": "error", "msg": "无效的批量任务"}
    batch_dir = os.path.join(OUTPUT_ROOT, batch_id)
    if not os.path.isdir(batch_dir):
        return {"status": "error", "msg": "批量任务不存在"}
    
    folders = sorted(
        name for name in os.listdir(batch_dir)
        if os.path.isdir(os.path.join(batch_dir, name))
    )
    if not folders:
        return {"status": "error", "msg": "批量任务没有图片结果"}
    
    zip_name = f"{batch_id}_sam2_results.zip"
    zip_path = os.path.join(OUTPUT_ROOT, zip_name)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in folders:
            image_dir = os.path.join(batch_dir, folder)
            for root, dirs, files in os.walk(image_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, batch_dir)
                    zf.write(file_path, arcname)
    
    return {
        "status": "ok",
        "zip_url": f"/batch_download/{zip_name}",
        "zip_name": zip_name,
        "total_images": len(folders),
    }


class Sam3ConceptRequest(BaseModel):
    name: str


def _sam3_load_error(exc, *, model_kind="模型"):
    """Turn low-level SAM3 load errors into an actionable user message.

    A local checkpoint is used by this application, so mentioning Hugging Face
    authentication for every exception is misleading (and was the main reason
    users could not diagnose failed loads).
    """
    text = str(exc).strip() or exc.__class__.__name__
    lower = text.lower()
    if (
        isinstance(exc, FileNotFoundError)
        or "不存在" in text
        or "no such file" in lower
        or "checkpoint" in lower and "not found" in lower
    ):
        return f"SAM3 {model_kind}加载失败：本地权重不存在，请确认文件 {SAM3_CHECKPOINT} 已随程序安装。"
    if "out of memory" in lower or "cuda error" in lower and "memory" in lower:
        return (
            f"SAM3 {model_kind}加载失败：显存不足。请关闭其他占用显存的程序，"
            "或改用 CPU/较小分辨率后重试。"
        )
    if "huggingface" in lower or "gated" in lower or "401" in lower or "403" in lower:
        return (
            f"SAM3 {model_kind}加载失败：需要 Hugging Face 权限或登录。"
            "如果使用本地 sam3.pt，请检查权重文件是否完整；否则执行 hf auth login。"
        )
    if (
        "state_dict" in lower
        or "missing key" in lower
        or "unexpected key" in lower
        or "size mismatch" in lower
    ):
        return (
            f"SAM3 {model_kind}加载失败：权重与当前代码版本不匹配，"
            "请使用对应版本的 sam3.pt（不要混用 SAM3.1 权重）。"
        )
    return f"SAM3 {model_kind}加载失败：{text}"


def _sam3_load_predictor():
    global SAM3_MODEL, SAM3_AUTOCAST_CTX
    if SAM3_MODEL is None:
        try:
            # Enable global autocast before loading model (as per official SAM3 qualitative_test.py line 222)
            # This must stay active for ALL subsequent operations (add_prompt, propagate_in_video)
            # 保存上下文管理器到全局变量，防止被垃圾回收
            if device == "cuda" and SAM3_AUTOCAST_CTX is None:
                SAM3_AUTOCAST_CTX = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                SAM3_AUTOCAST_CTX.__enter__()
            
            from sam3.model_builder import build_sam3_video_predictor
            if not os.path.isfile(SAM3_CHECKPOINT):
                raise FileNotFoundError(f"本地 SAM3 权重不存在: {SAM3_CHECKPOINT}")
            print(f"[sam3-load] 开始加载 SAM3 视频模型...")
            t0 = time.time()
            build_kwargs = dict(
                checkpoint_path=SAM3_CHECKPOINT,
                gpus_to_use=[0] if device == "cuda" else [],
                async_loading_frames=True,
            )
            # torch.compile is not reliable on all Windows/PyTorch builds. It
            # is opt-in and a failed compilation transparently falls back to
            # the regular predictor so model loading itself still succeeds.
            compile_requested = os.environ.get("SAM3_COMPILE", "0").lower() in {
                "1", "true", "yes", "on"
            }
            try:
                SAM3_MODEL = build_sam3_video_predictor(
                    **build_kwargs, compile=compile_requested
                )
            except Exception as compile_exc:
                if not compile_requested:
                    raise
                print(f"[sam3-load] torch.compile 失败，降级为非编译模式: {compile_exc}")
                if device == "cuda":
                    torch.cuda.empty_cache()
                SAM3_MODEL = build_sam3_video_predictor(
                    **build_kwargs, compile=False
                )
            print(f"[sam3-load] SAM3 视频模型加载完成 ({time.time()-t0:.2f}s)")
        except Exception as exc:
            SAM3_MODEL = None
            if SAM3_AUTOCAST_CTX is not None:
                try:
                    SAM3_AUTOCAST_CTX.__exit__(None, None, None)
                except Exception:
                    pass
                SAM3_AUTOCAST_CTX = None
            raise RuntimeError(_sam3_load_error(exc, model_kind="视频模型")) from exc
    return SAM3_MODEL


def _sam3_load_image_model():
    global SAM3_IMAGE_MODEL, SAM3_IMAGE_PROCESSOR
    if SAM3_IMAGE_MODEL is None:
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            if not os.path.isfile(SAM3_CHECKPOINT):
                raise FileNotFoundError(f"本地 SAM3 权重不存在: {SAM3_CHECKPOINT}")
            print(f"[sam3-load] 开始加载 SAM3 图像模型...")
            t0 = time.time()
            SAM3_IMAGE_MODEL = build_sam3_image_model(
                checkpoint_path=SAM3_CHECKPOINT,
                load_from_HF=False,
                device=device,
            )
            SAM3_IMAGE_MODEL.float()
            # Some SAM3 blocks intentionally emit bf16 activations even when
            # the checkpoint is loaded as fp32. On Windows CUDA, normalize
            # every Linear input to its weight dtype to avoid dtype mismatch.
            def _cast_linear_input(module, inputs):
                if inputs and torch.is_tensor(inputs[0]) and inputs[0].dtype != module.weight.dtype:
                    return (inputs[0].to(module.weight.dtype), *inputs[1:])
                return inputs
            for _layer in SAM3_IMAGE_MODEL.modules():
                if isinstance(_layer, torch.nn.Linear):
                    _layer.register_forward_pre_hook(_cast_linear_input)
            SAM3_IMAGE_PROCESSOR = Sam3Processor(
                SAM3_IMAGE_MODEL, device=device, confidence_threshold=0.35
            )
            print(f"[sam3-load] SAM3 图像模型加载完成 ({time.time()-t0:.2f}s)")
        except Exception as exc:
            SAM3_IMAGE_MODEL = None
            SAM3_IMAGE_PROCESSOR = None
            raise RuntimeError(_sam3_load_error(exc, model_kind="图像模型")) from exc
    return SAM3_IMAGE_MODEL, SAM3_IMAGE_PROCESSOR


def _sam3_output_image(frame, outputs):
    image = frame.copy()
    masks = _sam3_output_masks(outputs)
    if masks is None:
        return image
    if torch.is_tensor(masks):
        masks = masks.detach().float().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]
    for index, mask in enumerate(masks):
        mask = mask > 0.5
        color = np.array(COLORS[index % len(COLORS)], dtype=np.uint8)
        image[mask] = (image[mask].astype(np.float32) * 0.20 + color * 0.80).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), 6)
        cv2.drawContours(image, contours, -1, tuple(int(v) for v in color), 3)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cv2.rectangle(image, (x, y), (x + w, y + h), tuple(int(v) for v in color), 3)
            cv2.putText(
                image, f"instance {index + 1}", (x, max(28, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA,
            )
            cv2.putText(
                image, f"instance {index + 1}", (x, max(28, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(v) for v in color), 2, cv2.LINE_AA,
            )
    return image


def _sam3_output_masks(outputs):
    """Normalize SAM3 video outputs to the actual binary mask field."""
    if not isinstance(outputs, dict):
        return None
    # SAM3 video predictor returns this field. Keep the other names for
    # compatibility with older predictor builds.
    for key in ("out_binary_masks", "masks", "mask_logits"):
        value = outputs.get(key)
        if value is not None:
            return value
    return None


_SAM3_VIDEO_PROMPT_ALIASES = {
    "小孩": "child",
    "小朋友": "child",
    "儿童": "child",
    "孩子": "child",
    "男孩": "boy",
    "女孩": "girl",
    "人": "person",
    "人物": "person",
    "人类": "person",
    "小狗": "dog",
    "小猫": "cat",
}


def _sam3_video_prompt_candidates(name):
    """Prefer the user's prompt, then an English fallback for SAM3 video."""
    value = name.strip()
    normalized = _SAM3_VIDEO_PROMPT_ALIASES.get(value, value)
    candidates = [normalized]
    if value != normalized:
        candidates.append(value)
    if normalized not in ("person", "child") and any(
        token in normalized.lower() for token in ("kid", "child", "boy", "girl")
    ):
        candidates.append("person")
    return list(dict.fromkeys(candidates))


def _sam3_video_prompt_from_mask_state(state):
    """Build an explicit SAM3 video prompt from the saved first-frame input."""
    prompt = state.get("video_prompt")
    if not isinstance(prompt, dict):
        return None

    result = dict(prompt)
    # The video API expects boxes as [xmin, ymin, width, height], while the UI
    # stores boxes as [x1, y1, x2, y2].
    box = result.get("box")
    if box and len(box) == 4:
        x1, y1, x2, y2 = [float(value) for value in box]
        result["bounding_boxes"] = [[
            min(x1, x2),
            min(y1, y2),
            abs(x2 - x1),
            abs(y2 - y1),
        ]]
        result["bounding_box_labels"] = [1]
    result.pop("box", None)
    return result


@app.post("/sam3-tracking/upload")
async def sam3_upload(file: UploadFile = File(...)):
    data = await file.read()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    root = os.path.join(OUTPUT_ROOT, "sam3_tracking", stamp)
    frames_dir = os.path.join(root, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    video_filename = os.path.basename(file.filename or "input.mp4")
    video_path = os.path.join(root, video_filename)
    with open(video_path, "wb") as handle:
        handle.write(data)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "msg": "无法打开视频"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{count:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        count += 1
    cap.release()
    with SAM3_LOCK:
        timeline = _new_video_timeline(
            frames_dir,
            count,
            fps,
            "/sam3-tracking",
            randomize=True,
            seed=stamp,
        )
        SAM3_STATE.update({"frames_dir": frames_dir, "root": root, "session_id": None, "frame_count": count,
                           "width": width, "height": height, "fps": fps, "preview_path": os.path.join(root, "tracked.mp4"),
                           "video_path": video_path, "video_filename": video_filename,
                           "prompted": False, "status": "uploaded", "progress": 0,
                           "current_frame": 0, "current_time": 0.0,
                           "video_prompt": None,
                           "sample_interval": VIDEO_SAMPLE_INTERVAL,
                           "sample_mode": "random_per_second",
                           "run_mode": None,
                           "timeline": timeline,
                           "message": "视频已上传，请输入概念", "error": None})
    return {"status": "ok", "width": width, "height": height, "fps": fps, "frame_count": count,
            "sample_interval": VIDEO_SAMPLE_INTERVAL,
            "sample_mode": "random_per_second",
            "timeline": timeline,
            "video_url": f"/sam3-tracking/video/{stamp}/{video_filename}"}


@app.get("/sam3-tracking/video/{stamp}/{filename}")
def sam3_video(stamp: str, filename: str):
    """提供上传的视频文件"""
    safe_filename = os.path.basename(filename)
    video_path = os.path.join(OUTPUT_ROOT, "sam3_tracking", stamp, safe_filename)
    if not os.path.isfile(video_path):
        return {"status": "error", "msg": "视频文件不存在"}
    return FileResponse(
        video_path,
        media_type=mimetypes.guess_type(video_path)[0] or "video/mp4",
        filename=safe_filename,
    )


@app.post("/sam3-tracking/preload")
def sam3_preload():
    """预加载 SAM3 模型，加速后续首帧处理"""
    with SAM3_LOCK:
        if not SAM3_STATE["frames_dir"]:
            return {"status": "error", "msg": "请先上传视频"}
        try:
            print(f"[sam3-preload] 开始预加载模型...")
            t0 = time.time()
            # 预加载图像模型（首帧处理使用）
            _sam3_load_image_model()
            print(f"[sam3-preload] 图像模型预加载完成 ({time.time()-t0:.2f}s)")
            return {"status": "ok", "message": "模型预加载完成"}
        except Exception as exc:
            print(f"[sam3-preload] 预加载失败: {exc}")
            return {"status": "error", "msg": str(exc)}


@app.post("/sam3-tracking/concept")
def sam3_concept(req: Sam3ConceptRequest):
    import time as _time
    t0 = _time.time()
    if not req.name.strip():
        return {"status": "error", "msg": "请输入概念名称"}
    with SAM3_LOCK:
        if not SAM3_STATE["frames_dir"]:
            return {"status": "error", "msg": "请先上传视频"}
        try:
            print(f"[sam3-concept] 开始处理概念: {req.name}")
            
            # 使用 SAM3 图像模型处理首帧（快速，不需要视频 session）
            from PIL import Image
            _, sam3_processor = _sam3_load_image_model()
            sam3_processor.set_confidence_threshold(0.35)
            
            # 读取首帧
            frame_path = os.path.join(SAM3_STATE["frames_dir"], "000000.jpg")
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"[sam3-concept] 错误: 无法读取首帧 {frame_path}")
                return {"status": "error", "msg": "无法读取首帧文件"}
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            state = sam3_processor.set_image(pil_img)
            
            # 尝试多个候选提示
            response = None
            prompt_used = None
            masks = None
            for candidate in _sam3_video_prompt_candidates(req.name):
                print(f"[sam3-concept] 尝试提示: {candidate}")
                t1 = _time.time()
                output = sam3_processor.set_text_prompt(prompt=candidate, state=state)
                print(f"[sam3-concept] 推理完成 ({_time.time()-t1:.2f}s)")
                
                candidate_masks = output.get("masks")
                if candidate_masks is not None and torch.is_tensor(candidate_masks) and len(candidate_masks) > 0:
                    response = output
                    prompt_used = candidate
                    masks = candidate_masks
                    print(f"[sam3-concept] 提示成功: {candidate}, 实例={len(masks)}")
                    break
                print(f"[sam3-concept] 提示无有效 mask: {candidate}")
            
            if response is None or masks is None:
                return {
                    "status": "error",
                    "msg": f"SAM3 首帧没有识别到 {req.name}，请改用英文概念或切换到点击/框选",
                    "mask_count": 0,
                }
            
            # 保存概念提示和 mask
            SAM3_STATE.update({
                "prompted": True, 
                "status": "prompted", 
                "message": "SAM3 已生成首帧概念 mask",
                "concept_prompt": prompt_used,
                "prompt_type": "text",
                "first_frame_mask": masks,  # 保存 mask 用于后续追踪
                "video_prompt": {"text": prompt_used},
            })
            
            # 生成预览
            outputs = {"out_binary_masks": masks}
            _update_sam3_timeline(
                SAM3_STATE, SAM3_STATE["root"], 0, frame, outputs
            )
            preview = _sam3_output_image(frame, outputs)
            if preview is None:
                return {"status": "error", "msg": "预览图生成失败"}
            
            ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                return {"status": "error", "msg": "图像编码失败"}
            
            preview_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            print(f"[sam3-concept] 完成，总耗时: {_time.time()-t0:.2f}s")
            return {
                "status": "ok",
                "preview": preview_b64,
                "mask_count": int(len(masks)),
                "prompt_used": prompt_used,
            }
        except Exception as exc:
            print(f"[sam3-concept] 异常: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            SAM3_STATE.update({"status": "error", "error": str(exc), "message": str(exc)})
            return {"status": "error", "msg": str(exc)}


class Sam3TrackingPromptRequest(BaseModel):
    """SAM3 追踪页面提示请求（支持批处理两阶段逻辑）"""
    concept: str = ""                              # 目标文字（例如：蛋白）
    container: str = ""                            # 限制区域文字（例如：锅），可选
    points: Optional[List[List[float]]] = None     # 目标点击 [[x, y], ...] 归一化坐标 0~1
    labels: Optional[List[int]] = None             # 点击标签 [1, ...] 1=前景 0=背景
    box: Optional[List[float]] = None              # 框选 [x1, y1, x2, y2] 归一化坐标 0~1


@app.post("/sam3-tracking/prompt")
def sam3_tracking_prompt(req: Sam3TrackingPromptRequest):
    """SAM3 追踪页面：使用批处理两阶段逻辑生成首帧 mask

    逻辑与批处理一致：
      Stage 1: 用 container 文字识别限制区域，用 box/click 选择哪个区域
      Stage 2: 裁剪到限制区域，用 target 文字 + click 识别目标
    """
    import time as _time
    t0 = _time.time()
    with SAM3_LOCK:
        if not SAM3_STATE["frames_dir"]:
            return {"status": "error", "msg": "请先上传视频"}
        try:
            from PIL import Image

            concept = (req.concept or "").strip()
            container = (req.container or "").strip()
            model_concept = _batch_model_prompt(concept)
            model_container = _batch_model_prompt(container)
            has_container = bool(container)
            has_target = bool(concept)
            has_box = req.box is not None
            has_points = bool(req.points and req.labels)

            print(f"[sam3-tracking-prompt] 概念映射: 目标={concept!r}->{model_concept!r}, "
                  f"限制区域={container!r}->{model_container!r}")
            print(f"[sam3-tracking-prompt] 提示类型: container={has_container}, target={has_target}, "
                  f"box={has_box}, points={has_points}")

            if not has_target and not has_container and not has_box and not has_points:
                return {"status": "error", "msg": "请输入目标文字、限制区域文字，或在首帧上点击/框选"}

            # 使用 SAM3 图像模型处理首帧
            _, sam3_processor = _sam3_load_image_model()
            sam3_processor.set_confidence_threshold(0.35)

            # 读取首帧
            frame_path = os.path.join(SAM3_STATE["frames_dir"], "000000.jpg")
            frame = cv2.imread(frame_path)
            if frame is None:
                return {"status": "error", "msg": "无法读取首帧文件"}

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            final_masks_np = None  # list of HxW bool arrays
            container_mask_np = None

            if has_container:
                # ===== Stage 1: 识别限制区域 =====
                print(f"[sam3-tracking-prompt] Stage 1: 识别限制区域 '{model_container}'")
                t1 = _time.time()
                region_state = sam3_processor.set_image(Image.fromarray(rgb))
                region_output = sam3_processor.set_text_prompt(
                    prompt=model_container, state=region_state
                )
                region_masks = _sam3_masks_from_output(region_output)
                if not region_masks:
                    sam3_processor.set_confidence_threshold(0.15)
                    region_output = sam3_processor.set_confidence_threshold(
                        0.15, state=region_state
                    )
                    region_masks = _sam3_masks_from_output(region_output)
                print(f"[sam3-tracking-prompt] Stage 1 完成 ({_time.time()-t1:.2f}s): "
                      f"找到 {len(region_masks)} 个限制区域候选")

                # 使用 box/click 选择哪个 container
                # _batch_prompt_reference_point 期望像素坐标，但我们传入的是归一化坐标（0~1）
                # 所以设置 width=1, height=1，让它除以1保持不变
                container_reference_point = _batch_prompt_reference_point(
                    req.points, req.labels, req.box, 1, 1
                )
                # 将归一化坐标转换为像素坐标（_choose_container_mask 期望像素坐标）
                pixel_points = None
                if req.points:
                    pixel_points = [[p[0] * W, p[1] * H] for p in req.points]
                pixel_box = None
                if req.box:
                    pixel_box = [req.box[0] * W, req.box[1] * H, req.box[2] * W, req.box[3] * H]
                # 对于 container 选择，使用 box 和所有点（前景+背景都帮助定位）
                container_mask = _choose_container_mask(
                    region_masks, W, H,
                    prompt_points=pixel_points,
                    prompt_labels=req.labels,
                    prompt_box=pixel_box,
                    reference_point=container_reference_point,
                )

                if container_mask is None:
                    return {"status": "error", "msg": f"未识别到限制区域「{container}」，请调整框选或文字"}

                container_mask_np = container_mask
                print(f"[sam3-tracking-prompt] 选中限制区域: 面积={int(container_mask.sum())}")

                if not has_target and not has_points:
                    # 没有目标文字也没有点击，结果就是 container 本身
                    final_masks_np = [container_mask]
                else:
                    # ===== Stage 2: 裁剪到限制区域，识别目标 =====
                    region_bbox = _batch_mask_bbox(container_mask)
                    x0, y0, x1, y1 = region_bbox
                    pad_x = max(8, int((x1 - x0) * 0.08))
                    pad_y = max(8, int((y1 - y0) * 0.08))
                    crop_x = max(0, x0 - pad_x)
                    crop_y = max(0, y0 - pad_y)
                    crop_x2 = min(W, x1 + pad_x)
                    crop_y2 = min(H, y1 + pad_y)
                    crop = rgb[crop_y:crop_y2, crop_x:crop_x2]
                    crop_h, crop_w = crop.shape[:2]
                    print(f"[sam3-tracking-prompt] Stage 2: 裁剪到限制区域 "
                          f"({crop_x},{crop_y})-({crop_x2},{crop_y2}), 大小={crop_w}x{crop_h}")

                    if has_target:
                        t2 = _time.time()
                        sam3_processor.set_confidence_threshold(0.35)
                        target_state = sam3_processor.set_image(Image.fromarray(crop))
                        target_output = sam3_processor.set_text_prompt(
                            prompt=model_concept, state=target_state
                        )
                        print(f"[sam3-tracking-prompt] Stage 2 文字推理 '{model_concept}' "
                              f"({_time.time()-t2:.2f}s)")
                    else:
                        target_state = sam3_processor.set_image(Image.fromarray(crop))
                        target_output = None

                    # 使用点击选择目标（在裁剪坐标系中）
                    if has_points:
                        for point_idx, point in enumerate(req.points):
                            point_label = bool(
                                req.labels[point_idx]
                                if point_idx < len(req.labels)
                                else True
                            )
                            # 先将归一化坐标转换为像素坐标，再转换为裁剪坐标系
                            pixel_x = float(point[0]) * W
                            pixel_y = float(point[1]) * H
                            crop_point = [
                                (pixel_x - crop_x) / max(crop_w, 1),
                                (pixel_y - crop_y) / max(crop_h, 1),
                            ]
                            # 确保点在裁剪范围内
                            crop_point[0] = max(0.0, min(1.0, crop_point[0]))
                            crop_point[1] = max(0.0, min(1.0, crop_point[1]))
                            target_output = sam3_processor.add_point_prompt(
                                crop_point, point_label, target_state
                            )
                    elif has_box and not has_target:
                        # 没有目标文字但有框选，在裁剪坐标系中使用框选
                        # 先将归一化坐标转换为像素坐标
                        bx1 = float(req.box[0]) * W
                        by1 = float(req.box[1]) * H
                        bx2 = float(req.box[2]) * W
                        by2 = float(req.box[3]) * H
                        # 转换为裁剪坐标系的 [cx, cy, w, h] 格式
                        crop_box = [
                            ((bx1 + bx2) / 2 - crop_x) / max(crop_w, 1),
                            ((by1 + by2) / 2 - crop_y) / max(crop_h, 1),
                            max(1.0, bx2 - bx1) / max(crop_w, 1),
                            max(1.0, by2 - by1) / max(crop_h, 1),
                        ]
                        target_output = sam3_processor.add_geometric_prompt(
                            crop_box, True, target_state
                        )

                    target_masks = _sam3_masks_from_output(target_output) if target_output else []
                    if not target_masks and has_target:
                        sam3_processor.set_confidence_threshold(0.15)
                        target_output = sam3_processor.set_confidence_threshold(
                            0.15, state=target_state
                        )
                        target_masks = _sam3_masks_from_output(target_output)

                    print(f"[sam3-tracking-prompt] Stage 2 找到 {len(target_masks)} 个目标候选")

                    # 将裁剪区域的 mask 映射回全图
                    full_target_masks = [
                        _map_crop_mask(mask, crop_x, crop_y, H, W)
                        for mask in target_masks
                    ]

                    # 过滤：只保留与 container 有足够重叠的 target mask
                    filtered = []
                    for target_mask in full_target_masks:
                        target_area = int(target_mask.sum())
                        if target_area == 0:
                            continue
                        overlap = int((target_mask & container_mask).sum())
                        overlap_ratio = overlap / max(target_area, 1)
                        roi_area = int(target_mask[y0:y1, x0:x1].sum())
                        roi_ratio = roi_area / max(target_area, 1)
                        ys, xs = np.where(target_mask)
                        center_inside = bool(
                            container_mask[int(np.mean(ys)), int(np.mean(xs))]
                        ) if len(xs) else False
                        roi_center_inside = bool(
                            len(xs) and x0 <= int(np.mean(xs)) < x1
                            and y0 <= int(np.mean(ys)) < y1
                        )
                        if overlap_ratio >= 0.30 or (
                            center_inside and overlap_ratio >= 0.12
                        ) or (
                            roi_center_inside and roi_ratio >= 0.30
                        ):
                            filtered.append(target_mask)

                    if filtered:
                        final_masks_np = filtered
                        print(f"[sam3-tracking-prompt] 过滤后保留 {len(filtered)}/{len(full_target_masks)} 个目标 mask")
                    elif full_target_masks:
                        final_masks_np = full_target_masks
                        print(f"[sam3-tracking-prompt] 过滤后无匹配，使用全部 {len(full_target_masks)} 个目标 mask")
                    else:
                        return {
                            "status": "error",
                            "msg": f"在限制区域内没有识别到「{concept}」，请调整点击位置或文字",
                            "mask_count": 0,
                        }

            else:
                # ===== 无限制区域：直接使用目标文字 + 点击/框选 =====
                print(f"[sam3-tracking-prompt] 无限制区域，直接推理")
                sam3_processor.set_confidence_threshold(0.35)
                state = sam3_processor.set_image(Image.fromarray(rgb))
                output = None

                if has_target:
                    t1 = _time.time()
                    output = sam3_processor.set_text_prompt(prompt=model_concept, state=state)
                    print(f"[sam3-tracking-prompt] 文字推理 '{model_concept}' ({_time.time()-t1:.2f}s)")

                if has_points:
                    if len(req.points) != len(req.labels):
                        return {"status": "error", "msg": "点和标签数量不一致"}
                    if not any(int(label) == 1 for label in req.labels):
                        return {"status": "error", "msg": "请至少添加一个前景点"}
                    for point, label in zip(req.points, req.labels):
                        output = sam3_processor.add_point_prompt(point, bool(label), state)
                elif has_box:
                    x1, y1, x2, y2 = req.box
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    output = sam3_processor.add_geometric_prompt([cx, cy, w, h], True, state)

                if output is None:
                    return {"status": "error", "msg": "请提供目标文字、点击或框选"}

                masks = output.get("masks") if output else None
                if masks is None or (torch.is_tensor(masks) and len(masks) == 0):
                    return {
                        "status": "error",
                        "msg": "首帧没有生成有效 mask，请增加前景点、添加背景点，或改用框选",
                        "mask_count": 0,
                    }

                # 如果有框选，过滤不在框选区域内的 mask
                if has_box and torch.is_tensor(masks):
                    x1, y1, x2, y2 = req.box
                    x1_px, y1_px = int(x1 * W), int(y1 * H)
                    x2_px, y2_px = int(x2 * W), int(y2 * H)
                    if masks.dim() == 4:
                        masks = masks.squeeze(1)
                    filtered_masks = []
                    for mask in masks:
                        mask_bool = mask > 0.5
                        if mask_bool.any():
                            box_region = mask_bool[y1_px:y2_px, x1_px:x2_px]
                            overlap_pixels = box_region.sum().item()
                            mask_area = mask_bool.sum().item()
                            min_overlap = max(100, mask_area * 0.3)
                            if overlap_pixels >= min_overlap:
                                filtered_masks.append(mask)
                    if filtered_masks:
                        masks = torch.stack(filtered_masks)
                        print(f"[sam3-tracking-prompt] 框选过滤后保留 {len(filtered_masks)} 个 mask")

                # 转换为 numpy bool arrays
                if torch.is_tensor(masks):
                    if masks.dim() == 4:
                        masks = masks.squeeze(1)
                    final_masks_np = [m.cpu().numpy() > 0.5 for m in masks]
                else:
                    final_masks_np = _sam3_masks_from_output(output)

            # ===== 结果处理 =====
            if not final_masks_np:
                return {"status": "error", "msg": "首帧没有生成有效 mask", "mask_count": 0}

            # 转换为 tensor
            final_masks_tensor = torch.tensor(
                np.asarray(final_masks_np, dtype=np.float32)
            )
            print(f"[sam3-tracking-prompt] 最终 mask 数量: {len(final_masks_np)}, "
                  f"总耗时: {_time.time()-t0:.2f}s")

            # 保存状态
            # 提取首帧 mask 的中心点和边界框用于后续抽样推理
            combined_mask = np.any(np.asarray(final_masks_np), axis=0)
            ys, xs = np.where(combined_mask)
            first_frame_center = None
            first_frame_box = None
            if len(xs) > 0 and len(ys) > 0:
                center_x = xs.mean() / W
                center_y = ys.mean() / H
                first_frame_center = [center_x, center_y]
                x_min, x_max = xs.min() / W, xs.max() / W
                y_min, y_max = ys.min() / H, ys.max() / H
                first_frame_box = [
                    (x_min + x_max) / 2, (y_min + y_max) / 2,
                    x_max - x_min, y_max - y_min,
                ]

            SAM3_STATE.update({
                "prompted": True,
                "status": "prompted",
                "message": "SAM3 已生成首帧 mask",
                "concept_prompt": model_concept if has_target else None,
                "container_prompt": model_container if has_container else None,
                "prompt_type": "batch_two_stage" if has_container else "geometric",
                "first_frame_mask": final_masks_tensor,
                "first_frame_center": first_frame_center,
                "first_frame_box": first_frame_box,
                "first_frame_prompt": {
                    "points": req.points if req.points else None,
                    "point_labels": req.labels if req.labels else None,
                    "bounding_boxes": [[(req.box[0]+req.box[2])/2, (req.box[1]+req.box[3])/2,
                                        abs(req.box[2]-req.box[0]), abs(req.box[3]-req.box[1])]]
                                       if req.box else None,
                 },
            })

            # Keep the user's actual prompt for full-video tracking. The
            # image-model mask above is still used by the sampled image mode,
            # but it must not silently replace SAM3's native video prompt.
            if req.points and req.labels and any(int(label) == 1 for label in req.labels):
                SAM3_STATE["video_prompt"] = {
                    "points": req.points,
                    "point_labels": req.labels,
                    "obj_id": 1,
                }
            elif has_target:
                SAM3_STATE["video_prompt"] = {
                    "text": model_concept,
                    "box": req.box if req.box else None,
                }
            elif req.box:
                SAM3_STATE["video_prompt"] = {"box": req.box}
            elif has_container:
                SAM3_STATE["video_prompt"] = {"text": model_container}
            else:
                SAM3_STATE["video_prompt"] = None

            # 生成预览
            outputs = {"out_binary_masks": final_masks_tensor}
            _update_sam3_timeline(
                SAM3_STATE, SAM3_STATE["root"], 0, frame, outputs
            )
            preview = _sam3_output_image(frame, outputs)
            if preview is None:
                return {"status": "error", "msg": "预览图生成失败"}

            ok, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                return {"status": "error", "msg": "图像编码失败"}

            preview_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            return {
                "status": "ok",
                "preview": preview_b64,
                "mask_count": int(len(final_masks_np)),
            }
        except Exception as exc:
            print(f"[sam3-tracking-prompt] 异常: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return {"status": "error", "msg": str(exc)}


def _run_sam3_tracking(sample_only=False):
    try:
        with SAM3_LOCK:
            frames_dir, output_path = SAM3_STATE["frames_dir"], SAM3_STATE["preview_path"]
            total, fps, width, height = SAM3_STATE["frame_count"], SAM3_STATE["fps"], SAM3_STATE["width"], SAM3_STATE["height"]
            timeline = SAM3_STATE["timeline"].copy()
            prompt_type = SAM3_STATE.get("prompt_type", "text")
            first_frame_mask = SAM3_STATE.get("first_frame_mask", None)
            run_mode = "sampled" if sample_only else "full"
            message = "SAM3 正在对抽样帧独立推理（图像模式）" if sample_only else "SAM3 正在传播 mask，生成全程结果视频"
            SAM3_STATE.update({
                "status": "running",
                "progress": 0,
                "run_mode": run_mode,
                "message": message,
            })
            # 诊断信息
            print(f"[sam3-tracking] 开始追踪: 模式={run_mode}, 总帧数={total}")
            print(f"[sam3-tracking] 设备: {device}, autocast: {device == 'cuda'}")
        writer = None
        if not sample_only:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
        try:
            # torch.autocast 是线程局部的，后台线程需要重新启用
            ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
            with ctx:
                if sample_only:
                    # 抽样模式：使用 SAM3 图像模型对每一帧独立处理（与批处理逻辑一致）
                    from PIL import Image
                    
                    # 获取概念提示和首帧 mask
                    with SAM3_LOCK:
                        concept_prompt = SAM3_STATE.get("concept_prompt", None)
                        first_frame_prompt = SAM3_STATE.get("first_frame_prompt", None)
                        # 从 timeline 中获取首帧的 mask
                        if first_frame_mask is None:
                            for item in SAM3_STATE.get("timeline", []):
                                if item["frame_index"] == 0 and item.get("mask") is not None:
                                    first_frame_mask = item["mask"]
                                    break
                    
                    # 提取 timeline 中的帧索引（去重并排序，跳过首帧）
                    sample_frames = sorted([
                        item["frame_index"] for item in timeline 
                        if item["frame_index"] != 0
                    ])
                    print(f"[sam3-tracking-sample] 抽样模式（与批处理一致）：对 {len(sample_frames)} 个帧独立推理")
                    
                    # 从首帧 mask 提取提示（中心点或边界框）
                    first_frame_center = None
                    first_frame_box = None
                    if first_frame_mask is not None:
                        if torch.is_tensor(first_frame_mask):
                            mask_np = first_frame_mask.cpu().numpy()
                        else:
                            mask_np = np.asarray(first_frame_mask)
                        
                        # 确保是 2D mask
                        if mask_np.ndim > 2:
                            mask_np = mask_np.squeeze()
                        
                        if mask_np.any():
                            # 计算中心点
                            ys, xs = np.where(mask_np > 0.5)
                            if len(xs) > 0 and len(ys) > 0:
                                center_x = xs.mean() / mask_np.shape[1]
                                center_y = ys.mean() / mask_np.shape[0]
                                first_frame_center = [center_x, center_y]
                                
                                # 计算边界框
                                x_min, x_max = xs.min() / mask_np.shape[1], xs.max() / mask_np.shape[1]
                                y_min, y_max = ys.min() / mask_np.shape[0], ys.max() / mask_np.shape[0]
                                # 转换为 [cx, cy, w, h] 格式
                                cx = (x_min + x_max) / 2
                                cy = (y_min + y_max) / 2
                                w = x_max - x_min
                                h = y_max - y_min
                                first_frame_box = [cx, cy, w, h]
                                
                                print(f"[sam3-tracking-sample] 从首帧 mask 提取提示: center={first_frame_center}, box={first_frame_box}")
                    
                    # 加载 SAM3 图像模型
                    _, sam3_processor = _sam3_load_image_model()
                    sam3_processor.set_confidence_threshold(0.35)
                    
                    saved_count = 0
                    for done, frame_idx in enumerate(sample_frames, start=1):
                        frame_path = os.path.join(frames_dir, f"{frame_idx:06d}.jpg")
                        frame = cv2.imread(frame_path)
                        if frame is None:
                            print(f"[sam3-tracking-sample] 无法读取帧 {frame_idx}")
                            continue
                        
                        # 转换为 RGB 并设置为 PIL Image
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb)
                        
                        # 对该帧独立运行推理（使用 SAM3 图像模型）
                        state = sam3_processor.set_image(pil_img)
                        
                        # 使用概念文字进行分割（与批处理后续图片一致）
                        output = None
                        if concept_prompt:
                            output = sam3_processor.set_text_prompt(prompt=concept_prompt, state=state)
                        
                        # 如果有首帧 mask 提取的提示，使用它来约束分割（与批处理使用首图点击/框选一致）
                        if first_frame_center is not None:
                            # 使用中心点作为前景点提示
                            output = sam3_processor.add_point_prompt(first_frame_center, True, state)
                        elif first_frame_box is not None:
                            # 使用边界框作为框提示
                            output = sam3_processor.add_geometric_prompt(first_frame_box, True, state)
                        elif first_frame_prompt and "points" in first_frame_prompt and first_frame_prompt["points"]:
                            # 回退到原始点提示
                            points = first_frame_prompt["points"]
                            labels = first_frame_prompt.get("point_labels", [1] * len(points))
                            for point, label in zip(points, labels):
                                if output is None:
                                    output = sam3_processor.add_point_prompt(point, bool(label), state)
                                else:
                                    output = sam3_processor.add_point_prompt(point, bool(label), state)
                        elif first_frame_prompt and "bounding_boxes" in first_frame_prompt and first_frame_prompt["bounding_boxes"]:
                            # 回退到原始框提示
                            boxes = first_frame_prompt["bounding_boxes"]
                            if boxes:
                                box = boxes[0]
                                x1, y1, w, h = box
                                cx = x1 + w / 2
                                cy = y1 + h / 2
                                output = sam3_processor.add_geometric_prompt([cx, cy, w, h], True, state)
                        
                        if output is None:
                            output = {}
                        
                        # 转换输出为 SAM3 视频模型格式
                        masks = output.get("masks")
                        if masks is not None and torch.is_tensor(masks):
                            # 后处理过滤：只保留与首帧边界框有重叠的 mask
                            if first_frame_box is not None and len(masks) > 0:
                                # first_frame_box 是 [cx, cy, w, h] 归一化格式
                                cx, cy, w_box, h_box = first_frame_box
                                x1_box = max(0, cx - w_box / 2)
                                y1_box = max(0, cy - h_box / 2)
                                x2_box = min(1, cx + w_box / 2)
                                y2_box = min(1, cy + h_box / 2)
                                
                                # 转换为像素坐标
                                h_img, w_img = masks.shape[-2], masks.shape[-1]
                                x1_px = int(x1_box * w_img)
                                y1_px = int(y1_box * h_img)
                                x2_px = int(x2_box * w_img)
                                y2_px = int(y2_box * h_img)
                                
                                filtered_masks = []
                                for mask in masks:
                                    mask_bool = mask > 0.5
                                    if mask_bool.any():
                                        # 计算 mask 与首帧边界框的重叠
                                        box_region = mask_bool[y1_px:y2_px, x1_px:x2_px]
                                        overlap_pixels = box_region.sum().item()
                                        mask_area = mask_bool.sum().item()
                                        # 更严格的过滤：至少100像素或 mask 面积的30%在框内
                                        min_overlap = max(100, mask_area * 0.3)
                                        if overlap_pixels >= min_overlap:
                                            filtered_masks.append(mask)
                                
                                if filtered_masks:
                                    masks = torch.stack(filtered_masks)
                                    print(f"[sam3-tracking-sample] 帧 {frame_idx}: 过滤后保留 {len(filtered_masks)} 个 mask（共 {len(masks)} 个）")
                                else:
                                    print(f"[sam3-tracking-sample] 帧 {frame_idx}: 过滤后没有保留的 mask，使用原始结果")
                            
                            outputs = {"out_binary_masks": masks}
                        else:
                            outputs = {}
                        
                        # 更新 timeline
                        _update_sam3_timeline(
                            SAM3_STATE, os.path.dirname(frames_dir),
                            frame_idx, frame, outputs,
                        )
                        saved_count += 1
                        
                        with SAM3_LOCK:
                            SAM3_STATE["current_frame"] = int(frame_idx)
                            SAM3_STATE["current_time"] = float(frame_idx) / max(fps or 25.0, 1e-6)
                            SAM3_STATE["progress"] = int(done * 100 / max(len(sample_frames), 1))
                        
                        if saved_count <= 3 or saved_count % 10 == 0:
                            print(f"[sam3-tracking-sample] 处理抽样帧 {saved_count}/{len(sample_frames)}: frame_idx={frame_idx}")
                    
                    print(f"[sam3-tracking-sample] 抽样推理完成，共处理 {saved_count} 个帧")
                else:
                    # 完整追踪模式：创建视频 session，并用 SAM3 原生视频
                    # prompt 初始化首帧；图像模型生成的 mask 仅属于抽样模式。
                    print(f"[sam3-tracking] 创建视频 session 用于追踪...")
                    t0 = time.time()
                    predictor_model = _sam3_load_predictor()
                    session = predictor_model.handle_request({
                        "type": "start_session",
                        "resource_path": frames_dir,
                        "offload_video_to_cpu": True,
                    })
                    session_id = session["session_id"]
                    print(f"[sam3-tracking] Session 创建成功: {session_id} ({time.time()-t0:.2f}s)")

                    video_prompt = _sam3_video_prompt_from_mask_state(SAM3_STATE)
                    if video_prompt:
                        prompt_request = {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": 0,
                            **video_prompt,
                        }
                        print(
                            "[sam3-tracking] 使用 SAM3 视频首帧提示: "
                            f"{ {key: value for key, value in video_prompt.items() if key != 'points'} }"
                        )
                        t1 = time.time()
                        prompt_response = predictor_model.handle_request(prompt_request)
                        print(f"[sam3-tracking] add_prompt 完成 ({time.time()-t1:.2f}s)")
                    else:
                        raise RuntimeError(
                            "没有可用的 SAM3 视频首帧提示，请重新生成首帧提示"
                        )
                    
                    # 运行 propagate_in_video 处理所有帧
                    saved_count = 0
                    for done, response in enumerate(predictor_model.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}), start=1):
                        frame_idx = int(response.get("frame_index", done - 1))
                        frame = cv2.imread(os.path.join(frames_dir, f"{frame_idx:06d}.jpg"))
                        outputs = response.get("outputs", {})
                        
                        if writer is not None:
                            writer.write(_sam3_output_image(frame, outputs))
                        with SAM3_LOCK:
                            SAM3_STATE["current_frame"] = int(frame_idx)
                            SAM3_STATE["current_time"] = float(frame_idx) / max(fps or 25.0, 1e-6)
                            SAM3_STATE["progress"] = int(done * 100 / max(total, 1))
                            # 检查是否应该保存到 timeline
                            should_save = any(item["frame_index"] == int(frame_idx) for item in SAM3_STATE["timeline"])
                            if should_save:
                                saved_count += 1
                                if saved_count <= 3 or saved_count % 10 == 0:
                                    print(f"[sam3-tracking] 保存抽样帧 {saved_count}: frame_idx={frame_idx}")
                                _update_sam3_timeline(
                                    SAM3_STATE, os.path.dirname(frames_dir),
                                    frame_idx, frame, outputs,
                                )
        finally:
            if writer is not None:
                writer.release()
        with SAM3_LOCK:
            message = "SAM3 抽样分割完成（图像模式）" if sample_only else "SAM3 全程追踪完成"
            SAM3_STATE.update({"status": "done", "progress": 100, "message": message})
    except Exception as exc:
        traceback.print_exc()
        with SAM3_LOCK:
            SAM3_STATE.update({"status": "error", "error": str(exc), "message": str(exc)})


@app.post("/sam3-tracking/start")
def sam3_start():
    with SAM3_LOCK:
        if not SAM3_STATE["prompted"]:
            return {"status": "error", "msg": "请先输入概念或点击/框选生成首帧 mask"}
        if SAM3_STATE["status"] != "running":
            threading.Thread(target=_run_sam3_tracking, daemon=True).start()
    return {"status": "ok"}


@app.post("/sam3-tracking/start-sampled")
def sam3_start_sampled():
    """Propagate SAM3 while saving only the random per-second timeline frames."""
    with SAM3_LOCK:
        if not SAM3_STATE["prompted"]:
            return {"status": "error", "msg": "请先生成首帧 mask"}
        if SAM3_STATE["status"] != "running":
            threading.Thread(
                target=_run_sam3_tracking,
                kwargs={"sample_only": True},
                daemon=True,
            ).start()
    return {"status": "ok"}


@app.get("/sam3-tracking/status")
def sam3_status():
    with SAM3_LOCK:
        response = {
            key: SAM3_STATE[key]
            for key in (
                "status", "progress", "message", "error", "frame_count",
                "width", "height", "fps", "run_mode",
            )
        }
        response.update(_timeline_status(SAM3_STATE))
        return response


@app.get("/sam3-tracking/timeline/{frame_index}")
def sam3_timeline(frame_index: int):
    with SAM3_LOCK:
        frames_dir = SAM3_STATE.get("frames_dir")
        root = SAM3_STATE.get("root")
    if not root or not frames_dir or frame_index < 0:
        return {"status": "error", "msg": "采样帧不存在"}
    preview_path = os.path.join(root, "timeline", f"{frame_index:06d}.jpg")
    source_path = os.path.join(frames_dir, f"{frame_index:06d}.jpg")
    path = preview_path if os.path.isfile(preview_path) else source_path
    if not os.path.isfile(path):
        return {"status": "error", "msg": "采样帧不存在"}
    return FileResponse(path, media_type="image/jpeg")


@app.get("/sam3-tracking/result")
def sam3_result():
    with SAM3_LOCK:
        path = SAM3_STATE.get("preview_path")
        if SAM3_STATE.get("status") != "done" or not path or not os.path.isfile(path):
            return {"status": "error", "msg": "SAM3 结果尚未生成"}
    return FileResponse(path, media_type="video/mp4", filename="sam3_tracking.mp4")


# ============================================================
if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(TEMPLATE_XLSX):
        create_template()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
