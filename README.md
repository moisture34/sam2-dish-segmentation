# SAM2 菜式图像分割

基于 SAM 2 的食材图片分割小应用。上传菜品图片后，可通过自动、精细或颜色引导模式生成分割结果，并可选用 Excel 配置调整搜索区域和颜色参数。

本仓库只包含应用代码和 SAM 2 源码，不包含模型权重、用户图片、生成结果或密钥。

## 环境要求

- Python 3.10 或更高版本
- 推荐使用 NVIDIA GPU；请先根据本机 CUDA 版本安装 PyTorch 和 TorchVision
- SAM 2.1 Large checkpoint（约 900 MB）

## 安装与运行

在本目录下执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

从 Meta 官方下载 SAM 2.1 Large 权重，并保存为：

`checkpoints/sam2.1_hiera_large.pt`

官方下载地址：
<https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt>

然后启动应用：

```powershell
python predictor_web2.py
```

终端会显示本地 Gradio 页面地址。首次运行时，如果配置模板不存在，应用会生成 `配置模板.xlsx`。

## 启动 FastAPI 网页版

仓库也包含基于 FastAPI 的网页后端和交互界面。下载 SAM 2.1 Large 权重到上面的 `checkpoints/` 目录后，运行：

```powershell
python sam_backen.py
```

在浏览器打开 <http://127.0.0.1:8000>。需要 GPT 辅助分析时，通过环境变量配置 `OPENAI_API_KEY`，或配置 `QWEN_API_KEY`；密钥不要写入源码或提交到 Git。

SAM3 跟踪功能还需要另外准备 SAM3 源码和 `sam3.pt` 权重；这些大文件不包含在本仓库中。

## 项目结构

- `predictor_web2.py`：Gradio 食材分割应用
- `sam_backen.py`：FastAPI 后端
- `index.html`、`concept_tracking.html`、`sam3_tracking.html`：浏览器交互界面
- `sam2/`：SAM 2 源码与模型配置
- `checkpoints/`：本地模型权重目录（权重不提交到 Git）
- `配置模板.xlsx`：应用参数模板

SAM 2 源码及其许可证来自 [facebookresearch/sam2](https://github.com/facebookresearch/sam2)；请同时遵守模型权重的官方使用条款。
