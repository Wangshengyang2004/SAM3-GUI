# SAM3.1 图形界面/API

[English](../README.md) | **简体中文**

基于 **SAM 3.1 Object Multiplex** 的视频和图像分割图形界面和 HTTP API，支持文本、点和框提示。本仓库只支持 SAM 3.1 的 `sam3.1_multiplex.pt` 检查点，不支持旧版 SAM 3.0 `sam3.pt`。

## 主要功能

- **原生 SAM 3.1 支持**: 使用 `facebookresearch/sam3` 当前的 Object Multiplex API
- **文本提示**: 使用自然语言描述分割对象（例如"人"、"汽车"、"红鞋子"）
- **点选交互**: 使用正/负样本点进行交互式优化
- **框选提示**: 绘制边界框来分割对象
- **视频追踪**: 跨视频帧的多对象追踪，支持传播方向设置
- **HTTP API**: 提供会话、提示、传播、对象移除和图像分割接口
- **多对象管理**: 使用"添加新掩码"功能独立追踪多个对象

## 安装

### 环境要求

- Python 3.12 或更高版本
- PyTorch 2.7 或更高版本
- 支持 CUDA 的 GPU，CUDA 12.6 或更高版本
- **FFmpeg**（视频处理必需）：通过 `sudo apt-get install ffmpeg` (Ubuntu/Debian) 或 `brew install ffmpeg` (macOS) 安装

### 1. 安装 SAM3

首先安装 [SAM3](https://github.com/facebookresearch/sam3) 包。需要使用包含 SAM 3.1 Object Multiplex API 的当前版本：

```bash
# 创建新的 Conda 环境
conda create -n sam3 python=3.12
conda activate sam3

# 安装带 CUDA 支持的 PyTorch
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Blackwell RTX 50X0 GPU 用户注意：** 这些 GPU 可能需要从源码编译 torchvision；详见 [blackwell_support.md](blackwell_support.md)。

```bash
# 克隆仓库并安装包
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .

# 可选：安装示例笔记本或开发的额外依赖
# 用于运行示例笔记本
pip install -e ".[notebooks]"

# 用于开发
pip install -e ".[train,dev]"
```

### 2. 安装 GUI 依赖

```bash
cd SAM3-GUI
pip install -r requirements.txt
```

**注意：** 需要先安装最新版 SAM3 包，并确认 `sam3.model_builder.build_sam3_predictor` 支持 `version="sam3.1"`，才能使用本项目。

### 3. 下载 SAM 3.1 模型

本项目使用 `facebook/sam3.1` 的 `sam3.1_multiplex.pt`。Hugging Face 是默认下载源：

```bash
# 首次使用 Hugging Face gated checkpoint 前需要登录
hf auth login

# 默认保存到 ~/sam3/model/sam3.1_multiplex.pt
python -m tools.download_model --source huggingface

# 或指定自定义输出目录
python -m tools.download_model --source huggingface --output_dir /path/to/model/dir
```

如果 Hugging Face 不可用，可以改用 ModelScope：

```bash
python -m tools.download_model --source modelscope
```

两个下载路径都只会获取 `sam3.1_multiplex.pt`。如果你的 ModelScope 镜像使用不同 repo id，使用 `--modelscope_model_id <repo-id>` 指定。

## 使用说明

### 启动 GUI

```bash
python cli.py data.root_dir=data_root server.name=0.0.0.0 server.port=8890
```

常用 Hydra 覆盖项：
- `sam.checkpoint_path=/path/to/sam3.1_multiplex.pt`
- `sam.gpus=[0]`
- `sam.use_fa3=true`
- `data.vid_name=videos data.img_name=images data.mask_name=masks`
- `server.reload=true`

默认配置由 `sam3_gui/conf/config.yaml` 和 `server`、`data`、`sam` 分组组合而成。旧参数（例如 `--root_dir`、`--port`、`--use_fa3`）仍可使用，但新配置优先使用 Hydra 覆盖项。

### 数据组织

按以下结构组织你的数据：

```
data_root/
├── videos/          # 用于提取帧的 MP4 文件
│   ├── seq1.mp4
│   └── seq2.mp4
├── images/          # 预提取的帧序列
│   ├── seq1/
│   │   ├── frame_00000.png
│   │   ├── frame_00001.png
│   │   └── ...
│   └── seq2/
│       └── ...
└── masks/           # 保存掩码的输出目录
    ├── seq1/
    │   ├── frame_00000.png
    │   ├── frame_00000.npy
    │   └── ...
    └── seq2/
```

## 视频模式工作流程

### 1. 加载帧

- **选项 A**：选择视频文件并提取帧
  - 从下拉菜单选择视频
  - 设置开始/结束时间、FPS 和高度
  - 点击"提取帧"

- **选项 B**：加载预提取的帧
  - 从下拉菜单选择帧文件夹
  - 点击"加载所选帧"

### 2. 添加提示

从三种提示类型中选择：

#### **文本提示**
1. 输入文本描述（例如"人"、"汽车"、"红衬衫"）
2. 点击"使用文本检测"
3. 使用"添加新掩码"来分割更多对象

#### **点提示**
1. 点击"+ 正样本"添加包含点（绿色）
2. 点击"- 负样本"添加排除点（红色）
3. 在**输出图像**上点击放置点
4. 点是帧特定的 - 切换帧以在其他帧上添加点
5. 查看"已添加点"表格了解所有帧上的点

#### **框提示**
1. 点击"框选分割"
2. 在帧上点击两个角点绘制框
3. 框内的对象将被分割

### 3. 管理对象

- **查看追踪对象**：下拉菜单显示所有检测到的对象（0、1、2、...）
- **移除对象**：选择对象并点击"移除所选对象"

### 4. 视频追踪

1. 选择传播方向：
   - **向前**：从当前帧传播到结尾
   - **向后**：从当前帧传播到开头
   - **双向**：向两个方向传播（默认）

2. 点击"追踪所有帧"
3. 查看追踪视频输出
4. 使用帧滑块查看单个帧的结果

### 5. 保存掩码

- 掩码保存路径自动生成：`{root_dir}/masks/{sequence_name}/`
- 点击"保存掩码"将掩码导出为 PNG 和 NPZ 文件

## 图像模式工作流程

单图像分割，提供三种模式：

### **查找全部模式**
1. 输入文本提示（例如"鞋子"、"人"、"汽车"）
2. 调整置信度阈值（0.0-1.0）
3. 点击"查找全部"检测所有匹配的对象

### **框选模式**
1. 点击"框选分割"
2. 在图像上点击两个角点绘制框
3. 框内的对象将被分割

### **点选模式**
1. 点击"+ 正样本"或"- 负样本"
2. 在图像上点击放置点
3. 使用"按索引移除点"删除特定点

## 致谢

本应用基于 [shape-of-motion](https://github.com/vye16/shape-of-motion/) 修改，并适配 Meta 的 [SAM3](https://github.com/facebookresearch/sam3) SAM 3.1 Object Multiplex 实现和检查点。

![SAM3 GUI 视频模式](assets/sam3_1.png)

![SAM3 GUI 图像模式](assets/sam3_2.png)
