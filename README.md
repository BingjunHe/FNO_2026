# FNO 项目环境配置指南 (Environment Setup Guide)

本项目基于 Python 3.11 开发，以下是配置并运行运行环境的具体步骤。

## ⚙️ 环境要求 (Prerequisites)
- Anaconda 或 Miniconda
- NVIDIA 显卡（支持 CUDA 加速）

## 🚀 快速开始 (Quick Start)

### 1. 创建并激活 Conda 虚拟环境
首先，打开终端（Terminal）并运行以下命令，创建一个名为 `FNO` 且 Python 版本为 3.11 的虚拟环境：
```bash
conda create -n FNO python=3.11
```

创建完成后，请务必**激活**该环境：
```bash
conda activate FNO
```

### 2. 安装 Python 依赖库
进入到项目的根目录（即 `requirements.txt` 文件所在的文件夹），执行以下命令安装所需依赖：
```bash
pip install -r requirements.txt
```
> 💡 *注：标准的 pip 批量安装命令需要加上 `-r` 参数，且文件名通常为复数 `requirements.txt`。*

### 3. 手动安装对应版本的 CUDA
请根据项目对深度学习框架（如 PyTorch）的版本要求，前往 NVIDIA 官网手动下载并安装对应版本的 CUDA Toolkit，以确保 GPU 算力可以被正常调用。

### 4. 运行项目
所有环境配置完成后，切换到 `src` 文件夹并运行 `main.py` 主程序：
```bash
cd src
python main.py
```