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

### 3. 安装对应版本的 CUDA
 硬件驱动检查 (显卡关键步)
在安装深度学习框架前，请在 Windows 终端中执行以下命令，务必确认你的 NVIDIA 驱动版本：
```bash
nvidia-smi
```
⚠️ 注意：请检查表格右上角的 CUDA Version。若其版本 低于 12.6，请前往 NVIDIA 官网下载并更新最新的显卡驱动（Game Ready 或 Studio 驱动均可），否则底层 C++ 内存分配器在找不到 GPU 时会触发无能狂怒的 Warning 警告。

针对 torch==2.13.0 版本，强制重装官方指定的 CUDA 12.6 精准镜像包（体积约为几 GB，带有完整的底层频域变换硬件规划器）：
```bash
pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126) --force-reinstall
```

### 4. 运行项目
所有环境配置完成后，
进入`src\model\processing`运行`prediction.py`
生成3D网格的tensor，就是在data里面的pt文件
切换到 `src` 文件夹并运行 `main.py` 主程序：
```bash
cd src
python main.py
```