# Qwen2-0.5B-axi

> **⚠️ 免责声明（请务必仔细阅读）**
> 
> **本仓库仅作为模型部署参考，请勿未经全面测试直接将本仓库代码用于实际项目！**
> 
> **本仓库不承担任何不良后果！**

## 0.文件结构目录

```
├── config/                           # 配置文件目录
│   └── Qwen2-0___5B-Instruct/        # 模型配置文件
├── fmodel/                           # onnx模型
├── modelzoo_utils/                   # 模型库工具依赖包
├── Qwen2-0.5B_decoder.toml          # Icraft编译配置文件
├── qwen2_infer_decode_only_axi.py   # axi上板推理脚本
├── configuration.json               # 项目配置文件
└── README.md                        # 说明文档
```

## 1.项目简介

本工程是基于Icraft-zg330板上部署的qwen2-0.5B的示例工程，推理效果如下：

![alt text](image.png)




## 2.环境准备

### 1.克隆项目

您可以通过如下git clone命令，或者ModelScope SDK来下载模型

SDK下载
```bash
# 安装ModelScope
pip install modelscope
```
```python
# SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('AIBS/Qwen2_0.5B_zg3.36.1_axi_demo')
```
Git下载
```
# Git模型下载
git lfs install
git clone https://www.modelscope.cn/AIBS/Qwen2_0.5B_zg3.36.1_axi_demo.git
```

### 2.下载Icraft编译器安装包

适用于Icraft-v3.36.1，安装包请询问支持人员获取链接

### 3.环境部署

（1）win系统下安装Icraft（如果无需编译json&raw，可跳过该步骤）

进入安装包目录下，点击Icraft_Setup_v3.36.1.exe、CustomOp_Setup_v3.36.1.exe进行安装，安装好后可在终端输入`icraft --version` 打印版本信息

（2）板上安装Icraft（必需）

将安装包目录下的文件Icraft_3.36.1_onchip.deb、CustomOp_3.36.1_onchip.deb复制到板上，通过以下命令进行卸载旧版本和安装新版本：

```
dpkg -r icraft
dpkg -r customop 
dpkg -i Icraft_3.36.1_onchip.deb
dpkg -i CustomOp_3.36.1_onchip.deb
```

安装好后可在终端输入`icraft --version` 打印版本信息

（3）板上安装Icraft python api

将安装包目录下的`icraft-3.36.1-cp38-none-manylinux2014_aarch64.whl`文件复制到板上，执行以下命令进行安装：

```
pip install icraft-3.36.1-cp38-none-manylinux2014_aarch64.whl
```

（4）板子联网后，基于python3.8安装torch和transformer

```
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.46.3
```

## 3.Icraft编译（可跳过）

**目的**：使用Icraft编译器将实现保存好的**框架模型**转化为**硬件可部署模型**


**在 `compile`目录下执行编译：**

```shell
icraft compile Qwen2-0.5B_decoder.toml
```

如果过程顺利，将得到 icraft model（以 `.json` （graph）`.raw`（param）的格式保存）

其中包括编译各阶段产生的模型会被保存到：`./jr_path/decode/`路径下

## 4.运行工程

将本工程中所有文件复制到板上，然后在根目录下，直接执行`python3 qwen2_infer_decode_only_axi.py`，根据提示在终端输入文字可开始推理问答。

`ModelConfig`类中的关键参数说明：

- max_decode_length：最大解码长度，即KV缓存的最大序列长度（默认512，与模型配置文件中的max_seq_len一致）
- max_new_tokens：最大新生成token数量，达到该长度后停止生成（默认150）
- dtype_kv：KV缓存的数据类型
- run_backend：运行后端类型，如"zg330"、"host"等（默认"zg330"）
- ip：目标设备的IP地址（默认"192.168.125.143"）
- mmu_mode：是否启用MMU模式（默认True）
- cuda_mode：是否启用CUDA模式（默认False）
- vocab_size：词汇表大小
- sampling_method：采样方法，支持"greedy"、"top_k"、"top_p"（默认"greedy"）
- sampling_value：采样值，top_k或top_p的参数值（默认50）
- temperature：生成温度参数，控制生成多样性（默认1.0）
- repetition_penalty：重复惩罚参数，防止生成重复内容（默认1.2）



## 5.性能说明

Qwen2-0.5B模型在icraft 30TAI板子上纯端侧部署的实测效果如下：
<video controls src="qwen2-0.5B-30t-axi演示.mp4" title="Title"></video>
性能：

- 平均输出速度 = 总生成token数 / 总耗时 ≈ 4.68 tokens/s 

- 平均解码时间：199 ms

- 平均解码速度 = 1000 ms / 平均解码时间 ≈ 5.01 tokens/ms

  
