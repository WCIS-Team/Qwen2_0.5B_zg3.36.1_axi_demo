# Qwen2-0.5B ZG330 demo（厂商原样镜像）

本仓库是 ModelScope `AIBS/Qwen2_0.5B_zg3.36.1_axi_demo`(@664a4c4) 的**原样镜像**——除下方「改动清单」外，3 个文件**保持厂商字节原样**，我方改动**未预合并**，由你合并。合并后即等于 `repos/lang/llm/Qwen2`（WCIS 项目内那份）。

## 改动清单（3 个文件、6 行；本仓库尚未合并）

| 文件 | 位置 | 厂商原样（before） | 我方（after） |
|---|---|---|---|
| `qwen2_infer_decode_only_axi.py` | `:16` | `from pyrtutils.calctime_utils import *` | （整行删除） |
| `qwen2_infer_decode_only_axi.py` | `:364` | `token_dynamic_fm = np.array([[1, i+1]], dtype=np.float32)` | `np.array([[1, i+1, 0]], dtype=np.float32)` |
| `qwen2_infer_decode_only_axi.py` | `:416` | `token_dynamic_fm = np.array([[decode_seq_len, seq_pos]], dtype=np.float32)` | `np.array([[decode_seq_len, seq_pos, 0]], dtype=np.float32)` |
| `modelzoo_utils/pyrtutils/__init__.py` | 末行 | `from .calctime_utils import *` | （整行删除） |
| `modelzoo_utils/pyrtutils/utils/tools.py` | `:3` | `import pandas as pd` | （整行删除） |
| `modelzoo_utils/pyrtutils/utils/tools.py` | `:6` | `from matplotlib import pyplot as plt` | （整行删除） |

**为什么：**

- `calctime_utils`/`pandas`/`matplotlib` 三样在该文件里都没被使用（`calctime_detail` 只出现在驱动注释 `:373`/`:423`；`tools.py` 只有 `is_arm()`/`tname()`），不删的话板上（993 MB）得装 pandas＋matplotlib，且 `matplotlib` 首次 import 还要建字体缓存。
- `token_dynamic_fm` 2→3 元素：ZG 件把该输入声明成 `fp32[1,3]`＝12 B，而厂商驱动只喂 2 个值＝8 B，板上真跑会撞 `hardop.cpp:46 host_chunk byte_size:8 is diff from plddr_chunk byte_size:12`；本机 Host 后端不校验输入宽度，所以 PC 上永远看不出来。第 3 个元素图里不读（`[1,1,1]` 与 `[1,1,0]` 的 logits 逐位相同），补 `0` 是纯占位、不改语义。

**合并文件的正本（已完成上述改动的版本）**：

- 驱动（含全部 3 行）＝ `repos/lang/llm/Qwen2/board/qwen2_infer_decode_only_axi_pad3.py`（md5 `597eeb3b164311f0882ff32e814f2b26`）
- `__init__.py`＝ `repos/lang/llm/Qwen2/patch/modelzoo_utils/pyrtutils/__init__.py`（md5 `c6d5ad12c216dcb3a72c15203bcbcf2a`）
- `tools.py`＝ `repos/lang/llm/Qwen2/patch/modelzoo_utils/pyrtutils/utils/tools.py`（md5 `77540af7c50db5366be437fc4a4ef7cf`）
- unified diff：`repos/lang/llm/Qwen2/patch/qwen2_infer_162.diff`、`qwen2_infer_pad3_196.diff`、`modelzoo_pyrtutils_init.diff`、`modelzoo_utils_tools.diff`
- 完整记录：`repos/lang/llm/Qwen2/README.md`

## 排除件（未入库）

- `fmodel/qwen2_exported_model_cuda/model.onnx.data`（1.27 GB，LFS）——可从源仓库 `git lfs pull` 重新取得。
- `fmodel/qwen2_exported_model_cuda/tokenizer.json`（11.4 MB，LFS）——`config/Qwen2-0___5B-Instruct/tokenizer.json`(7.0 MB) 是同内容非 LFS 副本，无损失。

编译器用本机 iCraft 3.39.0（厂商自述 3.36.1）。

## 出处 / Provenance

- **上游（镜像来源）**：[AIBS/Qwen2_0.5B_zg3.36.1_axi_demo @ ModelScope](https://www.modelscope.cn/AIBS/Qwen2_0.5B_zg3.36.1_axi_demo) —— 厂商 FMSH 悟净 FMQL30TAI(ZG330) Qwen2-0.5B 上板示例，自述适配 iCraft 3.36.1
- **上游 commit**：`664a4c4`（"feat: 新增适用于v3.36.1的Qwen2_0.5B-axi上板示例"），本仓于 2026-11-04 镜像
- **本仓现状**：厂商原样镜像；我方改动未合并，见上方「改动清单」
- **上游 LICENSE**：Apache-2.0（`Copyright 2024 Alibaba Cloud`），见本仓根目录 `LICENSE`（与上游 `config/Qwen2-0___5B-Instruct/LICENSE` 同文）
- **完整记录**：`repos/lang/llm/Qwen2/README.md`（WCIS 项目内）
