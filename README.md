# Qwen2-0.5B ZG330 demo（厂商 + 我方上板改动）

从 ModelScope `AIBS/Qwen2_0.5B_zg3.36.1_axi_demo`(@664a4c4) 镜像，并合入我方为悟净 FMQL30TAI676H(ZG330) 上板所做的 3 处共 6 行改动：

1. `qwen2_infer_decode_only_axi.py`：删 `from pyrtutils.calctime_utils import *`；`token_dynamic_fm` 补成 fp32[1,3]（2→3 元素），修 ZG 件 12 B vs 厂商驱动 8 B 的喂料宽度坑。
2. `modelzoo_utils/pyrtutils/__init__.py`：删 `from .calctime_utils import *`。
3. `modelzoo_utils/pyrtutils/utils/tools.py`：删 `import pandas` / `from matplotlib import pyplot`（该文件未使用，免去板上装 pandas+matplotlib）。

## 排除件（未入库）

- `fmodel/qwen2_exported_model_cuda/model.onnx.data`（1.27 GB，LFS）——可从源仓库 `git lfs pull` 重新取得。
- `fmodel/qwen2_exported_model_cuda/tokenizer.json`（11.4 MB，LFS）——`config/Qwen2-0___5B-Instruct/tokenizer.json`(7.0 MB) 是同内容非 LFS 副本，无损失。

编译器用本机 iCraft 3.39.0（厂商自述 3.36.1）。完整记录见 `repos/lang/llm/Qwen2/README.md`。
