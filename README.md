# Qwen2-0.5B ZG330 demo（厂商镜像 ＋ 我方合并版 ＋ npuserv 适配器）

本仓库有两层身份，2026-11-06 起：

1. **出处**：ModelScope `AIBS/Qwen2_0.5B_zg3.36.1_axi_demo`（@`664a4c4`，厂商 FMSH 悟净
   FMQL30TAI／ZG330 的 Qwen2-0.5B 上板示例，自述适配 iCraft 3.36.1）的镜像。
2. **可用件**：厂商原样的 3 处依赖改动**已经合并**（以前是"留给你合并"，现在直接就是合并后的
   版本），并新增 `board/npuserv_adapter.py`——把这颗件接上 `lang/npuserv`，可以在板上
   `/load` → `/generate` → `/unload`。

板上实测（悟净 113，2026-11-06）：`/load` 137.4 s（`apply_s` 137.0），
`/generate {"text":"你是谁？","max_new":24}` 出 17 token／6.3 s，
文本 `我是阿里云开发的超大规模语言模型，我叫通义千问。`，`/unload` 后内存全部归还。

## 1. 与厂商原样的全部差异

| 文件 | 位置 | 厂商原样 | 本仓（已合并） |
|---|---|---|---|
| `qwen2_infer_decode_only_axi_pad3.py` | 导入段 | `from pyrtutils.calctime_utils import *` | 整行删除 |
| 同上 | 原 `:364`（prompt 循环） | `np.array([[1, i+1]], dtype=np.float32)` | `np.array([[1, i+1, 0]], dtype=np.float32)` |
| 同上 | 原 `:416`（decode 循环） | `np.array([[decode_seq_len, seq_pos]], dtype=np.float32)` | `np.array([[1, seq_pos, 0]], ...)` 同形 |
| `modelzoo_utils/pyrtutils/__init__.py` | 末行 | `from .calctime_utils import *` | 整行删除 |
| `modelzoo_utils/pyrtutils/utils/tools.py` | `:3` / `:6` | `import pandas as pd` / `from matplotlib import pyplot as plt` | 两行删除 |
| 驱动文件名 | — | `qwen2_infer_decode_only_axi.py` | 改名为 `qwen2_infer_decode_only_axi_pad3.py`（厂商原版不再保留，历史里可取） |

**为什么**：`calctime_utils`／`pandas`／`matplotlib` 在该文件里都没被使用（`calctime_detail`
只出现在两处注释；`tools.py` 只用 `is_arm()`／`tname()`）。不删的话板上（993 MB）要装
pandas＋matplotlib，且 matplotlib 首次 import 还会建字体缓存。
`token_dynamic_fm` 2→3 元素：ZG 件把该输入声明成 `fp32[1,3]`＝12 B，厂商驱动只喂 2 个值＝8 B，
板上真跑会撞 `hardop.cpp:46 host_chunk byte_size:8 is diff from plddr_chunk byte_size:12`
（本机 Host 后端不校验输入宽度，所以 PC 上永远看不出来）。第 3 个元素图里不读
（`[1,1,1]` 与 `[1,1,0]` 的 logits 逐位相同），补 `0` 不改变语义。

## 2. 文件与 md5

| 文件 | 字节 | md5 | 说明 |
|---|---|---|---|
| `qwen2_infer_decode_only_axi_pad3.py` | 23,549 | `597eeb3b164311f0882ff32e814f2b26` | 合并后的驱动（板上部署名 `/root/qwen2/qwen2_infer_pad3.py`，同字节） |
| `board/npuserv_adapter.py` | 9,288 | `668fc5d749bdc1d53846164acee89f78` | npuserv 适配器，见 §4 |
| `modelzoo_utils/pyrtutils/__init__.py` | 97 | `c6d5ad12c216dcb3a72c15203bcbcf2a` | 删过 import 的版本 |
| `modelzoo_utils/pyrtutils/utils/tools.py` | 365 | `77540af7c50db5366be437fc4a4ef7cf` | 删过 import 的版本 |
| `config/Qwen2-0___5B-Instruct/tokenizer.json` | 7,028,015 | `1c74fd33061313fafc6e2561d1ac3164` | 板端 `tokenizers` 0.20.3 读这颗（非 LFS 副本） |
| `config/Qwen2-0___5B-Instruct/config.json` | 659 | `ac22e24237a5c688f861b4aaf45df9a9` | 超参 |
| `Qwen2-0.5B_decoder.toml` | 1,060 | `7e7bdf8d4a4acfc519373e1233b08ad8` | `icraft compile` 的摄取约定 |

**不在本仓的件**（体积或 LFS）：

| 件 | 字节 | md5 | 哪来 |
|---|---|---|---|
| `Qwen2-0.5B-decoder_ZG.raw` | 1,314,012,420 | `2b107ec676de2b4dc35b772402b25b92` | `icraft compile` 产物；板上在 `/root/qwen2/jr_path/decode/` |
| `Qwen2-0.5B-decoder_ZG.json` | 2,183,817 | `24bf0b6b3d5990056fc66229a88f1020` | 同上；正本在 `repos/HA4FPAI/lang/llm/models/Qwen2/board/` |
| `fmodel/qwen2_exported_model_cuda/model.onnx.data` | 1.27 GB | — | 上游 LFS，`git lfs pull` |
| `fmodel/qwen2_exported_model_cuda/tokenizer.json` | 11.4 MB | `7875a23b…` | 上游 LFS；`config/…/tokenizer.json` 是同内容非 LFS 副本 |

## 3. 上板布局

板上的 base 是 `/root`（npuserv 在 `/root/npuserv`），契约要的是
`<base>/llm/models/<MODEL>/board/npuserv_adapter.py`。Qwen2 的件落在 `/root/qwen2/` 平铺，
所以：

```sh
# 1) 件与驱动（本仓内容）
/root/qwen2/{qwen2_infer_pad3.py, modelzoo_utils/, config/Qwen2-0___5B-Instruct/, jr_path/decode/…}
# 2) 适配器
/root/qwen2/board/npuserv_adapter.py
# 3) 让 registry 的相对路径解析得到
mkdir -p /root/llm/models
ln -s /root/qwen2 /root/llm/models/Qwen2
```

`jr_path/decode/` 里的 `.raw`（1.31 GB）与 `.json`（2.18 MB）不随本仓分发，按 §2 表取。

## 4. npuserv 注册与用法

`lang/npuserv/models.json` 里的那一条（registry 的 `root` 是 `lang/`，所以 `adapter` 写相对路径，
`artifacts` 写板上绝对路径）：

```json
{
  "id": "qwen2-0.5b",
  "adapter": "llm/models/Qwen2/board/npuserv_adapter.py",
  "compiled_with": "icraft 3.36.1 -- vendor's own 件",
  "artifacts": {
    "decode_json": "/root/qwen2/jr_path/decode/Qwen2-0.5B-decoder_ZG.json",
    "decode_raw": "/root/qwen2/jr_path/decode/Qwen2-0.5B-decoder_ZG.raw",
    "tokenizer": "/root/qwen2/config/Qwen2-0___5B-Instruct/tokenizer.json",
    "modelzoo_utils": "/root/qwen2/modelzoo_utils"
  },
  "defaults": {
    "json": "/root/qwen2/jr_path/decode/Qwen2-0.5B-decoder_ZG.json",
    "raw": "/root/qwen2/jr_path/decode/Qwen2-0.5B-decoder_ZG.raw",
    "config_path": "/root/qwen2/config/Qwen2-0___5B-Instruct",
    "modelzoo_utils": "/root/qwen2/modelzoo_utils",
    "driver_dir": "/root/qwen2",
    "max_new": 24,
    "stop_ids": [151643, 151645],
    "max_decode_length": 512
  }
}
```

```sh
python3 /root/npuserv/server.py --port 18082          # 18080 被 lang-server 的现役占着
curl -s localhost:18082/models | python3 -m json.tool  # 看 adapter_exists / artifacts_present
curl -s -XPOST localhost:18082/load     -d '{"model":"qwen2-0.5b"}'
curl -s -XPOST localhost:18082/generate -d '{"model":"qwen2-0.5b","text":"你是谁？","max_new":24}'
curl -s -XPOST localhost:18082/unload   -d '{"model":"qwen2-0.5b"}'
```

适配器做的事（全部按厂商驱动自己的两个入口走，不重写循环）：

| 阶段 | 调用 | `token_dynamic_fm` | 落点 |
|---|---|---|---|
| prefill | `process_prompt_character_by_character(ids)` | `[1, i+1, 0]` | 写回 `out[i+1]` → 抄进 slot `i` |
| decode | `run_decode(tok, kv, pos, generated)` | `[1, pos, 0]`，`pos = prompt_len + step - 1` | 不回抄（拿返回的新 cache） |

文本进出用板上 `tokenizers` 0.20.3 读 `config/…/tokenizer.json`，模板是厂商自己的
`<|im_start|>\n{text}<|im_end|>assistant\n`（不是 Qwen3 的 `|user` 形式）。停止词默认
`{151643, 151645}`（`<|endoftext|>`、`<|im_end|>`）；窗口是编译期的 512 槽，
`prompt + max_new` 超了适配器直接拒绝（报 `compiled window`），不会喂到窗口外。

## 5. 实测读数（悟净 113，2026-11-06）

| 项 | 读数 |
|---|---|
| `/load` | 137.4 s（`apply_s` 137.0；前一轮 137.5／136.8） |
| info | 24 层／2 KV head／head_dim 64／48 个 cache 张量／vocab 151,936／窗口 512 |
| `/generate`（`text`，max_new 24） | prompt 8 token → 17 token／6.3 s，`reason=eos` |
| 文本 | `我是阿里云开发的超大规模语言模型，我叫通义千问。` |
| 内存 | load 前 `MemAvailable 664,044 / CmaFree 47,004 kB`；load 后 `309,852 / 22,228`；unload 后 `662,800 / 23,372` |

同一颗件走板端自有 daemon 的对照读数：132.8 s 就绪、20 token／6.9 s／2.92 tok/s（台账 §卅八）。
两条路的装载时间和生成速率在同一量级；这次 npuserv 那轮 `max_new` 24 实际出 17 token，
速率 17÷6.3＝2.70 tok/s。

## 6. 出处 / Provenance（原样保留）

- **上游（镜像来源）**：[AIBS/Qwen2_0.5B_zg3.36.1_axi_demo @ ModelScope](https://www.modelscope.cn/AIBS/Qwen2_0.5B_zg3.36.1_axi_demo)
- **上游 commit**：`664a4c4`（"feat: 新增适用于v3.36.1的Qwen2_0.5B-axi上板示例"），本仓 2026-11-04 镜像
- **上游 LICENSE**：Apache-2.0（`Copyright 2024 Alibaba Cloud`），见本仓 `LICENSE`
  （与上游 `config/Qwen2-0___5B-Instruct/LICENSE` 同文）
- **正本**：`repos/HA4FPAI/lang/llm/models/Qwen2/`（同内容 ＋ 板端 daemon／桌面客户端／patch／
  `MANIFEST.md5`／完整记录 README）
- **台账**：`wiki/PS/板端我方代码与产物台账.md` §卅八（daemon 链与四次就绪）、§卅九（npuserv 接线）