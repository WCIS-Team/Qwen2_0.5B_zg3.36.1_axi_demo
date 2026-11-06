# 权重与编译件从哪来（WEIGHTS）

**TL;DR**：本仓不装大件。要在板上跑起来，只需要补**一颗** `Qwen2-0.5B-decoder_ZG.raw`（1,314,012,420 B，
md5 `2b107ec676de2b4dc35b772402b25b92`）——重编，或从板上／本机盘拷；其余（配置、分词器、驱动、适配器、
`.json` 图描述）都在仓里。上游原始模型权重的出处与校验见 §1。

## 1. 上游原始模型（这一份是"出处"，推理不直接读它）

厂商 demo 用的权重是 **Qwen2-0.5B-Instruct**，在仓里只留了一个 LFS 指针：

```
config/Qwen2-0___5B-Instruct/model.safetensors   (134 B)
version https://git-lfs.github.com/spec/v1
oid sha256:130282af0dfa9fe5840737cc49a0d339d06075f83c5a315c3372c9a0740d0b96
size 988097824
```

取法（任选其一，拿到的应是同一份 988,097,824 B）：

| 路子 | 命令／地址 |
|---|---|
| 从本仓的 LFS 指针拉 | `git lfs install && git lfs pull` —— 上游 demo 仓里就带着这份 |
| 上游 demo（ModelScope） | `git clone https://www.modelscope.cn/AIBS/Qwen2_0.5B_zg3.36.1_axi_demo.git`（厂商 README 原话），或 `snapshot_download('AIBS/Qwen2_0.5B_zg3.36.1_axi_demo')` |
| 识别上游模型本体 | Hugging Face `Qwen/Qwen2-0.5B-Instruct`（apache-2.0；`config/…/README.md` 的 frontmatter 也是这个许可） |

板端推理**不走** safetensors：走的是 `fmodel/` 的 ONNX（仅用于导出）和下面 §2 的 ZG 编译件。

## 2. 板端真正加载的件

| 件 | 字节 | md5 | 在仓里？ | 怎么拿 |
|---|---|---|---|---|
| `Qwen2-0.5B-decoder_ZG.json` | 2,183,817 | `24bf0b6b3d5990056fc66229a88f1020` | **不在** | 与 `.raw` 同一次 `icraft compile` 的两个产物；或从板 `/root/qwen2/jr_path/decode/` 拷 |
| `Qwen2-0.5B-decoder_ZG.raw` | 1,314,012,420 | `2b107ec676de2b4dc35b772402b25b92` | **不在**（体积 ＋ 属工具链输出，不发布） | ① 重编：`icraft compile Qwen2-0.5B_decoder.toml`（iCraft 3.39.0，本机 Windows CLI）；② 从板 `/root/qwen2/jr_path/decode/` 拷；③ 本机盘 `S:\apps\llm\agentwork\qwen\repo\jr_path\decode\`（2026-10-11 20:04 那份） |
| `config/Qwen2-0___5B-Instruct/tokenizer.json` | 7,028,015 | `1c74fd33061313fafc6e2561d1ac3164` | 在 | 直接用（板端 `tokenizers` 0.20.3 读） |
| `qwen2_infer_decode_only_axi_pad3.py` | 23,549 | `597eeb3b164311f0882ff32e814f2b26` | 在 | 直接用（板上部署名 `qwen2_infer_pad3.py`） |
| `board/npuserv_adapter.py` | 9,293 | `a5a1767a95b242f42e6dd1d82dbe8b71` | 在 | 直接用 |

**重编没做过的验证**：从没跑两次 `icraft compile` 比过 md5，所以"重编出来字节一致"这条**未验证**；
要稳就用 ② 或 ③ 那份（md5 与板上 `md5sum` 对得上）。

## 3. 不在仓里的其余大件

| 件 | 字节 | 为什么不在 | 怎么拿 |
|---|---|---|---|
| `fmodel/qwen2_exported_model_cuda/model.onnx.data` | 1,266,188,288 | 体积／LFS | 上游 demo 仓 `git lfs pull`；**只在重新导出 ONNX 时才用得到，跑板端不需要** |
| `fmodel/qwen2_exported_model_cuda/tokenizer.json` | 11.4 MB | 与 `config/…/tokenizer.json` 同内容 | 同上，或用 `config/` 那颗 |

## 4. 校验

`MANIFEST.md5` 是本仓入库件的逐件校验表（`<md5>  ./<路径>`，100 行量级）。自己核：

```sh
cd <本仓>
python3 - <<'PY'
import hashlib, os
bad = 0
for line in open("MANIFEST.md5", encoding="utf-8"):
    if not line.strip():
        continue
    md5, path = line.rstrip("\n").split("  ", 1)
    p = path.lstrip("./")
    if not os.path.exists(p):
        print("MISSING", p); bad += 1; continue
    h = hashlib.md5(open(p, "rb").read()).hexdigest()
    if h != md5:
        print("MISMATCH", p, md5, h); bad += 1
print("bad =", bad)
PY
```

（`config/…/model.safetensors` 是 LFS 指针，没 `git lfs pull` 时它自身的 md5 就是指针文本的 md5，属正常。）

## 5. 出处汇总

| 是什么 | 在哪 |
|---|---|
| 上游 demo（厂商 FMSH 悟净／ZG330，自述 iCraft 3.36.1） | ModelScope `AIBS/Qwen2_0.5B_zg3.36.1_axi_demo` @ `664a4c4` |
| 本仓 | `https://github.com/WCIS-Team/Qwen2_0.5B_zg3.36.1_axi_demo`（厂商件 ＋ 我方 6 行改动已合并 ＋ npuserv 适配器） |
| 正本／台账 | WCIS 仓 `repos/HA4FPAI` 里本目录是子模块；`wiki/PS/板端我方代码与产物台账.md` §卅八（对话链）／§卅九（npuserv 接线） |
| 编译主机 | 只能本机 Windows iCraft CLI（3.39.0）；板（993 MB）上不许编译 |