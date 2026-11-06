#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""npuserv adapter: Qwen2-0.5B decoder (ZG330) on the board.

The driver is the vendor script itself -- `board/qwen2_infer_decode_only_axi_pad3.py`,
which is `qwen2_infer_decode_only_axi.py` with `pyrtutils` imports dropped and
`token_dynamic_fm` widened to 3 fp32 = 12 B (ZG declares `[1,3]`; the vendor fed 8 B and
the board rejects it -- 台账 #196).  This adapter only turns a npuserv request into that
driver's own two calls and back:

  prefill  session.process_prompt_character_by_character(ids)   tdf = [1, i+1, 0] per token
  decode   session.run_decode(tok, kv, pos, generated)          tdf = [1, pos,  0], pos = prompt_len + step - 1

Both position rules are the vendored ones, verbatim: prompt entries land at out[i+1] and
are copied down into slot i; the decode loop hands the returned cache back un-shifted.
That is the sequence the 11-06 chain answered coherent Chinese with (台账 §卅八: four turns,
2.75-2.97 tok/s, `我是阿里云开发的超大规模语言模型…`).

VERIFICATION STATUS (2026-11-06): the pieces exist and have been driven in this exact order
on 113 -- 132.8 s apply, 20 tokens in 6.9 s, `bye` releases the NPU (台账 §卅八 第四节).
What has NOT been run is this adapter under npuserv: the board-side wiring (registry entry,
`<base>/llm/models/Qwen2` link, worker spawn) lands with it, so the first `/load` is the
first proof.  The stub test in `npuserv/tests/q2_adapter_stub_check.py` covers the shape of
the calls on a PC, not any arithmetic.

Request (POST /generate):
    {"ids":[...], "max_new":24}
    {"text":"你是谁？", "max_new":24}      # wrapped in the vendor's chat template
Load params (all optional; registry defaults fill the board paths):
    json, raw, config_path, modelzoo_utils, driver_dir, max_decode_length, max_new,
    stop_ids, eos_id, sampling_method, sampling_value, temperature, repetition_penalty
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, os.pardir),
           os.environ.get("NPUSERV_HOME", "")):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, os.path.abspath(_p))

import adapter_lib                                             # noqa: E402

IM_START = "<" + "|im_start" + "|>"
IM_END = "<" + "|im_end" + "|>"
EOS_DEFAULT = 151643          # the vendor's hard-coded end id


class Q2Engine(object):
    name = "qwen2-0.5b"

    def __init__(self, cfg, model, sm, params, content_filter):
        self.cfg = cfg
        self.model = model
        self.sm = sm
        self.params = params
        self.filter = content_filter
        self.n_steps = 0
        self.stop_ids = set(int(t) for t in (cfg.stop_ids or (EOS_DEFAULT,)))
        # Geometry straight off the cache the driver allocated, not off a config number.
        probe = sm._prepare_kv_cache(batch_size=1, seq_len=int(cfg.max_decode_length))
        shape = list(probe[0].shape)
        self.info = {
            "kv_depth_in_config": int(cfg.max_decode_length),
            "cache_tensors": len(probe), "cache_shape": shape,
            "n_layers": len(probe) // 2, "n_kv_heads": int(shape[1]),
            "head_dim": int(shape[3]), "vocab_size": int(cfg.vocab_size),
            "stop_ids": sorted(self.stop_ids),
            "sampling": {"method": cfg.sampling_method, "value": cfg.sampling_value,
                         "temperature": cfg.temperature,
                         "repetition_penalty": cfg.repetition_penalty},
            "position_rule": "prompt tdf=[1,i+1,0] out[i+1]->slot i; decode tdf=[1,pos,0], pos=prompt_len+step-1",
            "read": "whole (driver Tensor2Numpy, no narrow readback)",
            "json": cfg.decode_json, "raw": cfg.decode_raw,
            "tokenizer": os.path.join(cfg.config_path, "tokenizer.json"),
            "template": IM_START + "\\n{text}" + IM_END + "assistant\\n",
            "verified": "driver ran on 113 at 132.8 s / 20 tokens (台账 §卅八); "
                        "this adapter itself has not been through npuserv yet",
        }

    def template(self, text):
        # Same wrapper as the vendored driver's generate() -- `<|im_start|>\n{text}<|im_end|>assistant\n`
        # (Qwen2's own template, not Qwen3's `|user` form); it costs 4 tokens and is the
        # only reason the answers come back as an assistant turn.
        return IM_START + "\n" + str(text) + IM_END + "assistant\n"

    def generate(self, req, emit):
        ids = [int(t) for t in req.get("ids") or []]
        if not ids and req.get("text"):
            ids = list(self.model.tokenizer.encode(self.template(req["text"])).ids)
        if not ids:
            raise ValueError("ids or text required")
        depth = int(self.cfg.max_decode_length)
        max_new = int(req.get("max_new", self.params.get("max_new", 24)))
        if len(ids) >= depth:
            raise ValueError("prompt is %d tokens, the compiled window is %d" % (len(ids), depth))
        max_new = min(max_new, depth - len(ids))

        t0 = time.time()
        first, kv = self.sm.process_prompt_character_by_character(ids)
        emit({"pre": len(ids), "n": len(ids), "sec": round(time.time() - t0, 2)})

        gen = []
        cur = int(first)
        step = 1
        reason = "limit"
        while True:
            if cur in self.stop_ids:
                reason = "eos"
                break
            if len(gen) >= max_new:
                reason = "limit"
                break
            if len(ids) + step - 1 >= depth:
                reason = "full"
                break
            gen.append(cur)
            emit({"tok": cur, "i": len(gen) - 1, "sec": round(time.time() - t0, 2)})
            if len(gen) > 20:
                try:
                    if self.filter.should_stop_generation(gen, self.model.tokenizer, step):
                        reason = "filter"
                        break
                except Exception as e:                 # the filter must never end a turn
                    adapter_lib.log("content filter raised %r -- ignored" % (e,))
            # vendored call order: the token just emitted is the input, `generated`
            # passed for the repetition penalty is the history before it
            cur, kv = self.sm.run_decode(cur, kv, len(ids) + step - 1, gen[:-1])
            step += 1
            self.n_steps += 1

        out = {"ids": gen, "reason": reason, "prompt": len(ids), "gen": len(gen),
               "sec": round(time.time() - t0, 1), "model": self.name,
               "last_pos": len(ids) + step - 2}
        if req.get("decode", True):
            try:
                out["text"] = self.model.tokenizer.decode(gen)
            except Exception as e:
                out["text"] = None
                out["decode_error"] = repr(e)
        return out

    def close(self):
        # Session memory returns when this process exits (台账 §册三); the driver has no
        # release call and reset_device() is a per-step no-op, so nothing to do here.
        pass


def load(params):
    for key in ("driver_dir", "modelzoo_utils"):
        alt = params.get(key)
        if alt and os.path.isdir(alt) and alt not in sys.path:
            sys.path.insert(0, os.path.abspath(alt))
    ic39 = os.path.join(os.path.sep, "root", "icraft39")
    if os.path.isdir(ic39) and ic39 not in sys.path:
        sys.path.insert(0, ic39)              # 3.39.0 bindings ahead of the old pip one

    import numpy as np
    # Two names for the same 23.5 KB file: the repo carries it as
    # `qwen2_infer_decode_only_axi_pad3.py` (vendor script name + the pad3 suffix), while the
    # board's deployment on 11-06 landed it as `/root/qwen2/qwen2_infer_pad3.py` because the
    # chat daemon imports it that way.  Same md5 597eeb3b...; accept either.
    try:
        import qwen2_infer_decode_only_axi_pad3 as Q2
    except ImportError:
        import qwen2_infer_pad3 as Q2

    class Cfg(Q2.ModelConfig):
        pass

    Cfg.config_path = params.get("config_path", Cfg.config_path)
    Cfg.decode_json = params.get("json", Cfg.decode_json)
    Cfg.decode_raw = params.get("raw", Cfg.decode_raw)
    Cfg.max_decode_length = int(params.get("max_decode_length", Cfg.max_decode_length))
    Cfg.max_new_tokens = int(params.get("max_new", Cfg.max_new_tokens))
    Cfg.vocab_size = int(params.get("vocab_size", Cfg.vocab_size))
    Cfg.stop_ids = tuple(int(t) for t in params.get("stop_ids", (EOS_DEFAULT,)))
    Cfg.dtype_kv = {"fp16": np.float16, "fp32": np.float32}.get(
        params.get("dtype_kv", "fp16"), np.float16)
    Cfg.sampling_method = params.get("sampling_method", Cfg.sampling_method)
    Cfg.sampling_value = float(params.get("sampling_value", Cfg.sampling_value))
    Cfg.temperature = float(params.get("temperature", Cfg.temperature))
    Cfg.repetition_penalty = float(params.get("repetition_penalty", Cfg.repetition_penalty))

    cfg = Cfg()
    adapter_lib.log("loading tokenizer + device + decoder net for Qwen2-0.5B")
    model = Q2.IcraftModel(cfg)
    sm = Q2.DecodeOnlySessionManager(model)      # load_model + init_device + apply
    return Q2Engine(cfg, model, sm, params, Q2.ContentFilter())


if __name__ == "__main__":
    sys.exit(adapter_lib.run(load))