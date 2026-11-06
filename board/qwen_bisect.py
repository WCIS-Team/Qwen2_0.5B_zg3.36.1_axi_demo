#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen2 点火 bisect：把"起跑就整机挂死"拆成 4 步，逐步加码。

为什么拆：2026-10-11 20:17 那次板子整机挂死，journald 缓冲一行没落盘，
所以只知道"脚本启动后 30 秒内死了"，不知道死在哪一步。
把 openDevice → 建图 → 读 1.22 GiB 权重 → session.apply() 拆开单跑，
哪一步挂的就是哪一步；每步各自独立进程，前一步的结论不会被后一步的挂死吃掉。

用法（板上）：
    cap.sh --as 2G --tag qb_s1 -- python3 /root/qwen2/qwen_bisect.py 1
    cap.sh --as 2G --tag qb_s2 -- python3 /root/qwen2/qwen_bisect.py 2
    cap.sh --as 2G --tag qb_s3 -- python3 /root/qwen2/qwen_bisect.py 3
    cap.sh --as 2G --tag qb_s4 -- python3 /root/qwen2/qwen_bisect.py 4
或一次跑完（每步独立进程，挂了就停在那一步）：
    bash /root/qwen2/qwen_bisect.sh

每步结束都打一行 STEP<n> OK <耗时>s <VmSize> <VmRSS>，写 /root/qwen2/bisect.log。
"""

import json
import os
import sys
import time

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
JSON = "jr_path/decode/Qwen2-0.5B-decoder_ZG.json"
RAW = "jr_path/decode/Qwen2-0.5B-decoder_ZG.raw"
BACKEND = "zg330"
NPU = "0x40000000"
DMA = "0x80000000"


def mem():
    d = {}
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(("VmSize:", "VmRSS:", "VmHWM:")):
                k, v = line.split(":", 1)
                d[k] = v.strip()
    return d


def beat(tag, t0):
    m = mem()
    print("  [%s] t=%.1fs %s %s %s" % (tag, time.time() - t0, m.get("VmSize"), m.get("VmRSS"), m.get("VmHWM")))
    sys.stdout.flush()


def done(n, t0):
    m = mem()
    line = "STEP%d OK %.1fs %s %s %s" % (n, time.time() - t0, m.get("VmSize"), m.get("VmRSS"), m.get("VmHWM"))
    print(line)
    sys.stdout.flush()
    with open("/root/qwen2/bisect.log", "a") as f:
        f.write(line + "\n")
    sys.exit(0)


def main():
    t0 = time.time()
    beat("start", t0)

    # ---- 所有步骤都要的 import（这一步本身也会 import icraft，失败也有信息） ----
    import icraft
    from icraft import xir
    sys.path.append("./modelzoo_utils")
    from pyrtutils.icraft_utils import openDevice, initSession
    beat("import", t0)
    if STEP == 0:
        done(0, t0)

    # ---- 步骤 1：只开设备。aarch64 上走 axi:// 直通物理地址，无内核驱动、无 /dev 节点 ----
    print("STEP1 openDevice(%s, npu=%s dma=%s)" % (BACKEND, NPU, DMA))
    sys.stdout.flush()
    device = openDevice(BACKEND, "192.168.125.143", True, False)
    beat("after-openDevice", t0)
    done(1, t0) if STEP == 1 else None

    # ---- 步骤 2：建图（纯读 json，1823 个 op）----
    print("STEP2 CreateFromJsonFile(%s)" % JSON)
    sys.stdout.flush()
    network = icraft.xir.Network.CreateFromJsonFile(JSON)
    print("  ops=%d" % len(network.ops))
    sys.stdout.flush()
    beat("after-createJson", t0)
    done(2, t0) if STEP == 2 else None

    # ---- 步骤 3：读 1.22 GiB fp16 权重（suspect #1）----
    print("STEP3 lazyLoadParamsFromFile(%s, %d bytes)" % (RAW, os.path.getsize(RAW)))
    sys.stdout.flush()
    network.lazyLoadParamsFromFile(RAW)
    beat("after-lazyLoad", t0)
    done(3, t0) if STEP == 3 else None

    # ---- 步骤 4：建 session 并 apply = 真正给 NPU 编程（suspect #2）----
    print("STEP4 initSession + apply")
    sys.stdout.flush()
    session = initSession(BACKEND, network, device, 1)
    beat("after-initSession", t0)
    session.apply()
    beat("after-apply", t0)
    print("  device=%s" % type(device).__name__)
    done(4, t0)


if __name__ == "__main__":
    main()