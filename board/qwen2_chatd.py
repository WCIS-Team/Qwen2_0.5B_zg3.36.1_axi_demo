# -*- coding: utf-8 -*-
"""Board-side chat daemon for the vendor Qwen2-0.5B ZG330 demo.

Why a TCP server instead of stdin/stdout pipes (the v1 shape, measured 11-06):
the model needs 139.5-147.8 s to load (four runs that reached ready, 03:57:50 /
07:52:19 / 08:23:53 / 08:28:36 board CST), and a pipe transport dies with whatever
shell started it -- an SSH blip during those two and a half minutes threw away the
whole load and left nothing to inspect. The port is
bound before the load starts, so a client
can attach immediately, watch the phase, and the process is owned by no terminal.

Protocol, one JSON object per line:
  in : {"ping": true}
  in : {"text": "...", "max_new": 24}
  out: {"event":"hello","phase":"loading|ready","elapsed":..,"eta":..}   (on connect)
  out: {"event":"ready","load_secs":..,"mem":{..}}                       (pushed once)
  out: {"event":"pong","phase":..,"busy":bool}
  out: {"ok":true,"answer":"..","secs":..,"max_new":..,"gen_tokens":..,"prompt_tokens":..,"tok_s":..}
  out: {"ok":false,"error":".."}

Lifetimes: exits by itself IDLE_S after the last request (so the 256 MiB CMA pool
returns to the video chain), and exits TURN_DEADLINE_S into a turn if the NPU ever
hangs mid-step -- a wedged process holding CMA is worse than a dead one.
"""
import contextlib
import io
import json
import os
import signal
import socket
import sys
import threading
import time

PORT = int(os.environ.get("Q2_PORT", "18099"))
BIND = os.environ.get("Q2_BIND", "0.0.0.0")
IDLE_S = float(os.environ.get("Q2_IDLE_S", "240"))
LOAD_ETA_S = float(os.environ.get("Q2_LOAD_ETA_S", "150"))  # measured 139.5/144.1/147.8/143.0 s on 11-06
TURN_DEADLINE_S = float(os.environ.get("Q2_TURN_DEADLINE_S", "900"))
# VmHWM of the same bisect STEP4 (initSession+apply) measured twice: 673,848 kB on
# 10-14 (ledger #197) and 447,104 kB on 11-06 (/root/qwen2/bisect.log 01:12). VmSize
# matched to within 140 kB, RSS differed by 269 MB -- cause not established.
# The gate is NOT peak x 1.15. 11-06 has 20 loads that started listening
# (/root/qwen2/chatd.log, board CST): the 16 whose start MemAvailable fell in
# 402,160-413,576 kB never reached ready, and the 4 that started at 511,240 /
# 492,036 / 498,324 / 507,232 kB all did (144.1 / 139.5 / 147.8 / 143.0 s). A gate
# of 436 MB x 1.15 = 507 MB would have refused three of those four runs, which all
# worked, so NEED_KB is taken inside the empty band between the proven failures and
# the proven successes.
PEAK_KB = int(os.environ.get("Q2_PEAK_KB", "673848"))
NEED_KB = int(os.environ.get("Q2_NEED_KB", "470000"))   # 413,576 (top of the failure band) < 470,000 < 492,036 (lowest success)
PROGNOSIS_KB = int(os.environ.get("Q2_PROG_KB", "492036"))  # the lowest start MemAvailable that actually reached ready
PEAK2_KB = int(os.environ.get("Q2_PEAK2_KB", "447104"))
FLOOR_KB = int(os.environ.get("Q2_FLOOR_KB", "55000"))     # breaker on MemAvailable (in-load bottom 57,680; never reached ready)
NONCMA_FLOOR_KB = int(os.environ.get("Q2_NONCMA_FLOOR_KB", "22600"))  # MemFree - CmaFree, the quantity the kernel watermark actually compares
LOAD_DEADLINE_S = float(os.environ.get("Q2_LOAD_DEADLINE_S", "600"))
GRACE_S = float(os.environ.get("Q2_GRACE_S", "150"))   # how long a refused daemon keeps answering
KV_WINDOW = 512          # ModelConfig.max_decode_length, burned into the imodel
SAFETY_SLOTS = 8         # leave headroom so a request never runs off the window

here = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(here, "chatd.state")
READY_FLAG = os.path.join(here, "chatd.ready")
if here not in sys.path:
    sys.path.insert(0, here)
os.chdir(here)

# site-packages carries icraft 3.33.1, whose zg330backend fails to initialise
# (ImportError: initialization failed). The 3.39 tree wins by going first.
if os.path.isdir("/root/icraft39"):
    sys.path.insert(0, "/root/icraft39")

import qwen2_infer_pad3 as m

LOCK = threading.Lock()          # one turn at a time -- the NPU has one slot
CLIENTS = []                      # live sockets, for pushing state changes
GEN = None
PHASE = "loading"                 # loading -> ready (or died)
LOAD_SECS = None
LOAD_T0 = time.time()
BUSY_SINCE = [None]
LAST_ACTIVITY = [time.time()]
FATAL = None                     # payload replayed to late clients while we wind down
PROGNOSIS = None                 # set by load(); the client connects ~6s after we start,
                                 # so the odds line travels in hello(), not a broadcast


def log(msg):
    print(time.strftime("[%m-%d %H:%M:%S] ") + str(msg), flush=True)


# Coexistence: this board shares one 256 MiB CMA pool between the NPU runtimes and
# the SDI/VPU chain. Raise our own OOM score so that if RAM runs short, this process
# is the one that dies -- not somebody else's running pipeline.
# No RLIMIT_AS here: the iCraft runtime mallocs a 272 MB host-region arena for the
# Gather op, and a 2 GiB address-space cap makes that malloc fail (measured 11-06).
try:
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("300")
except Exception as e:
    log("oom_score_adj skipped: %r" % (e,))


def memline():
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k = line.split(":", 1)[0]
                if k in ("CmaFree", "MemAvailable", "MemFree"):
                    out[k] = int(line.split()[1])
    except Exception:
        pass
    return out


def set_state(text):
    try:
        with open(STATE, "w") as f:
            f.write(text + "\n")
    except Exception as e:
        log("state write failed: %r" % (e,))


def broadcast(obj):
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    dead = []
    for s in CLIENTS:
        try:
            s.sendall(data)
        except OSError:
            dead.append(s)
    for s in dead:
        try:
            CLIENTS.remove(s)
        except ValueError:
            pass
        try:
            s.close()
        except OSError:
            pass


def hello(sock):
    send_one(sock, {"event": "hello", "phase": PHASE, "model": "Qwen2-0.5B ZG330",
                    "load_secs": LOAD_SECS, "elapsed": round(time.time() - LOAD_T0, 1),
                    "eta": LOAD_ETA_S, "kv_window": KV_WINDOW, "busy": BUSY_SINCE[0] is not None,
                    "prognosis": PROGNOSIS, "mem": memline()})


def send_one(sock, obj):
    try:
        sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    except OSError:
        pass


def topmem(n=4):
    """Who is holding the RAM, straight out of /proc -- no subprocess, no ps."""
    rows = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/status" % pid) as f:
                comm = rss = None
                for line in f:
                    if line.startswith("Name:"):
                        comm = line.split(":", 1)[1].strip()
                    elif line.startswith("VmRSS:"):
                        rss = int(line.split()[1])
            if rss:
                rows.append((rss, pid, comm))
        except (OSError, ValueError):
            continue
    rows.sort(reverse=True)
    return [{"pid": p, "comm": c, "rss_kb": r} for r, p, c in rows[:n]]


def meminfo_kb(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    return -1


def preflight():
    return meminfo_kb("MemAvailable")


def arm_breaker():
    """A load that runs this board out of memory wedges sshd and every co-resident
    service for tens of minutes, and the thread that would check cannot run because
    icraft holds the GIL. Fork a monitor before any thread exists; it kills this pid.

    What 11-06 measured, straight from dmesg and mem_watch-1106.log (zone DMA32 is
    the only memory zone, and /proc/meminfo MemFree counts CMA pages, so the number
    the watermark check actually uses is MemFree - CmaFree; min = 22,528 kB):
      * all three OOM kills (01:47:11 / 02:07:46 / 02:19:12 board CST): free
        120,368/120,600/126,568 kB minus free_cma 98,916/98,376/104,372 kB =
        21,452 / 22,224 / 22,196 kB -- all under min, while MemAvailable still read
        87-106 MB and the victims' anon-rss was 344,592 / 349,516 / 349,524 kB.
        Raw MemFree looked healthy; it was not. The dmesg buffer still starts at
        boot (11-04 14:41:59 CST), so "three kills" is the whole story for this boot.
      * the four loads that reached ready bottomed, at the READY reading itself, at
        MemFree-CmaFree 32,524-43,380 kB (04:00:14 / 07:54:39 / 08:26:21 / 08:30:59)
        -- all above the 22,600 floor.
      * FLOOR_KB (MemAvailable 55,000) never fires on a load that can make it: the
        03:57:50 -> 04:00:14 window bottoms at 90,424 kB avail (mem_watch-1106c,
        19:58-20:04 UTC). A 120,000 rule would have fired inside that window -- it
        is what killed the 02:14:12 attempt (breaker.log: avail_kb=102,600).
    So the floor sits above every value the kernel killed at (22,600 > 22,224) and
    below the low-water mark of both loads that made it (22,600 < 35,816) -- a 13 MB
    margin on the success side and a 376 kB margin on the kill side, i.e. on a
    marginal load this is still a race we may lose. Killing ourselves costs one
    reload; the kernel picking the moment costs her whole chain, so the race is
    worth running. The other trip is the deadline: the two measured successes took
    144.1 s and 139.5 s, and tonight's attempt that never reached ready was still
    running 843 s after it started listening (01:33:08 -> killed 01:47:11) when the
    kernel took it. 600 s is 4.16x the slower success (600/144.1). Once the model is ready the
    deadline is off -- the marker is the chatd.ready flag, not chatd.state, because
    state reads "turn pid=..." for the whole of every turn and gating on it would
    kill a healthy daemon mid-answer. The floors stay armed for life, because at
    that point we are the largest holder on the board."""
    me = os.getpid()
    if os.fork() == 0:
        path = os.path.join(here, "breaker.log")
        while True:
            if not os.path.isdir("/proc/%d" % me):
                os._exit(0)
            free = meminfo_kb("MemFree")
            cma = meminfo_kb("CmaFree")
            avail = meminfo_kb("MemAvailable")
            noncma = free - cma if free >= 0 and cma >= 0 else -1
            trip = None
            if 0 <= noncma < NONCMA_FLOOR_KB:
                trip = "noncma_free_kb=%d (MemFree %d - CmaFree %d) floor=%d" % (noncma, free, cma, NONCMA_FLOOR_KB)
            elif 0 <= avail < FLOOR_KB:
                trip = "avail_kb=%d floor=%d" % (avail, FLOOR_KB)
            elif not os.path.exists(READY_FLAG) and time.time() - LOAD_T0 > LOAD_DEADLINE_S:
                trip = ("load_deadline_s=%d elapsed=%.0fs"
                        % (LOAD_DEADLINE_S, time.time() - LOAD_T0))
            if trip:
                stamp = time.strftime("%m-%d %H:%M:%S")
                try:
                    with open(path, "a", encoding="utf-8") as f:
                        f.write("%s BREAKER %s -> kill %d\n" % (stamp, trip, me))
                except OSError:
                    pass
                os.kill(me, signal.SIGKILL)
                # Kill first, then write: the parent cannot touch the file once it
                # is dead, so chatd.state stops lying about a live "loading pid=..."
                # (it did, 04:31:15 -> "loading pid=263482" for the next 15 min).
                set_state("dead breaker %s" % trip)
                try:
                    os.remove(READY_FLAG)
                except OSError:
                    pass
                os._exit(0)
            time.sleep(1.5)
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)


def load():
    global GEN, PHASE, LOAD_SECS, FATAL, PROGNOSIS
    avail = preflight()
    if avail < NEED_KB:
        free = meminfo_kb("MemFree")
        msg = ("板端可用内存 %d MB，门禁 %d MB。这条线是 11-06 实测出来的，不是推的："
               "那天 20 次起跑（/root/qwen2/chatd.log，板钟 CST），起点可用内存落在 "
               "402,160~413,576 kB 的 16 次一次都没走到就绪——这 16 次里只有 4 次留下了死因："
               "内核 OOM 杀 3 次（dmesg 01:47:11／02:07:46／02:19:12，死时 MemFree−CmaFree "
               "21,452／22,224／22,196 kB，都已跌破水位线 min 22,528 kB；MemFree 里那 "
               "98~104 MB 是 CMA，不算数），断路器自杀 1 次（04:31:15）；其余 12 次既没就绪"
               "也没有退出记录——那一窗我在反复换 daemon 版本，前一个进程是谁停的没留痕。"
               "起点 492,036／498,324／507,232／511,240 kB 的四次全就绪了"
               "（139.5／147.8／143.0／144.1 s）。门禁取这两档中间的空档 470,000 kB；"
               "按装载峰值×1.15 那套老算法（%d MB ← 10-14 台账 #197 的 VmHWM %d kB／"
               "%d MB ← 11-06 bisect.log 的 %d kB）会把四次里至少三次判死，而它们实跑通了，所以弃用。"
               "断路器现在按非 CMA free 跌破 22,600 kB 自我了断（三次死亡值都低于它，"
               "四次就绪那刻读到 32,524~43,380 kB，都高于它），不再让内核替板子挑时间。"
               "腾地方的路：停掉视觉/语音那六个服务（systemctl stop wcis-face-photo "
               "wcis-face-recording wcis-sdi-vpu-view lang-server-http lang-server-loop lang-server-voice）。"
               "重启不解决问题——这六个 unit 全是 enabled，开机就回来。"
               "那几条是别的会话的现役，我不替你关。"
               % (avail // 1024, NEED_KB // 1024,
                  int(PEAK_KB * 1.15) // 1024, PEAK_KB,
                  int(PEAK2_KB * 1.15) // 1024, PEAK2_KB))
        FATAL = {"event": "load-refused", "msg": msg, "avail_kb": avail, "free_kb": free,
                 "need_kb": NEED_KB, "peak_kb": PEAK_KB, "top": topmem()}
        set_state("load-refused avail_kb=%d free_kb=%d" % (avail, free))
        log("REFUSE: " + msg + " " + str(FATAL["top"]))
        broadcast(FATAL)
        time.sleep(GRACE_S)
        os._exit(3)
    t0 = time.time()
    if avail < PROGNOSIS_KB:
        # Not a refusal, just honesty about the odds: the lowest start MemAvailable
        # that actually reached ready on 11-06 was 492,036 kB (139.5 s), and every
        # start from 402,160 to 413,576 kB (16 of them) never got there. This band
        # is between those two sets, so nobody has measured it -- n=0, a prognosis,
        # not a verdict.
        note = ("起点可用内存 %d kB，落在 11-06 没测过的空档里（16 次 402,160~413,576 kB 的起跑"
                "一次都没就绪；走到就绪的四次起点是 492,036／498,324／507,232／511,240 kB，"
                "139.5~147.8 s）。"
                "闸会在非 CMA free 跌破 22,600 kB 时自我了断，不拖累别的进程；"
                "真被杀了就在桌面窗口里重开一次——那等于重新装载。" % avail)
        PROGNOSIS = note
        log("PROGNOSIS " + note)
        broadcast({"event": "prognosis", "msg": note, "avail_kb": avail})
    try:
        GEN = m.QwenGenerator(m.ModelConfig())
    except Exception as e:
        PHASE = "load-failed"
        set_state("load-failed %r at %.0fs" % (e, time.time() - t0))
        log("LOAD FAILED after %.1fs: %r" % (time.time() - t0, e))
        FATAL = {"event": "load-failed", "error": repr(e),
                 "load_secs": round(time.time() - t0, 1)}
        broadcast(FATAL)
        time.sleep(GRACE_S)      # stay listening so a late client still gets the reason
        os._exit(1)
    LOAD_SECS = time.time() - t0
    PHASE = "ready"
    try:
        with open(READY_FLAG, "w") as f:
            f.write("%.1f\n" % LOAD_SECS)
    except OSError as e:
        log("ready flag not written (%r); breaker will still enforce the deadline" % (e,))
    set_state("ready pid=%d load_secs=%.1f" % (os.getpid(), LOAD_SECS))
    log("READY %.1fs %s" % (LOAD_SECS, memline()))
    broadcast({"event": "ready", "load_secs": round(LOAD_SECS, 1), "mem": memline()})


def ask(text, max_new):
    ids = GEN.model.tokenizer.encode(text).ids
    room = KV_WINDOW - len(ids) - SAFETY_SLOTS
    if room < 1:
        raise ValueError("prompt is %d tokens, only %d slots left in the %d-token window"
                         % (len(ids), room, KV_WINDOW))
    max_new = int(min(max_new, room))
    noise = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(noise):
        answer = GEN.generate(prompt=text, max_length=max_new)
    secs = time.time() - t0
    gen_n = None
    for line in noise.getvalue().splitlines():
        if any(k in line for k in ("First generated token", "Average decode", "tokens/s",
                                   "EOS token", "Smart stopping", "Max length", "共生成")):
            log("  | " + line.strip())
        if "共生成" in line and gen_n is None:
            tail = line.split("共生成", 1)[1]
            digits = ""
            for ch in tail:
                if ch.isdigit():
                    digits += ch
                elif digits:
                    break
            gen_n = int(digits) if digits else None
    return answer, max_new, secs, len(ids), gen_n


def serve(sock):
    if FATAL is not None:
        # Refusing and vanishing is worse than refusing in her face: the Desktop
        # entry connects ~6s after spawning me, so hold the port through the grace
        # window and replay the reason to whoever shows up.
        try:
            send_one(sock, FATAL)
            # sendall only queues; closing straight away can RST the pending bytes
            # away and the client sees a bare EOF instead of the reason.
            sock.shutdown(socket.SHUT_WR)
            sock.settimeout(2.0)
            while sock.recv(4096):
                pass
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass
        return
    LAST_ACTIVITY[0] = time.time()
    CLIENTS.append(sock)
    hello(sock)
    f = sock.makefile("r", encoding="utf-8", newline="\n")
    for line in f:
        LAST_ACTIVITY[0] = time.time()
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            send_one(sock, {"ok": False, "error": "not json: " + line[:80]})
            continue
        if req.get("ping"):
            send_one(sock, {"event": "pong", "phase": PHASE, "busy": BUSY_SINCE[0] is not None,
                            "load_secs": LOAD_SECS})
            continue
        if req.get("bye"):
            log("client said bye, releasing the NPU")
            set_state("dead bye")
            os._exit(0)
        text = (req.get("text") or "").strip()
        if not text:
            send_one(sock, {"ok": False, "error": "empty text"})
            continue
        if PHASE != "ready":
            send_one(sock, {"ok": False, "error": "model is %s (elapsed %.0fs, eta %.0fs)"
                                   % (PHASE, time.time() - LOAD_T0, LOAD_ETA_S)})
            continue
        with LOCK:
            BUSY_SINCE[0] = time.time()
            set_state("turn pid=%d text=%r" % (os.getpid(), text[:40]))
            try:
                answer, used, secs, plen, gen_n = ask(text, int(req.get("max_new", 24)))
                n = gen_n if gen_n else used
                send_one(sock, {"ok": True, "answer": answer, "secs": round(secs, 1),
                                "max_new": used, "gen_tokens": n, "prompt_tokens": plen,
                                "tok_s": round(n / secs, 2)})
                log("TURN %.1fs gen=%d max_new=%d prompt=%d" % (secs, n, used, plen))
            except Exception as e:
                log("turn failed: %r" % (e,))
                send_one(sock, {"ok": False, "error": repr(e)})
            finally:
                BUSY_SINCE[0] = None
                LAST_ACTIVITY[0] = time.time()
                set_state("ready pid=%d idle" % os.getpid())
    try:
        CLIENTS.remove(sock)
    except ValueError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def watchdog():
    while True:
        time.sleep(5)
        now = time.time()
        if BUSY_SINCE[0] is not None and now - BUSY_SINCE[0] > TURN_DEADLINE_S:
            log("turn exceeded %.0fs deadline -- the NPU is wedged, exiting" % TURN_DEADLINE_S)
            set_state("dead turn-deadline")
            os._exit(1)
        if PHASE == "ready" and not CLIENTS and now - LAST_ACTIVITY[0] > IDLE_S:
            log("idle %.0fs, releasing the NPU" % IDLE_S)
            set_state("dead idle")
            os._exit(0)


def main():
    try:
        os.remove(READY_FLAG)   # a stale flag would disarm this run's deadline
    except OSError:
        pass
    arm_breaker()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((BIND, PORT))
    except OSError as e:
        log("cannot bind %s:%d (%r) -- another chatd owns the port, this one exits" % (BIND, PORT, e))
        return 2
    srv.listen(8)
    set_state("loading pid=%d" % os.getpid())
    log("listening on %s:%d, loading model (~%.0fs) %s" % (BIND, PORT, LOAD_ETA_S, memline()))
    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=load, daemon=True).start()
    while True:
        try:
            c, addr = srv.accept()
        except OSError as e:
            log("accept failed: %r" % (e,))
            return 1
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        log("client %s:%s" % addr)
        threading.Thread(target=serve, args=(c,), daemon=True).start()


if __name__ == "__main__":
    main()
