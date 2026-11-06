# -*- coding: utf-8 -*-
"""Qwen2-0.5B 板端对话入口（悟净 FMQL30TAI676H / ZG330，192.168.1.113:18099）。

双击桌面 `Qwen2 对话.lnk` 即可。窗口一开就连板子上的常驻进程 qwen2_chatd.py：
连不上就自己用 ssh 把它拉起来（不占终端、日志落在板上 /root/qwen2/chatd.log），
然后等它装载完。11-06 四次就绪实测 139.5／143.0／144.1／147.8 s；生成侧同一批四轮
20 个 token 用 6.7~7.5 s（2.75~2.97 token/s；10-14 台账 #197 那颗是 247.71 ms／2.83 token/s）。
起跑门禁＝板上 MemAvailable ≥ 470,000 kB（起点 402,160~413,576 kB 的 16 次零就绪、
492,036 kB 以上的 4 次全就绪，门禁取中间空档）；不够时窗口不关就自己再试。

关窗口 = 发一条 bye = 板端立刻退出并归还 NPU/CMA；就算崩了没发出去，
板端自己 240 s 空闲也会退。单轮作答、不记上下文：多轮历史要按 token 付预填时间。
"""
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import scrolledtext, ttk

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BOARD_IP = os.environ.get("Q2_IP", "192.168.1.113")
PORT = int(os.environ.get("Q2_PORT", "18099"))
SSH = os.environ.get("Q2_SSH", "root@192.168.1.113")
DAEMON = "/root/qwen2/qwen2_chatd.py"
START_CMD = ("cd /root/qwen2 && MALLOC_ARENA_MAX=1 setsid nohup python3 -u %s >> /root/qwen2/chatd.log 2>&1 "
             "< /dev/null &" % DAEMON)
LOAD_ETA_S = 150      # 11-06 measured 139.5 / 143.0 / 144.1 / 147.8 s (/root/qwen2/chatd.log)
PER_TOKEN_S = 0.365   # those four turns end to end: 20 generated tokens in 6.7-7.5 s
RX_TIMEOUT_S = 10.0    # how long one read waits before we ping and keep waiting
BACKOFF_S = 5.0        # re-dial pace when a daemon we already reached is busy
SPAWN_BUDGET_S = 750.  # one daemon's own lifetime ceiling: 600 s load deadline + 150 s grace
NO_READY_S = 900.      # after the first spawn, stop waiting -- and stop spawning
REFUSE_RETRY_S = 60.0        # the board refused on memory: first re-try after this long
REFUSE_RETRY_MAX_S = 300.0   # ... then 120,180,240,300,300 (a refused spawn never loaded)
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.sock = None
        self.ready = False
        self.pending = False
        self.busy = False
        self.turn_started = None
        self.phase = "connecting"
        self.queued = []
        self.load_elapsed_t0 = time.time()
        self.q = queue.Queue()
        self.spawn_t = None
        self.spawn_count = 0
        self.refused = False
        self.closing = False
        # 11-06 03:58 read: while the daemon is inside icraft it holds the GIL, so
        # accept() does not run at all -- our connect succeeds (the kernel finishes
        # the handshake into the listen backlog) and then reads nothing. Churning
        # re-dials and spawns a second daemon that collides on the port. So: one
        # patient connection, and a wall-clock budget instead of "connect failed
        # twice" to decide anything.
        self.buf = b""
        self.ever_connected = False
        self.dead = False          # last connect was refused: no daemon on the port
        self.last_rx = time.time()
        self.first_spawn_t = None
        self.given_up = False
        self.refuse_at = 0.0       # when the next spawn is allowed after a memory refusal
        self.refuse_tries = 0

        root.title("Qwen2-0.5B 板端对话（ZG330 @113）")
        root.geometry("780x540")
        self.log = scrolledtext.ScrolledText(root, wrap="word", state="disabled",
                                             font=("Microsoft YaHei UI", 11))
        self.log.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        bar = ttk.Frame(root)
        bar.pack(fill="x", padx=8, pady=(0, 4))
        self.status = ttk.Label(bar, text="连接中……")
        self.status.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="放弃本轮", command=self.on_cancel, state="disabled")
        self.stop_btn.pack(side="right")

        row = ttk.Frame(root)
        row.pack(fill="x", padx=8, pady=(0, 8))
        self.entry = ttk.Entry(row, font=("Microsoft YaHei UI", 11))
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self.on_send)
        ttk.Label(row, text="最多").pack(side="left", padx=(8, 2))
        self.maxvar = tk.IntVar(value=24)
        tk.Spinbox(row, from_=8, to=64, increment=8, textvariable=self.maxvar,
                   width=5, font=("Microsoft YaHei UI", 10)).pack(side="left")
        ttk.Label(row, text="token").pack(side="left", padx=(2, 6))
        self.send_btn = ttk.Button(row, text="发送", command=self.on_send)
        self.send_btn.pack(side="left")
        self.entry.focus_set()

        self.write("板子：%s:%d｜装载模型约 %d 秒（11-06 四次实测 139.5~147.8 s），就绪后每 token 约 %.2f 秒"
                   "（四轮 20 个 token 用 6.7~7.5 s；驱动自报平均解码 248.87~259.88 ms／平均输出 2.75~2.97 token/s，"
                   "10-14 台账 #197 那颗是 247.71 ms／2.83 token/s）。单轮作答，不记上文。\n\n"
                   % (BOARD_IP, PORT, LOAD_ETA_S, PER_TOKEN_S), "hint")
        threading.Thread(target=self.io_loop, daemon=True).start()
        self.tick()

    # ---------------- transport ----------------

    def connect(self):
        try:
            s = socket.create_connection((BOARD_IP, PORT), timeout=5)
        except ConnectionRefusedError:
            # Nothing is listening. That is decisive evidence the daemon is gone
            # (breaker kill, load-failed exit, `bye`, reboot) -- unlike a connect
            # that succeeds and then reads nothing, which is only icraft holding
            # the GIL. The 04:31:15 breaker kill sat behind the patient-wait rule
            # for the rest of the budget because last_rx could not tell the two
            # apart.
            self.dead = True
            return None
        except OSError:
            self.dead = False     # board unreachable/filtered: nothing to re-spawn yet
            return None
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s
        self.buf = b""
        self.ever_connected = True
        self.dead = False
        self.last_rx = time.time()
        self.phase = "loading"
        self.load_elapsed_t0 = time.time()
        self.q.put(("phase", "connected"))
        return s

    def start_daemon(self):
        """Bring the board daemon up through ssh, detached, logging on the board."""
        cmd = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", SSH, START_CMD]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                 errors="replace", creationflags=CREATE_NO_WINDOW)
        except OSError as e:
            self.q.put(("line", "【错】启动 ssh 失败：%r\n" % (e,)))
            return False
        try:
            _, err = p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()          # the daemon is already detached; only the wrapper lingers
            err = ""
        if (err or "").strip():
            self.q.put(("line", "· " + err.strip()[:400] + "\n"))
        return True

    def io_loop(self):
        """One thread: keep a socket, push lines to the UI, ping so we don't idle out."""
        last_ping = 0.0
        while not self.closing:
            if self.sock is None:
                if self.given_up:
                    # the board said why, or we ran out of budget; re-dialling only
                    # re-prints the same thing (and costs the board a 145 s load)
                    time.sleep(10)
                    continue
                if self.refused:
                    if time.time() < self.refuse_at:
                        time.sleep(2)
                        continue
                    # A refused spawn never loaded, so it spent none of the load
                    # budget, and the earlier connection was only the refusal message:
                    # clear the "we already talked to a live daemon" state so the
                    # spawn path below is reachable again.
                    self.refused = False
                    self.ever_connected = False
                    self.dead = False
                    self.last_rx = 0.0
                    self.first_spawn_t = time.time()
                if self.connect():
                    last_ping = time.time()
                    continue
                now = time.time()
                if (self.first_spawn_t is not None
                        and now - self.first_spawn_t > NO_READY_S):
                    self.given_up = True
                    self.q.put(("line", "【没等到就绪】拉起过 %d 次，等满 %.0f s 也没等到 ready。"
                                        "这一条不是内存判定——板上到底卡在哪一步只看三个文件："
                                        "/root/qwen2/chatd.state（活着／拒绝／就绪）、"
                                        "/root/qwen2/breaker.log（被内存闸自我了断）、"
                                        "/root/qwen2/chatd.log（装载过程）。重开窗口＝重来一轮。\n"
                                        % (self.spawn_count, NO_READY_S)))
                    time.sleep(10)
                    continue
                if (self.ever_connected and not self.dead
                        and now - self.last_rx < SPAWN_BUDGET_S):
                    time.sleep(BACKOFF_S)      # a daemon we reached is alive, just busy
                    continue
                if self.spawn_count >= 2:
                    time.sleep(BACKOFF_S)      # two loads already spent; wait for the budget
                    continue
                if self.ever_connected and self.dead:
                    self.q.put(("line", "板端对话进程已经不在这个端口上了（被内存闸杀掉、"
                                        "装载失败退出、或上一轮说了 bye）。第 %d 次重新拉起装载。\n"
                                        % (self.spawn_count + 1)))
                self.spawn_count += 1
                if self.first_spawn_t is None:
                    self.first_spawn_t = now
                self.spawn_t = now
                self.q.put(("phase", "spawn"))
                self.start_daemon()
                time.sleep(6)
                continue
            try:
                self.sock.settimeout(RX_TIMEOUT_S)
                chunk = self.sock.recv(4096)
            except socket.timeout:
                if time.time() - last_ping > 20:
                    last_ping = time.time()
                    try:
                        self.sock.sendall(b'{"ping": true}\n')
                    except OSError:
                        self.drop()
                continue
            except OSError:
                self.drop()
                time.sleep(BACKOFF_S)
                continue
            if not chunk:
                self.drop()
                time.sleep(BACKOFF_S)      # peer closed: don't hammer the port
                continue
            self.last_rx = time.time()
            self.buf += chunk
            while b"\n" in self.buf:
                raw, self.buf = self.buf.split(b"\n", 1)
                if raw.strip():
                    self.q.put(("line", raw.decode("utf-8", "replace").rstrip("\r")))
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def drop(self):
        s = self.sock
        self.sock = None
        self.buf = b""
        self.ready = False
        if s is not None:
            # Closing matters: a leaked socket gets GC'd later, and the board's
            # reader thread dies with "OSError: [Errno 9] Bad file descriptor"
            # (measured 11-06 04:00:14, 8 such connections in one second).
            try:
                s.close()
            except OSError:
                pass

    def send(self, obj):
        if self.sock is None:
            return False
        try:
            self.sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
            return True
        except OSError:
            self.drop()
            return False

    # ---------------- ui ----------------

    def on_send(self, _=None):
        text = self.entry.get().strip()
        if not text or self.pending:
            return
        self.entry.delete(0, "end")
        self.write("你：%s\n" % text, "me")
        if not self.ready:
            self.queued.append(text)
            self.write("（模型还在 %s，先排队，就绪后自动发出。）\n" % self.phase, "hint")
            return
        self.begin_turn(text)

    def begin_turn(self, text):
        if not self.send({"text": text, "max_new": int(self.maxvar.get())}):
            self.write("（发送失败，正在重连……）\n", "err")
            self.queued.insert(0, text)
            return
        self.pending = True
        self.turn_started = time.time()
        self.stop_btn.config(state="normal")
        self.set_status("生成中……")

    def on_cancel(self):
        if not self.pending:
            return
        self.write("（本轮已放弃；板端这一轮跑完才会让出 NPU，答复直接丢弃。）\n", "hint")
        self.pending = False
        self.turn_started = None
        self.stop_btn.config(state="disabled")
        self.drop()          # reply goes to the dead socket; we reconnect below

    def set_status(self, s):
        self.status.config(text=s)

    def write(self, s, tag=None):
        self.log.config(state="normal")
        if tag:
            self.log.insert("end", s, tag)
        else:
            self.log.insert("end", s)
        self.log.see("end")
        self.log.config(state="disabled")

    def tick(self):
        self.drain()
        if self.pending and self.turn_started:
            el = time.time() - self.turn_started
            self.set_status("生成中…… %.0f s（约 %.2f s/token）" % (el, PER_TOKEN_S))
        elif self.refused:
            left = max(0.0, self.refuse_at - time.time())
            self.set_status("板端内存不够（第 %d 次被拒），%.0f s 后自动再试" % (self.refuse_tries, left))
        elif not self.ready:
            el = time.time() - self.load_elapsed_t0
            self.set_status("装载模型中…… 已 %.0f s / 约 %d s" % (el, LOAD_ETA_S))
        else:
            self.set_status("已就绪")
        self.root.after(200, self.tick)

    def drain(self):
        for _ in range(50):
            try:
                kind, payload = self.q.get_nowait()
            except queue.Empty:
                return
            if kind == "phase":
                if payload == "spawn":
                    self.write("板端对话进程没在跑，正在启动并装载模型……\n", "hint")
                elif payload == "connected":
                    self.write("已连上板端对话进程。\n", "hint")
                continue
            if kind == "line":
                if payload.startswith("{"):
                    try:
                        self.handle(json.loads(payload))
                        continue
                    except ValueError:
                        pass
                # Anything that is not a JSON event is a status line we raised
                # ourselves or ssh stderr -- print it. Filtering by the "【错】"
                # prefix alone silently ate the 装不起来/没等到就绪 lines.
                if payload.startswith("【"):
                    self.write(payload + "\n", "err")
                elif payload.startswith("· "):
                    self.write(payload + "\n", "noise")
                else:
                    self.write(payload + "\n")
                continue

    def handle(self, msg):
        ev = msg.get("event")
        if ev == "hello":
            self.phase = msg.get("phase", "?")
            if msg.get("prognosis"):
                self.write("【预判】%s\n\n" % msg["prognosis"], "hint")
            if self.phase == "ready":
                self.mark_ready(msg.get("load_secs") or 0.0)
            else:
                self.load_elapsed_t0 = time.time() - float(msg.get("elapsed") or 0)
                self.write("板端状态：%s（已用 %.0f s，约需 %.0f s）\n"
                           % (self.phase, msg.get("elapsed") or 0, msg.get("eta") or LOAD_ETA_S),
                           "hint")
            return
        if ev == "ready":
            self.mark_ready(msg.get("load_secs") or 0.0, msg.get("mem"))
            return
        if ev == "load-failed":
            self.write("【错】板端装载失败：%s\n" % msg.get("error"), "err")
            self.phase = "load-failed"
            return
        if ev == "load-refused":
            self.refused = True
            if self.spawn_count > 0:
                self.spawn_count -= 1     # a refused daemon never loaded: nothing was spent
            self.refuse_tries += 1
            delay = min(REFUSE_RETRY_S * self.refuse_tries, REFUSE_RETRY_MAX_S)
            self.refuse_at = time.time() + delay
            need_kb = msg.get("need_kb") or 470000
            avail_kb = msg.get("avail_kb") or 0
            short = "%d kB / 门禁 %d kB，差 %d kB" % (avail_kb, need_kb, max(0, need_kb - avail_kb))
            if self.refuse_tries == 1:
                self.write("【板端内存不够，这次没有装载】%s\n" % msg.get("msg"), "err")
                self.write("占内存前几名：%s\n" % "、".join(
                    "%s(%d MB)" % (t.get("comm"), (t.get("rss_kb") or 0) // 1024)
                    for t in (msg.get("top") or [])), "hint")
                self.write("门禁线＝起跑时 MemAvailable ≥ %d kB（11-06 实测：起点 402,160~413,576 kB 的"
                           " 16 次零就绪，492,036 kB 以上的 4 次全就绪）。这次读到 %s。\n"
                           "停哪几条板上的活儿由你定，我不替你关，也不替你重启板子。\n"
                           "此窗口不关就一直在等：%.0f s 后自动再拉起一次，之后每次加长到 %d s。\n"
                           % (need_kb, short, delay, int(REFUSE_RETRY_MAX_S)), "hint")
            else:
                self.write("（第 %d 次被拒：可用 %s；%.0f s 后再试）\n"
                           % (self.refuse_tries, short, delay), "hint")
            return
        if ev == "pong":
            self.phase = msg.get("phase", self.phase)
            if msg.get("phase") == "ready" and not self.ready:
                self.mark_ready(msg.get("load_secs") or 0.0)
            return
        self.pending = False
        self.turn_started = None
        self.stop_btn.config(state="disabled")
        if msg.get("ok"):
            secs = msg.get("secs", 0.0)
            self.write("Qwen2：%s\n（%.1f s，生成 %d token，%.2f tok/s）\n\n"
                       % (msg.get("answer", ""), secs,
                          msg.get("gen_tokens") or msg.get("max_new", 0),
                          msg.get("tok_s") or 0.0), "bot")
            if self.queued:
                self.begin_turn(self.queued.pop(0))
        else:
            self.write("【错】%s\n\n" % msg.get("error", "?"), "err")
            if self.queued:
                self.write("（排队的那条改天再发，请重发。）\n", "hint")
                self.queued = []

    def mark_ready(self, load_secs, mem=None):
        self.ready = True
        self.phase = "ready"
        self.refused = False
        self.given_up = False
        self.spawn_count = 0
        self.refuse_tries = 0
        self.refuse_at = 0.0
        self.spawn_t = None
        self.first_spawn_t = None
        tail = ""
        if mem:
            tail = "，CmaFree %.0f MB / 可用 %.0f MB" % (mem.get("CmaFree", 0) / 1024.0,
                                                       mem.get("MemAvailable", 0) / 1024.0)
        self.write("模型就绪（装载 %.1f s%s），可以发了。\n\n" % (load_secs, tail), "hint")
        if self.queued and not self.pending:
            self.begin_turn(self.queued.pop(0))

    def on_close(self):
        self.closing = True
        self.send({"bye": True})
        self.drop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ChatApp(root)
    for tag, color in (("me", "#1a5276"), ("bot", "#14342b"), ("hint", "#7d6608"),
                       ("err", "#922b21"), ("noise", "#7f8c8d")):
        app.log.tag_configure(tag, foreground=color)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        # pythonw has no console, so a crash would otherwise be invisible
        tb = traceback.format_exc()
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "chat-crash.log"), "a", encoding="utf-8") as f:
                f.write(time.strftime("[%m-%d %H:%M:%S] ") + tb + "\n")
        except OSError:
            pass
        try:
            from tkinter import messagebox
            messagebox.showerror("Qwen2 对话", tb[-1200:])
        except Exception:
            pass
        raise
