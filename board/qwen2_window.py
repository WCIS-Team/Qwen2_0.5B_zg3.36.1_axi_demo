"""Board-side: wait for a memory window, load the daemon, drive one real turn.

Why this exists: 11-06 logged 15 loads that started listening. The nine that began
at MemAvailable 402,160-413,576 kB none reached ready; the one that came up in
144.1 s began at 511,240 kB. So the missing ingredient is not the daemon, it is a
window. This waits for one instead of gambling a load every time, and once the
model is ready it asks it one question over 127.0.0.1 so the generation numbers
from ledger #197 get reproduced (or don't) independently of the Windows client.

Run detached on the board:
  cd /root/qwen2 && setsid nohup python3 -u qwen2_window.py >> /root/qwen2/window.log 2>&1 &
"""
import json
import os
import socket
import subprocess
import sys
import time

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
DAEMON = os.path.join(HERE, "qwen2_chatd.py")
PORT = 18099
NEED_AVAIL_KB = int(os.environ.get("W_AVAIL", "470000"))
NEED_NONCMA_KB = int(os.environ.get("W_NONCMA", "55000"))
POLL_S = 20.0
MAX_POLLS = int(os.environ.get("W_POLLS", "120"))     # ~40 min of watching
ATTEMPTS = 3
TEXT = os.environ.get("W_TEXT", "你是谁？")
MAX_NEW = 24


def say(*a):
    print(time.strftime("[%m-%d %H:%M:%S] "), *a, flush=True)


def kb(name):
    for line in open("/proc/meminfo"):
        if line.startswith(name):
            return int(line.split()[1])
    return -1


def window():
    avail = kb("MemAvailable")
    noncma = kb("MemFree") - kb("CmaFree")
    return avail, noncma, (avail >= NEED_AVAIL_KB and noncma >= NEED_NONCMA_KB)


def start_daemon():
    cmd = ("cd %s && MALLOC_ARENA_MAX=1 setsid nohup python3 -u %s >> %s/chatd.log 2>&1 &"
           % (HERE, DAEMON, HERE))
    subprocess.run(["sh", "-c", cmd], check=False)


def ready_and_turn():
    """Wait for the daemon to say ready, then ask one question and read the answer."""
    t0 = time.time()
    s = None
    while time.time() - t0 < 600:
        try:
            s = socket.create_connection(("127.0.0.1", PORT), timeout=5)
            break
        except OSError:
            time.sleep(5)
    if s is None:
        say("no daemon on the port within 600 s")
        return False
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    buf = b""
    s.settimeout(600.0)
    # one hello, then wait for the ready event
    while True:
        try:
            chunk = s.recv(4096)
        except OSError as e:
            say("recv gone while waiting for ready: %r" % (e,))
            return False
        if not chunk:
            say("daemon closed while we waited for ready (see breaker.log)")
            return False
        buf += chunk
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            try:
                msg = json.loads(raw)
            except ValueError:
                say("plain: %s" % raw.decode("utf-8", "replace")[:160])
                continue
            ev = msg.get("event")
            if ev == "hello":
                say("hello phase=%s elapsed=%.1f eta=%s prog=%s"
                    % (msg.get("phase"), msg.get("elapsed") or 0.0, msg.get("eta"),
                       "yes" if msg.get("prognosis") else "no"))
            elif ev == "ready":
                say("READY load_secs=%s mem=%s" % (msg.get("load_secs"), msg.get("mem")))
                s.sendall((json.dumps({"text": TEXT, "max_new": MAX_NEW},
                                      ensure_ascii=False) + "\n").encode("utf-8"))
                return read_answer(s, buf)
            elif ev in ("load-refused", "load-failed"):
                say("%s %s" % (ev, str(msg)[:400]))
                return False
    return False


def read_answer(s, buf):
    while True:
        try:
            chunk = s.recv(4096)
        except OSError as e:
            say("answer recv failed: %r" % (e,))
            return False
        if not chunk:
            say("answer stream closed with no reply")
            return False
        buf += chunk
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if "ok" not in msg:
                continue
            say("ANSWER ok=%s secs=%s tok_s=%s max_new=%s prompt_tokens=%s"
                % (msg.get("ok"), msg.get("secs"), msg.get("tok_s"),
                   msg.get("max_new"), msg.get("prompt_tokens")))
            say("TEXT=%s" % (msg.get("answer") or msg.get("error") or ""))
            ok = bool(msg.get("ok"))
            if os.environ.get("W_BYE", "1") == "1":
                try:
                    s.sendall(b'{"bye": true}\n')
                    s.shutdown(socket.SHUT_WR)
                    s.settimeout(5.0)
                    while s.recv(4096):
                        pass
                except OSError:
                    pass
            else:
                # Hang up without `bye` so the model stays loaded for the next client
                # (the Desktop entry under test). The daemon's own IDLE_S timer is the
                # backstop: with no client attached it releases the NPU and the CMA
                # pool by itself, so a watcher that quits can never strand 122 MB of
                # CMA on top of her video chain.
                say("W_BYE=0: hanging up without bye; daemon stays ready for IDLE_S then exits")
            try:
                s.close()
            except OSError:
                pass
            return ok


def main():
    say("window watcher up: need avail>=%d noncma>=%d, %d polls x %.0f s, %d attempts"
        % (NEED_AVAIL_KB, NEED_NONCMA_KB, MAX_POLLS, POLL_S, ATTEMPTS))
    for n in range(1, ATTEMPTS + 1):
        polls = MAX_POLLS if n == 1 else 30
        hit = None
        for i in range(polls):
            avail, noncma, ok = window()
            if i % 6 == 0 or ok:
                say("poll %d avail=%d noncma=%d open=%s" % (i, avail, noncma, ok))
            if ok:
                hit = (avail, noncma)
                break
            time.sleep(POLL_S)
        if hit is None:
            say("no window in this stretch; giving up")
            return 1
        say("attempt %d: window at avail=%d noncma=%d -> starting daemon" % (n, hit[0], hit[1]))
        start_daemon()
        if ready_and_turn():
            say("attempt %d produced an answer" % n)
            return 0
        say("attempt %d did not produce an answer" % n)
        time.sleep(30)
    return 1


if __name__ == "__main__":
    sys.exit(main())
