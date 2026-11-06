"""Protocol-only stand-in for qwen2_chatd.py: listens on 18099, answers hello/ping/
text/bye, never touches icraft or the NPU.

Purpose: the Windows client's transport was rewritten tonight (single patient
connection, byte buffer, refusal-vs-dead distinction). A memory window for a real
load comes and goes, so the transport gets tested against this fake instead; it
answers instantly, so the test covers the socket, the JSON framing, the Tk render
path and the shutdown handshake, and nothing about the model.
"""
import json
import socket
import sys
import threading
import time

BIND = "0.0.0.0"
PORT = 18099
ANSWER = "（这是协议替身，不是模型输出：只验证桌面客户端的连接、收发与渲染）"


def log(m):
    print(time.strftime("[%m-%d %H:%M:%S] ") + m, flush=True)


def serve(c, addr):
    log("client %s:%s" % addr)
    c.sendall((json.dumps({"event": "hello", "phase": "ready", "model": "fake",
                           "load_secs": 1.2, "elapsed": 1.2, "eta": 300.0,
                           "kv_window": 256, "busy": False, "prognosis": None,
                           "mem": {"MemFree": 1, "MemAvailable": 2, "CmaFree": 3}},
                          ensure_ascii=False) + "\n").encode("utf-8"))
    f = c.makefile("r", encoding="utf-8", newline="\n")
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            c.sendall((json.dumps({"ok": False, "error": "not json"}) + "\n").encode("utf-8"))
            continue
        if req.get("ping"):
            c.sendall((json.dumps({"event": "pong", "phase": "ready", "busy": False,
                                   "load_secs": 1.2}) + "\n").encode("utf-8"))
            continue
        if req.get("bye"):
            log("bye from client, fake daemon exits")
            sys.exit(0)
        c.sendall((json.dumps({"ok": True, "answer": ANSWER, "secs": 0.5,
                               "max_new": int(req.get("max_new", 24)),
                               "prompt_tokens": 3, "tok_s": 4.0}) + "\n").encode("utf-8"))


def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((BIND, PORT))
    srv.listen(8)
    log("fake daemon listening on %s:%d" % (BIND, PORT))
    while True:
        c, addr = srv.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=serve, args=(c, addr), daemon=True).start()


if __name__ == "__main__":
    main()
