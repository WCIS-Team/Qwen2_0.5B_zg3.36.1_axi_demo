#!/bin/sh
# 1 s 心跳采样，整行追加到 eMMC（板子若整机挂死，断电重启后这是唯一能看到死前最后状态的通道）。
# 用法：setsid sh /root/qwen2/mem_watch.sh /root/qwen2/mem_watch.log qwen_bisect 600 &
LOG=${1:-/root/qwen2/mem_watch.log}
PAT=${2:-qwen_bisect}
N=${3:-600}
i=0
echo "=== watch start $(date -u '+%F %T UTC') pat=$PAT n=$N ===" >> "$LOG"
while [ "$i" -lt "$N" ]; do
  i=$((i+1))
  MI=$(awk '/^MemFree|^MemAvailable|^CmaFree|^SwapFree|^SwapUsed|^Active\(anon\)/{printf "%s=%s ",$1,$2}' /proc/meminfo)
  ST=$(awk '/^VmSize|^VmRSS|^VmHWM|^VmSwap/{printf "%s=%s ",$1,$2}' "/proc/$(pgrep -f "$PAT" | head -1)/status" 2>/dev/null)
  echo "$(date -u '+%T') $MI $ST" >> "$LOG"
  sleep 1
done
echo "=== watch end $(date -u '+%F %T UTC') ===" >> "$LOG"
