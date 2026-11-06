#!/bin/bash
# qwen_bisect.sh — 逐步定位"起跑就整机挂死"死在哪一步
#
# 必须人在板子旁边、随时能断电：这块板没有 /dev/watchdog（DT disabled），
# 挂了以后 SSH/网络全死，只有 COM6 串口和物理电源开关。
#
# 建议同时开着串口抓取（那是唯一能看到内核 printk 的通道，journald 那时已经死了）：
#   PowerShell: $p = New-Object System.IO.Ports.SerialPort "COM6",115200,"None",8,"One"
#               $p.Open(); $p.Write("`r`n"); Start-Sleep 3; 读 BytesToRead
#
# 每步都是独立进程 + 独立心跳：前一步的结论不会被后一步的挂死吃掉。
set -u
cd /root/qwen2 || { echo "必须在 /root/qwen2 里跑（modelzoo_utils 是相对路径引入）"; exit 2; }
LOG=/root/qwen2/bisect.log
AS=${AS:-2G}
echo "=== bisect 开始 $(date -u '+%F %T UTC')  as=$AS  log=$LOG" | tee -a "$LOG"

for n in 1 2 3 4; do
  echo "" | tee -a "$LOG"
  echo "--------- STEP $n ---------"
  cap.sh --as "$AS" --tag "qb_s$n" --interval 1 -- \
      python3 /root/qwen2/qwen_bisect.py "$n"
  rc=$?
  echo "--------- STEP $n 退出码 $rc ($(date -u '+%F %T UTC')) ---------" | tee -a "$LOG"
  if [ "$rc" != "0" ]; then
    echo "★ STEP $n 不是 0 ⇒ 死点就在这一步。看 /root/hb/qb_s$n.log 最后几行。" | tee -a "$LOG"
    break
  fi
done
echo "=== bisect 结束 $(date -u '+%F %T UTC') ===" | tee -a "$LOG"