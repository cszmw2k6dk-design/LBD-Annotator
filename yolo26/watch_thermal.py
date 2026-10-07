"""散热旁路监视器：用训练的实时速度当"温度代理"，顺便记录热降频事件。

为什么这么绕：这台机器读不到 CPU 温度。
- ACPI 热区（性能计数器/root\\wmi）只有一个 27.9C 的机壳传感器，不是核心温度；
- Win32_Processor 只报标称 2500MHz，拿不到实际频率；
- 事件日志里的"固件限速"（Kernel-Processor-Power 37）实测是开机时上报一次，
  跟负载无关（消息里写着"已处于该状态 86400 秒"），不能当信号；
- 硬盘温度接口也被拒。
真实核心温度要么装 LibreHardwareMonitor/HWiNFO 这类带驱动的工具，要么读不到。

能拿到的最有用信号：**降频会让训练变慢**。同一份数据、同一个 batch、同样的分辨率，
每批耗时应该稳定；一旦持续变慢，多半就是在降频（或者内存换页）。

本脚本独立运行，不碰训练进程，也不要求重启训练。

两条重要教训（2026-09-29 踩过）：
1. **只统计训练阶段的读数**。验证阶段每批只要 10 秒左右，混进来会把"历史最好"拉到
   11 s/批，然后正常训练就显得"慢 60%"，纯属自欺。用 Class/Instances 表头做分界。
2. **本脚本只报警、绝不停训练**。慢不代表要停——降频是芯片的保护动作，停训练既救不了
   机器、又白丢进度。真异常由心跳自动化通知用户，让用户决定（比如把电源计划调到 90%）。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "thermal.log"
STATE = RUNS / "thermal_state.json"


def train_log_path(name: str) -> Path:
    return RUNS / f"{name}_train.log"

INTERVAL = 60              # 每分钟看一次
WINDOW = 40                # 用最近 40 个读数算中位数
WARN_DRIFT = 0.25          # 比历史最好慢 25% -> 记警告
ALERT_DRIFT = 0.50         # 慢 50% -> 记严重（仍然不停训练，只通知）
SIT_RE = re.compile(r"([\d.]+)\s*s/it")


def log(msg: str) -> None:
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def recent_speeds() -> list[float]:
    """从训练日志里抓最近的 s/it 读数——**只要训练阶段的，不要验证阶段的**。

    日志里每轮的结构是：Epoch 表头 -> 训练进度条 -> "Class Images Instances" 表头 -> 验证进度条。
    所以用这两个标志切换状态，验证阶段整段跳过。
    """
    if not TRAIN_LOG.exists():
        return []
    text = TRAIN_LOG.read_text(encoding="utf-8", errors="ignore")
    vals: list[float] = []
    in_val = False
    for chunk in re.split(r"[\r\n]+", text):
        s = chunk.strip()
        if not s:
            continue
        if "Class" in s and "Instances" in s:      # 验证阶段的小表头
            in_val = True
            continue
        if s.startswith("Epoch"):                   # 下一轮开始
            in_val = False
        if in_val:
            continue
        m = SIT_RE.search(s)
        if m:
            vals.append(float(m.group(1)))
    return vals[-WINDOW:]


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {"best": None, "slow_streak": 0, "warned": False}


def save_state(st: dict) -> None:
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    global TRAIN_LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="yolo26n_2560_v2", help="训练 run 名字（决定读哪个日志）")
    args = ap.parse_args()
    TRAIN_LOG = train_log_path(args.name)

    log("=" * 60)
    log("散热旁路监视启动：用训练速度当温度代理（本机读不到 CPU 核心温度）")
    log(f"监控日志: {TRAIN_LOG.name}")
    st = load_state()
    if st.get("best"):
        log(f"沿用历史基线 {st['best']:.2f} s/批")

    while True:
        vals = recent_speeds()
        if len(vals) >= 10:
            cur = statistics.median(vals)
            if st["best"] is None or cur < st["best"]:
                st["best"] = cur
            drift = cur / st["best"] - 1.0

            if drift >= ALERT_DRIFT:
                st["slow_streak"] += 1
            else:
                st["slow_streak"] = 0

            if st["slow_streak"] == 1 or st["slow_streak"] % 5 == 0:
                log(f"速度 {cur:.1f} s/批（历史最好 {st['best']:.1f}）慢 {drift * 100:+.0f}%"
                    f"{'  [持续变慢 ' + str(st['slow_streak']) + ' 分钟]' if drift >= ALERT_DRIFT else ''}")
            elif drift >= WARN_DRIFT and not st["warned"]:
                st["warned"] = True
                log(f"[警告] 训练比历史最好慢了 {drift * 100:.0f}%（现在 {cur:.1f} s/批，"
                    f"最好 {st['best']:.1f}）——可能是散热降频或内存换页")
            elif drift < WARN_DRIFT:
                st["warned"] = False

            if st["slow_streak"] == 30:
                log(f"[严重] 已经连续 30 分钟比历史最好慢 {drift * 100:.0f}%——大概率在降频。"
                    f"本脚本不会停训练，请人工决定（可把电源计划的最大处理器状态调到 90%）")
            save_state(st)
        else:
            log(f"等待训练日志（当前只抓到 {len(vals)} 个读数）")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
