# -*- coding: utf-8 -*-
"""LBD 标注工具（内嵌标注页原型）

直接读写 agent3-debug 识别结果 JSON：改框、改 LBD 名字，另存出一份完整 JSON 给下游程序用。

用法:
    python lbd_annotator.py                 # 打开对话框选 JSON
    python lbd_annotator.py 识别结果.json
    python lbd_annotator.py --selftest      # 无界面自检（改一页 -> 另存 -> 用下游代码验证）

设计要点:
  * 不动原文件。默认「另存为」xxx_annotated.json；「覆盖保存」会先备份 .bak-时间戳。
  * 图片直接从 JSON 里内嵌的 png_base64 解出来，不落地 PNG。
  * 246MB 的 JSON 按字节流式读写：只替换改过的页所在的三个顶层数组，
    其它段落（含没改过的页）逐字节保留原样。
  * Node 框与 ocr_node_name_results 记录一一对应（按 node_bbox 对齐），
    改名写进 final_node_name —— 这正是下游 extract_lines_from_debug 读的字段。
"""
import argparse
import collections
import copy
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

CLASSES = ("Node", "Tracker", "Box")
# 调外部程序（pdftoppm / pdfinfo）时别弹那个黑窗口：Windows 下加 CREATE_NO_WINDOW
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
# Qt 默认单张图片解码上限 256MB：6000x4000(96MB) 没问题，但 333dpi 的
# 11988x7992 要 383MB，会被拒（表现就是"渲染结果读不出来"）。放宽到 1GB。
os.environ.setdefault("QT_IMAGEIO_MAXALLOC", "1024")
TRACKER_SECTION = "yolo_tracker_detection_results"
BOX_SECTION = "yolo_box_detection_results"
OCR_SECTION = "ocr_node_name_results"
SECTION_ORDER = (TRACKER_SECTION, BOX_SECTION, OCR_SECTION)
DEFAULT_CLASS_ID = {"Node": 1, "Tracker": 0, "Box": 0}
WS = b" \t\r\n"
# 没有"默认打开某份文件"这回事了：要么命令行给路径，要么在工具里点「打开 JSON」。
DEFAULT_JSON = ""
ANNOTATOR_VERSION = "0.47"                      # 标注工具自己的版本号
def _build_stamp():
    """这份 exe（或源码）的生成时间 —— 放在窗口标题里，方便确认到底跑的哪一版。"""
    try:
        p = sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__)
        return time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
    except Exception:
        return "?"


BUILD_STAMP = _build_stamp()
# 自检（--smoke / --memtest / --roundtrip）模式：设置文件和渲染缓存都放临时目录。
# 以前自检会读写用户真正的 %LOCALAPPDATA%\LBD标注工具\annotator_settings.json，
# 把「上次打开」覆盖成自检用的假图纸 —— 下次开工具就弹出那张 "LBD SMOKE PAGE" 假图。
TEST_MODE = False
RACK_LEN_TOL = 0.10                             # 支架长度差 ≤10% 算同一类
UPDATE_REPO = "cszmw2k6dk-design/LBD-Annotator"   # 在线更新读这个仓库的 Release


def parse_version(tag):
    """'v0.6' / 'v0.5.1' -> (0, 6) / (0, 5, 1)；认不出来返回 ()。"""
    nums = re.findall(r"\d+", str(tag or ""))
    return tuple(int(n) for n in nums[:3]) if nums else ()


def fetch_json_curl(url, token="", timeout=12, proxy=""):
    """用系统自带的 curl.exe 取 JSON。

    为什么要这个：打包成 exe 之后，防火墙/杀软经常把"未签名 exe 的外连"静默丢掉，
    urllib 会一直等到超时；而 curl.exe 是系统程序，一般放行。
    token 通过 stdin 传（--config -），不出现在命令行里。
    """
    import shutil as _sh
    import subprocess as _sp
    exe = _sh.which("curl") or _sh.which("curl.exe")
    if not exe:
        return None, "系统里没有 curl.exe"
    cfg = ('header = "User-Agent: LBD-Annotator/%s"\n'
           'header = "Accept: application/vnd.github+json"\n' % ANNOTATOR_VERSION)
    if (token or "").strip():
        cfg += 'header = "Authorization: Bearer %s"\n' % token.strip()
    cfg += 'url = "%s"\n' % url
    try:
        args = [exe, "-sS", "-L", "--max-time", str(int(timeout)), "--config", "-"]
        if (proxy or "").strip():
            args += ["--proxy", proxy.strip()]
        p = _sp.run(args,
                    input=cfg.encode("utf-8"), capture_output=True,
                    timeout=timeout + 8, creationflags=_NO_WINDOW)
        if p.returncode == 0 and p.stdout.strip():
            return json.loads(p.stdout.decode("utf-8", "replace")), ""
        return None, "curl 退出码 %s：%s" % (
            p.returncode, (p.stderr.decode("utf-8", "replace") or "").strip()[:200])
    except Exception as e:                     # noqa: BLE001
        return None, "curl 调用失败：%s" % e


def _github_json(url, token="", timeout=6, proxy=""):
    """取 GitHub API 的 JSON：**先 curl.exe**，失败再 urllib。

    为什么 curl 优先：打包成 exe 之后，防火墙/杀软常把"未签名 exe 的外连"静默丢掉，
    urllib 会一直等到超时（白等一整个 timeout）；curl.exe 是系统程序，一般放行。
    实测：先 urllib 要 21 秒才拿到结果，curl 优先后 10 秒左右就有。
    """
    import urllib.request
    info, err_curl = fetch_json_curl(url, token, timeout, proxy)
    if info is not None:
        return info, ""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "LBD-Annotator/%s" % ANNOTATOR_VERSION,
                          "Accept": "application/vnd.github+json"})
        if (token or "").strip():
            req.add_header("Authorization", "Bearer %s" % token.strip())
        opener = (urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
            if (proxy or "").strip() else urllib.request)
        with opener.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")), ""
    except Exception as e:                     # noqa: BLE001
        return None, "curl：%s；urllib：%s" % (err_curl, e)


def git_remote_tags(repo, timeout=15, limit=0):
    """用 git ls-remote 问 GitHub 有哪些版本 tag，**按版本从新到旧**返回（limit=0 全要）。"""
    import subprocess as _sp
    exe = shutil.which("git") or shutil.which("git.exe")
    if not exe:
        return []
    url = "https://github.com/%s.git" % repo
    try:
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        p = _sp.run([exe, "ls-remote", "--tags", url], capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=timeout, env=env,
                    creationflags=_NO_WINDOW)
        if p.returncode != 0:
            return []
        tags = set()
        for line in (p.stdout or "").splitlines():
            m = re.search(r"refs/tags/(v[\d.]+)\s*$", line.strip())
            if m:
                tags.add(m.group(1))
        out = sorted(tags, key=parse_version, reverse=True)
        return out[:limit] if limit else out
    except Exception:                          # noqa: BLE001
        return []


def git_remote_latest_tag(repo, timeout=15):
    """兜底：用 git ls-remote 问 GitHub 有哪些 tag（这条通道和你 clone 用的是同一条，
    国内网络经常"api.github.com 不通、github.com 能通"）。

    返回最新版本 tag（如 "v0.41"）；失败返回空串。
    """
    tags = git_remote_tags(repo, timeout=timeout, limit=1)
    return tags[0] if tags else ""


def head_release_asset(repo, tag, name="LBD.exe", timeout=8, proxy=""):
    """直接 HEAD「github.com/<repo>/releases/download/<tag>/<name>」，看这个版本有没有安装包。

    为什么不用 API：国内直连 api.github.com 时通时不通，而 github.com（网页那个域名）
    通常好得多。公开仓库的安装包地址是可拼的，HEAD 一下就知道在不在、多大。
    返回 (在不在, 字节数)。
    """
    exe = shutil.which("curl") or shutil.which("curl.exe")
    if not (exe and tag):
        return False, 0
    url = "https://github.com/%s/releases/download/%s/%s" % (repo, tag, name)
    args = [exe, "-sS", "-I", "-L", "--max-time", str(int(timeout)),
            "-H", "User-Agent: LBD-Annotator/%s" % ANNOTATOR_VERSION]
    if (proxy or "").strip():
        args += ["--proxy", proxy.strip()]
    args.append(url)
    try:
        p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout + 5,
                           creationflags=_NO_WINDOW)
        code, size = 0, 0
        for line in (p.stdout or "").splitlines():
            m = re.match(r"HTTP/\S+\s+(\d{3})", line.strip())
            if m:
                code = int(m.group(1))
            m2 = re.match(r"[Cc]ontent-[Ll]ength:\s*(\d+)", line.strip())
            if m2:
                size = int(m2.group(1))
        return (200 <= code < 300), size
    except Exception:                          # noqa: BLE001
        return False, 0


def fetch_latest_release(repo, token="", timeout=6, rounds=1, on_round=None, proxy="",
                         on_step=None):
    """查 GitHub 的 release，返回 (info, 错误文本)。不写任何文件。

    顺序很讲究（实测按这个来最快也最稳，几秒出结果）：
      ① **git ls-remote** 拿最新 tag —— 和你 clone 同一条通道，实测 2 秒；
      ② 直接 **HEAD 那个 tag 的安装包**（github.com 域名，公开仓库地址可拼）——
         有的话当场就能给"下载到程序目录"，完全绕开 api.github.com；
      ③ 前面拿不到才回退 **api.github.com**（curl.exe 优先 + 匿名优先），
         它知道"哪些版本真的挂了 exe"，能把没挂包的版本跳过。
    """
    import queue as _queue
    import threading as _threading
    q = _queue.Queue()
    errs, lock = [], _threading.Lock()

    def _say(msg):
        with lock:
            errs.append(str(msg))

    def _step(name):
        if on_step:
            try:
                on_step(name)
            except Exception:
                pass

    def worker_git():
        """① git ls-remote 拿 tag（和 clone 同一条通道，实测 2 秒）
           ② 从新到旧 HEAD 探测"哪个版本挂了安装包"（github.com，公开仓库地址可拼）。"""
        try:
            _step("① 问 github.com：有哪些版本（git ls-remote）")
            tags = git_remote_tags(repo)
            if not tags:
                _say("git ls-remote 没拿到 tag")
                return
            _step("② 问 github.com：哪个版本挂了安装包（HEAD 探测）")
            # 最新那个可能还没挂包（上传中断之类）-> 往前找几个，最多试 6 个
            for tag in tags[:6]:
                ok, size = head_release_asset(repo, tag, timeout=8, proxy=proxy)
                if ok:
                    dl = ("https://github.com/%s/releases/download/%s/LBD.exe"
                          % (repo, tag))
                    asset = {"name": "LBD.exe", "size": size, "browser_download_url": dl}
                    q.put({"tag_name": tag, "assets": [asset], "_lbd_asset": asset,
                           "_from": "git"})
                    return
                _say("%s 没挂 LBD.exe" % tag)
            q.put({"tag_name": tags[0], "assets": [], "_lbd_asset": None, "_from": "git"})
        except Exception as e:                 # noqa: BLE001
            _say("git 通道：%s" % e)

    def worker_api():
        """api.github.com —— **不再使用**。

        用户机器上实测：github.com / git 都通，但 api.github.com 被网络挡住（诊断就停在这一步）。
        而"哪个版本挂了安装包"用 HEAD 探测就能得到，够用了，所以干脆不碰 api，
        免得再因为它卡住整条检查。留这个空函数只是为了让 diff 小、逻辑位置清楚。
        """
        return
        # ---- 以下为旧实现，保留备查，不再执行 ----
        _step("③ 问 api.github.com（可能慢，①② 好就不用等它）")
        urls = ["https://api.github.com/repos/%s/releases?per_page=10" % repo,
                "https://api.github.com/repos/%s/releases/latest" % repo]
        for rd in range(max(1, int(rounds))):
            if on_round:
                try:
                    on_round(rd + 1)
                except Exception:
                    pass
            # 第 1 轮**不带 token**（和主程序 Voltage-CAD MAP 同一条通道：仓库公开，
            # 匿名就能读；带 token 反而多一层可能出问题的地方），第 2 轮才带。
            tok_try = "" if rd == 0 else token
            for url in urls:
                data, err = _github_json(url, tok_try, timeout, proxy)
                if data is None:
                    _say(err)
                    continue
                rels = data if isinstance(data, list) else [data]
                rels = [r for r in rels if isinstance(r, dict) and not r.get("draft")]
                if not rels:
                    _say("这个仓库还没有 Release")
                    continue
                with_exe = []
                for r in rels:
                    hit = None
                    for a in (r.get("assets") or []):
                        if str(a.get("name") or "").lower().endswith(".exe"):
                            hit = a
                            break
                    r["_lbd_asset"] = hit
                    if hit:
                        with_exe.append(r)
                if with_exe:
                    with_exe.sort(key=lambda r: (parse_version(r.get("tag_name")),
                                                 str(r.get("published_at") or "")),
                                  reverse=True)
                    q.put(with_exe[0])
                else:
                    q.put(rels[0])
                return

    # 只跑 ①②（git + github.com），**完全不碰 api.github.com**：
    # 用户机器上 api 那个域名被挡（诊断停在第 3 步），而 github.com 是通的。
    worker_git()
    deadline, best, t_first = time.time() + 30.0, None, 0.0

    def _drain(max_wait=0.5):
        """把队列里已有的结果收下来（快通道最多等 max_wait 秒）。"""
        nonlocal best, t_first
        end = time.time() + max_wait
        while time.time() < end:
            try:
                info = q.get(timeout=min(end - time.time(), 0.2))
            except _queue.Empty:
                continue
            if not isinstance(info, dict):
                continue
            if info.get("_lbd_asset"):
                return info
            if best is None:
                best, t_first = info, time.time()
        return None

    got = _drain(1.0)
    if got is not None:
        return got, ""                       # ①② 就搞定了
    # 走到这儿说明①②没给出可下载的结果（最新几个版本都没挂包）——
    # 现在不再去问 api（那域名在用户机器上被挡），直接把手上的 tag 报上去
    _threading.Thread(target=worker_api, daemon=True).start()    # 空函数，立即返回
    while time.time() < deadline:
        try:
            info = q.get(timeout=min(deadline - time.time(), 3.0))
        except _queue.Empty:
            if best is not None and time.time() - t_first > 2.0:
                return best, ""
            continue
        if isinstance(info, dict):
            if info.get("_lbd_asset"):
                return info, ""
            if best is None:
                best, t_first = info, time.time()
    if best is not None:
        return best, ""
    return None, "；".join(errs)[:400]


def windows_git_credential(host="github.com"):
    """读本机 git 存在 Windows 凭据管理器里的 GitHub 凭据（当前用户，读不到就返回空串）。

    仓库是私有的，匿名读 API 会被 GitHub 回 404；而能 clone 这个仓库的机器上，
    git 早就存好凭据了，直接借用它，省得再让人去申请 token。
    """
    if os.name != "nt":
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        class CREDENTIAL(ctypes.Structure):
            _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
                        ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
                        ("LastWritten", wintypes.FILETIME),
                        ("CredentialBlobSize", wintypes.DWORD),
                        ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
                        ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
                        ("Attributes", ctypes.c_void_p),
                        ("TargetAlias", wintypes.LPWSTR), ("UserName", wintypes.LPWSTR)]

        adv = ctypes.windll.advapi32
        ptr = ctypes.POINTER(CREDENTIAL)()
        target = "git:https://%s" % host
        if not adv.CredReadW(ctypes.c_wchar_p(target), 1, 0, ctypes.byref(ptr)):
            return ""
        try:
            cred = ptr.contents
            blob = ctypes.wstring_at(
                ctypes.cast(cred.CredentialBlob, ctypes.c_wchar_p),
                max(0, cred.CredentialBlobSize // 2))
        finally:
            adv.CredFree(ptr)
        m = re.search(r"password=([^\s;]+)", blob)
        return m.group(1) if m else blob.strip("\x00 \r\n\t")
    except Exception:                          # noqa: BLE001
        return ""


def update_token(settings=None):
    """在线更新用的 GitHub token：设置里 > 环境变量 > 本机 git 凭据。"""
    tok = ""
    if settings:
        tok = str(settings.get("update_token") or "").strip()
    if not tok:
        tok = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    if not tok:
        tok = windows_git_credential()
    return tok


def update_proxy(settings=None):
    """更新检查/下载走的代理：设置里的 update_proxy > 环境变量 HTTPS_PROXY/HTTP_PROXY。

    国内直连 api.github.com 常常时通时不通；如果你本机有代理（Clash/V2Ray 之类），
    在 annotator_settings.json 里写一行 "update_proxy": "http://127.0.0.1:7890" 就行。
    """
    p = ""
    if settings:
        p = str(settings.get("update_proxy") or "").strip()
    if not p:
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
            p = (os.environ.get(k) or "").strip()
            if p:
                break
    return p


def _run_with_timeout(fn, timeout):
    """在子线程里跑 fn，最多等 timeout 秒；超时就返回一句"超时"，绝不无限等。

    为什么每一步都要这样包一层：用户机器上"卡在界面"到底是什么卡住了，
    光看错误看不出来（DNS、spawn 子进程、TLS 握手都可能卡），包一层之后
    报告里会明确写"（这步超时）"。
    """
    import threading as _th
    box = {}

    def run():
        try:
            box["v"] = fn()
        except Exception as e:                 # noqa: BLE001
            box["v"] = "异常：%r" % e

    t = _th.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return "⚠ 这一步超过 %ds 没反应（卡在这里）" % int(timeout)
    return box.get("v", "（没结果）")


def update_diagnose(repo=None, settings=None, on_step=None):
    """把「检查更新」的每一步都跑一遍并记下来，返回一份可以直接发给人的报告文本。

    为什么要有这个：用户机器上"一直等 / 卡住"时，光看界面分不清到底是
    DNS 慢、防火墙拦、杀毒软件拦、代理没配还是凭据有问题 —— 这份报告里
    每一步都有**耗时**和**原始错误**，对照着看一目了然。
    """
    import platform
    import socket
    repo = repo or UPDATE_REPO
    out = ["LBD 标注工具 — 更新诊断",
           "时间：%s" % time.strftime("%Y-%m-%d %H:%M:%S"),
           "工具版本：v%s" % ANNOTATOR_VERSION,
           "仓库：%s" % repo,
           "系统：%s" % platform.platform(),
           "打包成 exe：%s" % bool(getattr(sys, "frozen", False)),
           "程序路径：%s" % (os.path.abspath(sys.executable) if getattr(sys, "frozen", False)
                            else os.path.abspath(__file__)),
           "代理设置：%r（环境变量 %r）"
           % (str((settings or {}).get("update_proxy") or ""),
              os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""),
           ""]

    def step(name, fn, limit=20):
        if on_step:
            try:
                on_step(name)
            except Exception:
                pass
        t0 = time.time()
        msg = _run_with_timeout(fn, limit)
        out.append("[%s] %-30s %6.1f 秒  %s"
                   % (time.strftime("%H:%M:%S"), name, time.time() - t0, msg))
        return msg

    def _dns():
        t = time.time()
        try:
            infos = socket.getaddrinfo("github.com", 443, proto=socket.IPPROTO_TCP)
            ips = sorted({i[4][0] for i in infos})[:3]
            return "github.com -> %s（%.1fs）" % (", ".join(ips), time.time() - t)
        except Exception as e:                 # noqa: BLE001
            return "DNS 失败：%r" % e

    step("DNS 解析 github.com", _dns, limit=15)

    def _curlver():
        exe = shutil.which("curl") or shutil.which("curl.exe")
        if not exe:
            return "系统里没有 curl.exe"
        try:
            p = subprocess.run([exe, "--version"], capture_output=True, text=True,
                               timeout=10, creationflags=_NO_WINDOW)
            return (p.stdout or "").splitlines()[0] if p.stdout else "（没输出）"
        except Exception as e:                 # noqa: BLE001
            return "调不动 curl.exe：%r" % e

    step("系统 curl.exe 可用性", _curlver, limit=15)

    def _gitver():
        exe = shutil.which("git") or shutil.which("git.exe")
        if not exe:
            return "系统里没有 git（更新会用 API 那条路）"
        try:
            p = subprocess.run([exe, "--version"], capture_output=True, text=True,
                               timeout=10, creationflags=_NO_WINDOW)
            return (p.stdout or "").strip() or "（没输出）"
        except Exception as e:                 # noqa: BLE001
            return "调不动 git.exe：%r" % e

    step("系统 git.exe 可用性", _gitver, limit=15)

    tag = [""]

    def _gitremote():
        t = time.time()
        tag[0] = git_remote_latest_tag(repo)
        return ("最新 tag = %s" % tag[0]) if tag[0] else "没拿到 tag（通道不通）"

    step("① git ls-remote 问 tag", _gitremote, limit=25)

    def _head():
        if not tag[0]:
            return "跳过（不知道 tag）"
        ok, size = head_release_asset(repo, tag[0], timeout=8)
        return ("安装包在，%.1f MB" % (size / 1048576.0)) if ok else "那个 tag 没挂 LBD.exe"

    step("② HEAD 安装包（github.com）", _head, limit=20)

    def _final():
        info, err = fetch_latest_release(repo, windows_git_credential(), timeout=6, rounds=1)
        if info:
            a = info.get("_lbd_asset") or {}
            return "成功：tag=%s 资产=%s（来自 %s）" % (info.get("tag_name"), a.get("name"),
                                                    info.get("_from", "api"))
        return "失败：%s" % str(err)[:200]

    step("③ 整体检查（工具实际用的）", _final, limit=40)
    out.append("")
    out.append("说明：检查更新**只用 github.com 和 git**（api.github.com 那个域名在不少网络里被挡，")
    out.append("     所以工具从 v0.47 起完全不碰它）。上面哪一步不通，就是哪一层被防火墙/杀软拦了。")
    return "\n".join(out)


OCR_BOX_SCRIPT = r'''
import json, os, re, sys
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

# 一张图最多拼几块。拼太多会读乱 —— 实测（Steel River 第 1 页，20 个 LBD 框）：
#   每张 30 块：20 个框只读对 2 个（读出来是一堆不相干的字，用户看到的就是这个）
#   每张 5 块 ：20 个框全对
# 原因：拼成一张 2000x7000 的超长图后，检出/识别出来的框会串行、串块，认出来的字就跑了。
GROUP = 5
GAP = 12
LBD_RE = re.compile(r"INV\s*\d+\s*[A-Z]\s*\d+\s*[-_ ]?\s*LBD\s*[-_ ]?\s*\d+", re.I)

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
img = Image.open(cfg["img"]).convert("RGB")
sx = img.width / float(cfg.get("page_w") or img.width)
sy = img.height / float(cfg.get("page_h") or img.height)


def crop_of(b, mw, mh):
    """按框裁一块。mw/mh 是外扩比例，0 就是"就按这个框裁"（实测最准：
    框挨着框时，往外扩会把隔壁框的编号一起吃进来）。"""
    x1, y1, x2, y2 = b["bbox"]
    X1, Y1, X2, Y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
    w, h = max(1.0, X2 - X1), max(1.0, Y2 - Y1)
    mx, my = w * mw + 3, h * mh + 3
    box = (max(0, int(X1 - mx)), max(0, int(Y1 - my)),
           min(img.width, int(X2 + mx)), min(img.height, int(Y2 + my)))
    c = img.crop(box)
    if c.width < 8 or c.height < 8:
        return None
    b["h_raw"] = min(c.size)                   # 原图里这一块有多高（判断分辨率够不够）
    raw_w, raw_h = c.size                     # 旋转前的大小，反算坐标要用
    if c.height > c.width * 1.15:            # 竖排：转成横的（180 度交给角度分类器）
        c = c.rotate(90, expand=True)
        rot = True
    else:
        rot = False
    z = 1
    if min(c.size) < 40:                     # 太小就放大，OCR 才认得出
        z = max(2, int(40 / max(1, min(c.size))) + 1)
        c = c.resize((c.width * z, c.height * z), Image.LANCZOS)
    # 记下这一块的几何，等会儿把"识别到的文字在这张拼图上的坐标"换算回页面坐标
    b["_geo"] = (box[0], box[1], raw_w, raw_h, rot, z)
    return c


eng = RapidOCR()


def ocr_items(items, dbg_name=""):
    """items = [(框, 裁好的图)] -> {ix: (文字, 置信度)}。竖着拼一张图，一次识别。"""
    if not items:
        return {}
    W = max(c.width for _b, c in items) + 24
    H = sum(c.height + GAP for _b, c in items) + GAP
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    ys, y = {}, GAP
    for b, c in items:
        sheet.paste(c, (12, y))
        ys[b["ix"]] = (y, y + c.height)
        y += c.height + GAP
    tmp = os.path.join(os.path.dirname(os.path.abspath(sys.argv[1])), "_ocr_sheet.png")
    sheet.save(tmp)
    dbg = cfg.get("debug_dir") or ""
    if dbg and dbg_name:
        try:
            os.makedirs(dbg, exist_ok=True)
            sheet.save(os.path.join(dbg, "sheet_p%s_%s.png" % (cfg.get("page", 0), dbg_name)))
        except OSError:
            pass
    res, _el = eng(tmp)
    try:
        os.remove(tmp)
    except OSError:
        pass
    got = {}
    pos = {}
    lbox = {}
    for box, t, s in (res or []):
        xs1 = min(p[0] for p in box)
        xs2 = max(p[0] for p in box)
        ys1 = min(p[1] for p in box)
        ys2 = max(p[1] for p in box)
        yy = (ys1 + ys2) / 2.0
        xx = (xs1 + xs2) / 2.0
        for ix, (a, b2) in ys.items():
            if a - 4 <= yy <= b2 + 4:
                old = got.get(ix)
                if old is None or float(s) > old[1]:
                    got[ix] = (t, float(s))
                    pos[ix] = to_page(ix, xx, yy, a)
                    lbox[ix] = to_page_box(ix, xs1, ys1, xs2, ys2, a)
                break
    return got, pos, lbox


def to_page(ix, xs, ys_, paste_y):
    """拼图上的一个点 -> 页面坐标（和 pm.shapes 里 bbox 同一个坐标系）。

    一路反着来：去掉拼图留白 -> 去掉放大倍数 -> 反掉那次 90° 旋转 -> 加上裁块原点 -> 除以缩放。
    （PIL rotate(90, expand=True) 的映射实测是 原(x,y) -> 新(y, 原宽-1-x)）
    """
    geo = geo_of.get(ix)
    if not geo:
        return None
    x0, y0, raw_w, raw_h, rot, z = geo
    cx = (xs - 12.0) / float(z)
    cy = (ys_ - paste_y) / float(z)
    if rot:
        px_, py_ = raw_w - 1 - cy, cx
    else:
        px_, py_ = cx, cy
    return ((x0 + px_) / sx, (y0 + py_) / sy)


def to_page_box(ix, xs1, ys1, xs2, ys2, paste_y):
    """拼图上一个矩形 -> 页面上的矩形（四个角都反算一遍再取外框，旋转也不怕）。"""
    pts = [to_page(ix, xs1, ys1, paste_y), to_page(ix, xs2, ys1, paste_y),
           to_page(ix, xs1, ys2, paste_y), to_page(ix, xs2, ys2, paste_y)]
    pts = [p for p in pts if p]
    if not pts:
        return None
    return (min(p[0] for p in pts), min(p[1] for p in pts),
            max(p[0] for p in pts), max(p[1] for p in pts))


# 1) 主跑：按框本身裁，5 块拼一张
crops = []
for b in cfg["boxes"]:
    c = crop_of(b, 0.0, 0.0)
    if c is not None:
        crops.append((b, c))
geo_of = {b["ix"]: b.get("_geo") for b, _c in crops}
out = {}
for gi in range(0, len(crops), GROUP):
    got_g, pos_g, box_g = ocr_items(crops[gi:gi + GROUP], "g%d" % (gi // GROUP + 1))
    for ix, v in got_g.items():
        out[ix] = (v[0], v[1], pos_g.get(ix), box_g.get(ix))

# 2) 补救：没读到标准编号的框单独再跑一次；还不行就把框往外扩 25%/15% 再单独跑一次
#    （标签压在框线上、或者框裁得太紧把字切掉的情况，靠这一步捞回来）
bad = [b for b in cfg["boxes"]
       if not (out.get(b["ix"]) and LBD_RE.search((out[b["ix"]][0] or "").replace(" ", "")))]
for n, b in enumerate(bad[:80]):
    for mw, mh, tag in ((0.0, 0.0, "r1"), (0.25, 0.15, "r2")):
        c = crop_of(b, mw, mh)
        if c is None:
            continue
        geo_of[b["ix"]] = b.get("_geo")
        got2, pos2, box2 = ocr_items([(b, c)], "p%s_%s_%d" % (cfg.get("page", 0), tag, n))
        v = got2.get(b["ix"])
        if v and LBD_RE.search((v[0] or "").replace(" ", "")):
            out[b["ix"]] = (v[0], v[1], pos2.get(b["ix"]), box2.get(b["ix"]))
            break

_hm = {b["ix"]: b.get("h_raw") for b in cfg["boxes"]}
print("@@" + json.dumps([{"ix": k, "text": v[0], "score": v[1], "h": _hm.get(k),
                          "pos": ([round(float(v[2][0])), round(float(v[2][1]))]
                                  if v[2] else None),
                          "bbox": ([round(float(x)) for x in v[3]] if v[3] else None)}
                         for k, v in sorted(out.items())], ensure_ascii=False))
'''


def parse_page_spec(text, available):
    """把 "3-8,12" / "3，8" 这种页码写法解析成页号列表（只保留文档里真有的页）。

    返回空列表 = 没指定有效页。页码用的是 JSON 里的页号。
    """
    avail = set(available)
    out = set()
    for part in re.split(r"[,，、;；\s]+", str(text or "").strip()):
        if not part:
            continue
        m = re.match(r"^(\d+)\s*[-~－—]\s*(\d+)$", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            for p in range(min(a, b), max(a, b) + 1):
                if p in avail:
                    out.add(p)
            continue
        if part.isdigit() and int(part) in avail:
            out.add(int(part))
    return sorted(out)


def peek_data_yaml(path):
    """粗略读一下 data.yaml 的类别名和数据集根目录，返回 (names, path)。

    训练前拿来给人核对用：如果读出来是 person/dog/horse 这种，就说明选错文件了
    （ultralytics 的默认数据集 coco8 就是那些类别）。
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except Exception:                          # noqa: BLE001
        return [], ""
    names = []
    m = re.search(r"names\s*:\s*\[([^\]]*)\]", txt)
    if m:
        names = [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
    else:
        blk = re.search(r"names\s*:\s*\n((?:[ \t]+\d+[ \t]*:[^\n]*\n?)+)", txt)
        if blk:
            names = [ln.split(":", 1)[1].strip().strip("'\"")
                     for ln in blk.group(1).splitlines() if ":" in ln]
    root = ""
    mr = re.search(r"^[ \t]*path[ \t]*:[ \t]*(.+)$", txt, re.M)
    if mr:
        root = mr.group(1).strip().strip("'\"")
    return names, root


def find_pythons():
    """机器上可能能用的 Python 解释器（训练/推理要用）。

    注意：Python 3.14 现在装不上 torch，所以优先列 3.12 / 3.11。
    """
    out = []
    try:
        w = shutil.which("python")
        if w:
            out.append(w)
    except Exception:
        pass
    bases = ["C:\\", os.environ.get("LOCALAPPDATA") or "",
             os.path.join(os.environ.get("USERPROFILE") or "", "anaconda3"),
             os.path.join(os.environ.get("USERPROFILE") or "", "miniconda3")]
    for base in bases:
        if not base:
            continue
        for ver in ("312", "311", "313", "310", "39"):
            for p in (os.path.join(base, "Python%s" % ver, "python.exe"),
                      os.path.join(base, "Programs", "Python", "Python%s" % ver, "python.exe"),
                      os.path.join(base, "python%s" % ver, "python.exe"),
                      os.path.join(base, "envs", "py%s" % ver, "python.exe")):
                if os.path.exists(p) and p not in out:
                    out.append(p)
        p = os.path.join(base, "python.exe")
        if os.path.exists(p) and p not in out:
            out.append(p)
    return out

# ------------------------------------------------------- 高分辨率底图（重新渲染 PDF）
POPPLER_HINTS = (
    # 便携位置：放个 poppler 文件夹，或直接把 pdftoppm.exe 放程序旁边，都认
    r"poppler\pdftoppm.exe",
    r"pdftoppm.exe",
    r"poppler\Library\bin\pdftoppm.exe",
    r"%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies"
    r"\native\poppler\Library\bin\pdftoppm.exe",
    r"C:\Program Files\poppler\Library\bin\pdftoppm.exe",
    r"C:\Program Files (x86)\poppler\Library\bin\pdftoppm.exe",
    r"C:\poppler\Library\bin\pdftoppm.exe",
)


def app_dir():
    """程序所在目录：打包成 exe 后是 exe 的目录，源码运行时是脚本目录。

    设置文件、渲染缓存都放这儿，打包后才是「跟着程序走」的便携目录 ——
    不会写进 exe 解包出来的临时目录（那里面一关就没了）。
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def find_pdftoppm():
    for h in POPPLER_HINTS:
        p = os.path.expandvars(h)
        if not os.path.isabs(p):
            p = os.path.join(app_dir(), p)          # 相对路径按程序目录找
        if os.path.exists(p):
            return p
    return shutil.which("pdftoppm") or ""


def find_poppler(exe):
    """同一个 poppler 目录下的其它工具（pdfinfo 等）。"""
    for h in POPPLER_HINTS:
        p = os.path.expandvars(h)
        if not os.path.isabs(p):
            p = os.path.join(app_dir(), p)
        cand = os.path.join(os.path.dirname(p), exe)
        if os.path.exists(cand):
            return cand
    return shutil.which(exe) or ""


# ------------------------------------------------------- AI 识别的输入图
# 喂 YOLO 之前先把渲染图平滑缩到这个尺寸（长边）。别直接喂原始大图：
# ultralytics 内部是用 cv2 的一步线性插值缩到 imgsz 的（INTER_LINEAR，只采 2x2 邻域），
# 9000x6000 缩到 1920 是 4.7 倍，Tracker 那种只有 20 多像素宽的细线会被"跳过"，
# 实测 Tracker 召回从 0.68 掉到 0.44、整体 F1 从 0.835 掉到 0.655。
# 训练集就是按 2560 预处理的，喂同样尺寸，工具里的精度才和验证指标对得上。
MODEL_INPUT_SIDE = 2560


# ------------------------------------------------------- 边缘吸附
# 模型给的是"大概位置"，框的边常常差半个到一两个像素。CAD 图是纯线条、
# 元素的边界都是实打实的墨线，所以可以在预测边的附近找那条最黑的线，
# 把边吸过去；再对峰值做加权重心，拿到亚像素位置。
# 阈值 need_contrast 用来兜底：附近没有明显墨线（空洞、误检）就原样不动。
#
# ⚠ 暂未接入识别流程。实测（40 张高清验证图）：Tracker 的标注本来就已经贴在
#   图纸边界上（中位差 0.1 px），吸了没收益；Node 的标注没按几何边界画
#   （中位差 1.3 px，只有 15.7% 在 1 px 内），吸过去反而把框拽偏
#   （平均 IoU 0.937→0.814）。留着是准备做"标注规范化"用的，别直接接到推理后。
SNAP_RATIO = 0.25          # 搜索半径 = 框宽/高的这个比例
SNAP_MIN_R = 2.0           # 但至少 ±2 像素
SNAP_MAX_R = 30.0          # 最多 ±30 像素（别吸到隔壁去）
SNAP_MIN_CONTRAST = 8.0    # 峰值要比窗口背景黑这么多才算一条线


def _snap_edge(prof, pos, radius, need_contrast=SNAP_MIN_CONTRAST):
    """把 pos 吸到 prof 上附近最黑的那条线，返回亚像素位置；找不到就原样返回。"""
    import numpy as np
    n = prof.shape[0]
    lo = max(0, int(np.floor(pos - radius)))
    hi = min(n - 1, int(np.ceil(pos + radius)))
    if hi - lo < 2:
        return float(pos)
    seg = prof[lo:hi + 1].astype(np.float32)
    i = int(np.argmax(seg))
    peak = float(seg[i])
    base = float(np.percentile(seg, 50))
    if peak - base < need_contrast:
        return float(pos)
    a, b = max(0, i - 1), min(len(seg) - 1, i + 1)
    wgt = np.clip(seg[a:b + 1] - base, 0, None)
    if wgt.sum() <= 0:
        return float(pos)
    idx = np.arange(a, b + 1)
    return float(lo + (wgt * idx).sum() / wgt.sum())


def _snap_edge_step(prof, pos, radius, need=6.0, w=3):
    """在 prof 上找最明显的"亮度台阶"（一侧亮、一侧暗）——填充块的边界是台阶，不是黑线。"""
    import numpy as np
    n = prof.shape[0]
    lo = max(0, int(np.floor(pos - radius)))
    hi = min(n - 1, int(np.ceil(pos + radius)))
    if hi - lo < 2 * w + 1:
        return float(pos)
    p = prof.astype(np.float32)
    vals = []
    for i in range(lo, hi + 1):
        a = p[max(0, i - w):i]
        b = p[i:i + w]
        if a.size == 0 or b.size == 0:
            vals.append(0.0)
            continue
        vals.append(abs(float(b.mean()) - float(a.mean())))
    vals = np.asarray(vals, dtype=np.float32)
    j = int(np.argmax(vals))
    if float(vals[j]) < need:
        return float(pos)
    a, b = max(0, j - 1), min(len(vals) - 1, j + 1)
    wgt = np.clip(vals[a:b + 1], 0, None)
    if wgt.sum() <= 0:
        return float(pos)
    k = np.arange(a, b + 1)
    return float(lo + (wgt * k).sum() / wgt.sum())


def _snap_edge_grad(prof, pos, radius, need=4.0):
    """用一阶差分找最强边界（对细线和台阶都敏感）。"""
    import numpy as np
    n = prof.shape[0]
    lo = max(1, int(np.floor(pos - radius)))
    hi = min(n - 1, int(np.ceil(pos + radius)))
    if hi - lo < 2:
        return float(pos)
    p = prof.astype(np.float32)
    g = np.abs(np.diff(p[lo - 1:hi + 1]))
    j = int(np.argmax(g))
    if float(g[j]) < need:
        return float(pos)
    a, b = max(0, j - 1), min(len(g) - 1, j + 1)
    wgt = np.clip(g[a:b + 1], 0, None)
    if wgt.sum() <= 0:
        return float(pos)
    k = np.arange(a, b + 1)
    return float(lo - 1 + (wgt * k).sum() / wgt.sum() + 0.5)


def snap_boxes_to_ink(gray, boxes, mode="step"):
    """把每个框的四条边吸到图纸墨线上。

    gray: (H, W) uint8 灰度图（越暗越像线），坐标系要和 boxes 一致
    boxes: [(类, x1, y1, x2, y2), ...]
    返回同样格式的新列表；吸附后不合法（宽或高 <=1）的就保持原样。
    """
    import numpy as np
    if gray is None or getattr(gray, "size", 0) == 0 or not boxes:
        return boxes
    edge_fn = {"peak": _snap_edge, "step": _snap_edge_step, "grad": _snap_edge_grad}[mode]
    H, W = int(gray.shape[0]), int(gray.shape[1])
    if H < 4 or W < 4:
        return boxes
    dark = 255.0 - np.asarray(gray, dtype=np.float32)
    out = []
    for item in boxes:
        cls, x1, y1, x2, y2 = item[0], *[float(v) for v in item[1:5]]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        rx = min(max(SNAP_MIN_R, SNAP_RATIO * w), SNAP_MAX_R)
        ry = min(max(SNAP_MIN_R, SNAP_RATIO * h), SNAP_MAX_R)
        ya = int(max(0, min(H - 1, y1)))
        yb = int(max(ya + 1, min(H, y2)))
        xa = int(max(0, min(W - 1, x1)))
        xb = int(max(xa + 1, min(W, x2)))
        colprof = dark[ya:yb, :].mean(axis=0)     # 每列有多黑
        rowprof = dark[:, xa:xb].mean(axis=1)     # 每行有多黑
        nx1 = edge_fn(colprof, x1, rx)
        nx2 = edge_fn(colprof, x2, rx)
        ny1 = edge_fn(rowprof, y1, ry)
        ny2 = edge_fn(rowprof, y2, ry)
        if nx2 - nx1 >= 1.0 and ny2 - ny1 >= 1.0:
            out.append((cls, nx1, ny1, nx2, ny2, *item[5:]))
        else:
            out.append(tuple(item))
    return out


def qimage_to_gray(qimg):
    """QImage -> (H, W) uint8 灰度数组（给 snap_boxes_to_ink 用）。"""
    import numpy as np
    from PySide6.QtGui import QImage
    if qimg.isNull():
        return None
    if qimg.format() != QImage.Format.Format_Grayscale8:
        qimg = qimg.convertToFormat(QImage.Format.Format_Grayscale8)
    w, h = qimg.width(), qimg.height()
    bpl = qimg.bytesPerLine()
    buf = np.frombuffer(qimg.constBits(), dtype=np.uint8)
    if buf.size < bpl * h:
        return None
    return np.ascontiguousarray(buf[:bpl * h].reshape(h, bpl)[:, :w])


# ------------------------------------------------------- 清理多余/重复框
# 只删"模型画的框"（source == "model"），人工框默认一律不动。
# 四层判据，前面命中就不再往下判：
#   1) 人工优先：和人工框同类别重叠 >= KEEP_MANUAL_IOU 的模型框 -> 删
#   2) 同编号：两个模型框的文字编号一样 -> 留置信度高的
#   3) 高重叠：两个模型框同类别重叠 >= DUP_IOU -> 留面积小的（通常更贴）
#   4) 形状离谱：Tracker 应当又细又高，宽/高 > TRACKER_AR 的直接删
# 门槛为什么取 0.3：先看两条实测
#   1) 同一个物体被两次识别画出来，重叠分布很散：0.9~1.0 占 72.5%，0.8~0.9 占
#      24.9% —— 门槛设 0.9 会漏掉近三成重复框（"一个位置两个标注"）。
#   2) 现有 76752 个标注框里，同类框互相重叠的分布是：IoU<0.2 有 1287 对（相邻的），
#      IoU 0.2~0.7 **一对都没有**，IoU>0.7 有 47 对（真重复）。
# 也就是说 0.2~0.7 这一段是空的，门槛取 0.3 既抓得到"重叠一半面积"的重复框
# （那种 IoU 其实只有 0.33），又不会碰到任何正常相邻的框。
KEEP_MANUAL_IOU = 0.3
DUP_IOU = 0.3
TRACKER_AR = 0.1
# 同类别的两个框，如果一个有 90% 以上落在另一个里面，也算重复（"套着"的那种）。
# 依据：现有 76752 个标注框里，同类嵌套的对数是 0 —— 所以这条不会误伤。
CONTAIN_RATIO = 0.9


def _bbox_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def _bbox_area(b):
    return max(0.0, abs(b[2] - b[0])) * max(0.0, abs(b[3] - b[1]))


def _is_duplicate_pair(a, b):
    """两个同类框算不算"画重了"：重叠够大，或者一个几乎完全套在另一个里面。"""
    if _bbox_iou(a, b) >= DUP_IOU:
        return True
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return False
    smaller = min(_bbox_area(a), _bbox_area(b))
    return smaller > 0 and inter / smaller >= CONTAIN_RATIO


def _is_model_shape(s):
    return (s.get("source") or "") == "model"


def model_meta(model_path):
    """读模型旁边的 meta.txt（导出模型时写的）：imgsz / classes / 来源。

    有了它，"模型是 2560 训的、工具里 imgsz 还写着 1920"这种错就不会再犯。
    """
    try:
        p = os.path.join(os.path.dirname(os.path.abspath(str(model_path))), "meta.txt")
        if not os.path.exists(p):
            return {}
        out = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                k, _, v = line.strip().partition("=")
                if k.strip():
                    out[k.strip().lower()] = v.strip()
        return out
    except Exception:
        return {}


def _has_read_name(s):
    """框里真的读到过编号文字（排除"按标签表位置补的号"——那个是猜的）。"""
    return bool((s.get("name") or "").strip()) and not s.get("_auto")


def clean_shapes(shapes, keep_manual=True, use_number=True, use_shape=True,
                 clean_manual=False):
    """清理多余的框。返回 (保留的框列表, 统计)。

    默认只删模型画的框（人工框不动）。clean_manual=True 时多做一遍：
    同类别的**人工框**之间如果互相重叠 >= DUP_IOU，也只留一个
    （优先留框里真读到过编号的，其次留置信度高的，最后留大的）。
    """
    stats = {"total": len(shapes), "manual": 0, "kept_manual": 0, "by_manual": 0,
             "by_number": 0, "by_overlap": 0, "by_shape": 0, "manual_dup": 0,
             "removed": 0, "kept": 0}
    manual = [s for s in shapes if not _is_model_shape(s)]
    models = [s for s in shapes if _is_model_shape(s)]
    stats["manual"] = len(manual)
    # 注意：这里不能在 clean_manual=True 时提前返回 —— 否则"整页都是手工框"
    # 的情况（没跑过识别、只想清理旧标注）就永远走不到第 5 步。
    if not models and not clean_manual:
        stats["kept"] = len(shapes)
        return list(shapes), stats

    dropped = set()

    # 1) 人工优先
    if keep_manual:
        for m in models:
            mb = m.get("bbox")
            if not mb:
                continue
            for h in manual:
                hb = h.get("bbox")
                if hb and h.get("label") == m.get("label") and _bbox_iou(mb, hb) >= KEEP_MANUAL_IOU:
                    dropped.add(id(m))
                    stats["by_manual"] += 1
                    break

    # 4) 形状离谱（先做，免得带进后面的配对）
    if use_shape:
        for m in models:
            if id(m) in dropped or m.get("label") != "Tracker":
                continue
            b = m.get("bbox")
            if not b:
                continue
            w, h = abs(b[2] - b[0]), abs(b[3] - b[1])
            if h > 0 and w / h > TRACKER_AR:
                dropped.add(id(m))
                stats["by_shape"] += 1

    # 2) 同编号
    if use_number:
        groups = {}
        for m in models:
            if id(m) in dropped or not _has_read_name(m):
                continue
            key = ((m.get("name") or "").strip().lower(), m.get("label"))
            groups.setdefault(key, []).append(m)
        for g in groups.values():
            if len(g) < 2:
                continue
            g.sort(key=lambda s: (-float(s.get("confidence") or 0.0),
                                  _bbox_area(s.get("bbox") or [0, 0, 0, 0])))
            for extra in g[1:]:
                dropped.add(id(extra))
                stats["by_number"] += 1

    # 3) 高重叠
    alive = [m for m in models if id(m) not in dropped and m.get("bbox")]
    for i in range(len(alive)):
        a = alive[i]
        if id(a) in dropped:
            continue
        for j in range(i + 1, len(alive)):
            b = alive[j]
            if id(b) in dropped or a.get("label") != b.get("label"):
                continue
            if _is_duplicate_pair(a["bbox"], b["bbox"]):
                # 留置信度高的（同一物体两次识别，置信度高的那次通常更靠谱），
                # 置信度一样再留面积小的（一般更贴）。
                key_a = (float(a.get("confidence") or 0.0), -_bbox_area(a["bbox"]))
                key_b = (float(b.get("confidence") or 0.0), -_bbox_area(b["bbox"]))
                if key_a >= key_b:
                    dropped.add(id(b))
                    stats["by_overlap"] += 1
                else:
                    dropped.add(id(a))
                    stats["by_overlap"] += 1
                    break

    # 5) 人工框之间的重复（默认不做，要调用方明确要求）
    if clean_manual:
        rest = [s for s in shapes if id(s) not in dropped and s.get("bbox")]
        for i in range(len(rest)):
            a = rest[i]
            if id(a) in dropped:
                continue
            for j in range(i + 1, len(rest)):
                b = rest[j]
                if id(b) in dropped or a.get("label") != b.get("label"):
                    continue
                if not _is_duplicate_pair(a["bbox"], b["bbox"]):
                    continue
                # 留哪个：框里真读到过编号 > 置信度高 > 面积大
                key_a = (bool(_has_read_name(a)), float(a.get("confidence") or 0.0),
                         _bbox_area(a["bbox"]))
                key_b = (bool(_has_read_name(b)), float(b.get("confidence") or 0.0),
                         _bbox_area(b["bbox"]))
                if key_a >= key_b:
                    dropped.add(id(b))
                    stats["manual_dup"] += 1
                else:
                    dropped.add(id(a))
                    stats["manual_dup"] += 1
                    break

    # 锁定的框一律保留：用户锁上就是明确说"别动它"，清理也不许删
    for s in shapes:
        if s.get("locked"):
            dropped.discard(id(s))

    out = [s for s in shapes if id(s) not in dropped]
    stats["removed"] = len(shapes) - len(out)
    stats["kept"] = len(out)
    stats["kept_manual"] = len(manual)
    return out, stats


def model_input_image(path, max_side=MODEL_INPUT_SIDE):
    """把渲染图按训练时的做法平滑缩到 max_side 长边；本来就够小就原样返回。"""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage
        im = QImage(path)
        if im.isNull():
            return path
        long_side = max(im.width(), im.height())
        if long_side <= max_side:
            return path
        sc = float(max_side) / float(long_side)
        small = im.scaled(max(1, round(im.width() * sc)), max(1, round(im.height() * sc)),
                          Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        out = os.path.join(tempfile.gettempdir(), "lbd_predict_input.png")
        return out if small.save(out, "PNG") else path
    except Exception:
        return path


def pdf_page_sizes(pdf):
    """{页号: (宽pt, 高pt, 旋转角)}：一次 pdfinfo 读完整册页尺寸。

    注意 pdfinfo 报的是**未旋转**的 mediabox，而 pdftoppm 渲染时会应用 /Rotate，
    所以这里把旋转角一并带出来，让调用方决定要不要把宽高对调。
    """
    exe = find_poppler("pdfinfo.exe")
    if not exe:
        return {}
    try:
        out = subprocess.run([exe, pdf], capture_output=True, text=True,
                             errors="replace", timeout=120,
                             creationflags=_NO_WINDOW).stdout
        m = re.search(r"^Pages:\s+(\d+)", out, re.M)
        if not m:
            return {}
        n = int(m.group(1))
        out = subprocess.run([exe, "-f", "1", "-l", str(n), pdf], capture_output=True,
                             text=True, errors="replace", timeout=600,
                             creationflags=_NO_WINDOW).stdout
    except Exception:
        return {}
    sizes = {}
    for m in re.finditer(r"^Page\s+(\d+)\s+size:\s+([\d.]+)\s+x\s+([\d.]+)\s+pts", out, re.M):
        sizes[int(m.group(1))] = [float(m.group(2)), float(m.group(3)), 0]
    for m in re.finditer(r"^Page\s+(\d+)\s+rot:\s+(-?\d+)", out, re.M):
        n = int(m.group(1))
        if n in sizes:
            sizes[n][2] = int(m.group(2)) % 360
    sizes = {n: tuple(v) for n, v in sizes.items()}
    return sizes


def rss_mb():
    """当前进程占用的物理内存（MB）：Windows 用 psapi，其它平台尽力而为。"""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        p = PMC()
        p.cb = ctypes.sizeof(PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(p), p.cb)
        return p.WorkingSetSize / 1048576.0
    except Exception:
        return 0.0


class BlankDoc:
    """没有识别结果、直接拿 PDF 标注时用的空文档：页数/尺寸来自 PDF，标注从零开始。

    坐标空间 = 按 self.dpi 渲染出来的像素（和底图渲染分辨率一致，框不会飘）。
    保存时生成一份和识别结果同格式的 JSON，可以直接喂给下游。
    """

    is_blank = True

    def __init__(self, pdf, dpi=250.0):
        self.path = pdf
        self.dpi = float(dpi)
        self.project = os.path.splitext(os.path.basename(pdf))[0]
        self.pages = {}
        for n, (w_pt, h_pt, rot) in sorted(pdf_page_sizes(pdf).items()):
            if rot in (90, 270):                 # 渲染时会旋转，宽高对调
                w_pt, h_pt = h_pt, w_pt
            self.pages[n] = {"page_number": n,
                             "width": max(1, int(round(w_pt * self.dpi / 72.0))),
                             "height": max(1, int(round(h_pt * self.dpi / 72.0))),
                             "png_span": None}

    def page_numbers(self):
        return sorted(self.pages)

    def page_data(self, key, page_number):
        return None

    def save(self, dest, modified):
        doc = {"schema_version": "agent3-debug-v1",
               "generator": "lbd_annotator（直接标注 PDF，无识别结果）",
               "source_pdf": os.path.basename(self.path),
               "render_dpi": self.dpi,
               "input_data": {"project_name": self.project,
                              "pages": [{"page_number": n, "media_type": "image/png",
                                         "width": self.pages[n]["width"],
                                         "height": self.pages[n]["height"]}
                                        for n in self.page_numbers()]},
               "yolo_tracker_detection_results": [],
               "yolo_box_detection_results": [],
               "ocr_node_name_results": []}
        for n in self.page_numbers():
            m = modified.get(n)
            if not m:
                # 没标注的页不写记录：下游是按"有东西的页"给底图排页号的，
                # 给每页都写一条空记录会让页号映射退化成 1:1，底图就对不上了。
                continue
            doc["yolo_tracker_detection_results"].append({"page_number": n, "data": m["tracker"]})
            doc["yolo_box_detection_results"].append({"page_number": n, "data": m["box"]})
            doc["ocr_node_name_results"].append({"page_number": n, "data": m["ocr"]})
        tmp = dest + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        os.replace(tmp, dest)
        return sorted(modified)


SUG_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def image_size_of(path):
    """只读文件头拿图片宽高（不整张解码，所以翻页/开文件夹不会卡）。"""
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return (int.from_bytes(head[16:20], "big"),
                        int.from_bytes(head[20:24], "big"))
    except Exception:
        pass
    try:                                    # 非 PNG：交给 Qt 解码（只在拿不到文件头时用）
        from PySide6.QtGui import QImage
        img = QImage(path)
        if not img.isNull():
            return img.width(), img.height()
    except Exception:
        pass
    return 0, 0


def label_to_ui(label):
    """补标 json 里的类别名 -> 标注工具认识的类别（Typical 就是 Tracker）。"""
    base = (label or "").strip()
    if base.endswith("?"):
        base = base[:-1].strip()
    low = base.lower()
    if low in ("tracker", "typical"):
        return "Tracker"
    if low == "node":
        return "Node"
    return "Box"


def _natkey(text):
    """自然排序用的键：p2 排在 p10 前面（不然文件名排序会出现 1,10,100,11…）。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(text))]


class SuggestFolderDoc:
    """把「补标文件夹」当一份文档来审：底图直接用配对的图片，保存就地写回同一个 json。

    json 是 X-AnyLabeling / labelme 格式（shapes + points），只含模型建议补的框，
    类别名带问号（Node? / Typical? / Tracker?）。问号是"模型建议、还没人工确认"的标记，
    合并回原始标注的脚本（yolo26/merge_suggest.py）靠它认出来，所以存回去时必须留着。
    """

    is_folder = True

    def __init__(self, folder):
        self.path = os.path.abspath(folder)
        self.project = os.path.basename(os.path.normpath(self.path))
        self.pages = {}
        self._cache = {}
        self._missing = []
        pairs = []
        for root, _dirs, files in os.walk(self.path):
            for name in sorted(files):
                if not name.lower().endswith(".json"):
                    continue
                stem = name[:-5]
                img = ""
                for cand in sorted(glob.glob(os.path.join(root, stem + ".*"))):
                    if os.path.splitext(cand)[1].lower() in SUG_IMG_EXTS:
                        img = cand
                        break
                if not img:
                    self._missing.append(os.path.join(root, name))
                    continue
                pairs.append((img, os.path.join(root, name)))
        pairs.sort(key=lambda t: (_natkey(os.path.dirname(t[0])),
                                  _natkey(os.path.basename(t[0]))))
        for i, (img, js) in enumerate(pairs, 1):
            w = h = 0
            try:
                doc = self._read(js)
                w = int(doc.get("imageWidth") or 0)
                h = int(doc.get("imageHeight") or 0)
            except Exception:
                pass
            if not (w and h):
                w, h = image_size_of(img)
            self.pages[i] = {"page_number": i, "width": w or 6000, "height": h or 4000,
                             "png_span": None, "png_path": img, "json_path": js,
                             "group": os.path.basename(os.path.dirname(img)),
                             "title": os.path.splitext(os.path.basename(js))[0]}
        if not self.pages:
            raise RuntimeError(
                "这个文件夹里没有「同名图片 + json」的成对文件。\n"
                "补标包的每个 json 旁边应该有同名的 png，先确认选对了文件夹。")

    # ---------------- 读
    def _read(self, js_path):
        doc = self._cache.get(js_path)
        if doc is None:
            with open(js_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            self._cache[js_path] = doc
            if len(self._cache) > 12:        # 只留最近翻过的几页，省内存
                for k in list(self._cache)[:-6]:
                    self._cache.pop(k, None)
        return doc

    def _doc_of(self, n):
        return self._read(self.pages[n]["json_path"])

    def page_numbers(self):
        return sorted(self.pages)

    def png_path(self, n):
        return self.pages[n]["png_path"]

    def title_of(self, n):
        return self.pages[n]["title"]

    def group_of(self, n):
        return self.pages[n]["group"]

    def missing_json(self):
        return list(self._missing)

    def _det_of(self, s):
        pts = s.get("points") or []
        xs, ys = [], []
        for p in pts:
            try:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
            except Exception:
                continue
        if not xs or not ys:
            return None
        raw_label = (s.get("label") or "").strip()
        ui = label_to_ui(raw_label)
        return {"label": ui, "name": "",
                # 注意：bbox 要用字典（PageModel 读的就是 x1/y1/x2/y2 这四个键）
                "bbox": {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)},
                "confidence": s.get("score"),
                "class_id": DEFAULT_CLASS_ID.get(ui, 0),
                "source": ("suggest" if (raw_label.endswith("?")
                                         or s.get("description") == "model_suggest")
                           else "context"),
                "raw": {"xl_label": raw_label, "xl_score": s.get("score"),
                        "xl_desc": s.get("description")},
                "ocr_index": None, "locked": False}

    def page_data(self, key, page_number):
        if page_number not in self.pages:
            return None
        if key == OCR_SECTION:
            return []
        want_box = (key == BOX_SECTION)
        dets = []
        try:
            shapes = self._doc_of(page_number).get("shapes") or []
        except Exception:
            shapes = []
        for s in shapes:
            d = self._det_of(s)
            if d is None:
                continue
            if want_box:
                if d["label"] == "Box":
                    dets.append(d)
            elif d["label"] in ("Node", "Tracker"):
                dets.append(d)
        return {"model_type": "box" if want_box else "tracker",
                "coordinates": "original_page_pixels",
                "detections": dets, "error": None}

    def _elements(self, key):
        """给 drawing_pages() 这类按页扫的函数用：补标包里每页都算「有内容」。"""
        return [{"page": n, "span": None, "data_span": None} for n in self.page_numbers()]

    def suggest_stats(self):
        """整包统计：(页数, 建议框数)。"""
        n_box = 0
        for n in self.page_numbers():
            try:
                n_box += len(self._doc_of(n).get("shapes") or [])
            except Exception:
                pass
        return len(self.pages), n_box

    # ---------------- 写
    def _shape_of(self, det):
        raw = det.get("raw") if isinstance(det.get("raw"), dict) else {}
        ui = det.get("label") or "Tracker"
        old = (raw.get("xl_label") or "").strip()
        if old and label_to_ui(old) == ui:
            label = old                       # 类别没改：原样保留（Node? / Typical? 都不动）
        else:
            label = ui + "?"                  # 改了类别或新画的：仍按「建议框」写，合并脚本才认
        b = det.get("bbox") or {}
        x1, y1 = float(b.get("x1") or 0), float(b.get("y1") or 0)
        x2, y2 = float(b.get("x2") or 0), float(b.get("y2") or 0)
        return {"label": label, "score": raw.get("xl_score"),
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "group_id": None, "description": "model_suggest",
                "difficult": False, "shape_type": "rectangle",
                "flags": {}, "attributes": {}, "kie_linking": []}

    def save(self, dest, modified):
        """就地写回：每一页写回它自己的 json。

        modified 的格式和 DebugJson.save 一致：{页号: {"tracker": …, "box": …, "ocr": …}}
        """
        written = []
        for n, mod in sorted(modified.items()):
            if n not in self.pages:
                continue
            js = self.pages[n]["json_path"]
            with open(js, "r", encoding="utf-8") as f:
                doc = json.load(f)
            shapes = [self._shape_of(d) for d in
                      ((mod.get("tracker") or {}).get("detections") or [])]
            shapes += [self._shape_of(d) for d in
                       ((mod.get("box") or {}).get("detections") or [])]
            doc["shapes"] = shapes
            doc["checked"] = False
            doc["imagePath"] = os.path.basename(self.pages[n]["png_path"])
            tmp = js + ".part"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            os.replace(tmp, js)
            self._cache.pop(js, None)
            written.append(n)
        return written


def project_name(json_path):
    """从 JSON 开头取 project_name（只读头 4KB，不整份解析）。"""
    try:
        with open(json_path, "rb") as f:
            head = f.read(4096).decode("utf-8", "ignore")
        m = re.search(r'"project_name"\s*:\s*"([^"]+)"', head)
        return m.group(1).strip() if m else ""
    except Exception:
        return ""


def source_pdf_name(json_path):
    """从 JSON 开头取 source_pdf（这份识别结果是哪份 PDF 做出来的）。

    比 project_name 靠谱：project_name 有可能是空的，而 source_pdf 就是文件名本身。
    """
    try:
        with open(json_path, "rb") as f:
            head = f.read(8192).decode("utf-8", "ignore")
    except Exception:
        return ""
    m = re.search(r'"source_pdf"\s*:\s*"([^"]+)"', head)
    return os.path.basename(m.group(1).strip()) if m else ""


def _pdf_search_dirs(json_path=""):
    """找 PDF 时看这些目录：JSON 旁边、下载、桌面（含 OneDrive 的"桌面"）、文档。"""
    home = os.path.expanduser("~")
    cands = []
    if json_path:
        cands.append(os.path.dirname(os.path.abspath(json_path)))
    cands += [os.path.join(home, "Downloads"), os.path.join(home, "Desktop"),
              os.path.join(home, "Documents")]
    for pat in ("OneDrive*/桌面", "OneDrive*/Desktop", "OneDrive*/桌面/*"):
        cands += glob.glob(os.path.join(home, pat))
    out, seen = [], set()
    for d in cands:
        if d and os.path.isdir(d) and d.lower() not in seen:
            seen.add(d.lower())
            out.append(d)
    return out


def find_pdf_by_name(name, json_path="", fuzzy=False, max_depth=3):
    """按文件名找 PDF。fuzzy=True 时再按"名字里像"的找一遍（例如只差日期/版本）。"""
    want = os.path.basename(str(name or "")).strip().lower()
    if not want:
        return ""
    stem = os.path.splitext(want)[0]
    for d in _pdf_search_dirs(json_path):
        base = d.rstrip(os.sep).count(os.sep)
        for root, subdirs, files in os.walk(d):
            if root.count(os.sep) - base > max_depth:
                subdirs[:] = []
                continue
            for f in files:
                if f.lower() == want:
                    return os.path.join(root, f)
    if not fuzzy or not stem:
        return ""
    # 差日期的版本（"2026.04.16 Bigway Solar - ... _Plans.pdf" vs "2026.6.2 Bigway Solar- E&S 90_.pdf"）：
    # 名字里的"词"（长度≥4）有一半以上重合就当像
    words = [w for w in re.findall(r"[a-z]{4,}", stem)]
    if not words:
        return ""
    for d in _pdf_search_dirs(json_path):
        base = d.rstrip(os.sep).count(os.sep)
        for root, subdirs, files in os.walk(d):
            if root.count(os.sep) - base > max_depth:
                subdirs[:] = []
                continue
            for f in sorted(files):
                fl = f.lower()
                if not fl.endswith(".pdf"):
                    continue
                hit = sum(1 for w in words if w in fl)
                if hit >= max(2, (len(words) + 1) // 2):
                    return os.path.join(root, f)
    return ""


def guess_pdf(json_path, exact_only=False):
    """按 JSON 里的 project_name 猜 PDF（同目录优先，其次 Downloads / 桌面）。
    exact_only=True 时只认文件名完全对得上的，不用兜底。"""
    name = project_name(json_path)
    home = os.path.expanduser("~")
    dirs = [os.path.dirname(os.path.abspath(json_path)),
            os.path.join(home, "Downloads"), os.path.join(home, "Desktop")]
    if name:
        want = (name + ".pdf").lower()
        for d in dirs:
            p = os.path.join(d, name + ".pdf")
            if os.path.exists(p):
                return p
        for d in dirs:
            if not os.path.isdir(d):
                continue
            base_depth = d.rstrip(os.sep).count(os.sep)
            for root, subdirs, files in os.walk(d):
                if root.count(os.sep) - base_depth > 3:
                    subdirs[:] = []
                    continue
                for f in files:
                    if f.lower() == want:
                        return os.path.join(root, f)
    if exact_only:
        return ""
    # 兜底**不再**"目录里随便拿一个 PDF" —— 那会拿别项目的图纸当底图
    #（用户反馈的"PDF 底图不对"就有这一条）。宁可没底图，也不要错的。
    return ""
    return ""


class Renderer:
    """用 poppler 按更高 DPI 重渲染某一页，结果缓存到磁盘。"""

    def __init__(self, exe, cache_root):
        self.exe = exe
        self.cache_root = cache_root
        self.sweep()

    def sweep(self):
        """清掉上次被中断留下的临时文件（渲染到一半被杀会残留）。"""
        try:
            for p in glob.glob(os.path.join(self.cache_root, "*", "_tmp_*.png")):
                if time.time() - os.path.getmtime(p) > 1800:
                    os.remove(p)
        except Exception:
            pass

    @staticmethod
    def pdf_tag(pdf):
        """把 PDF 的身份揉进缓存路径：换一份 PDF 必须换一套缓存，否则会拿到别人的图。"""
        try:
            st = os.stat(pdf)
            key = "%s|%d|%d" % (os.path.abspath(pdf).lower(), st.st_size, int(st.st_mtime))
        except OSError:
            key = os.path.abspath(pdf).lower()
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:8]

    def target(self, pdf, page, dpi):
        return os.path.join(self.cache_root, "%ddpi_%s" % (dpi, self.pdf_tag(pdf)),
                            "p%03d.png" % page)

    def render(self, pdf, page, dpi):
        out = self.target(pdf, page, dpi)
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            return out
        os.makedirs(os.path.dirname(out), exist_ok=True)
        # 每个线程用各自的临时前缀，避免预取和当前页撞车（os.replace 是原子的）
        tmp_prefix = os.path.join(os.path.dirname(out),
                                  "_tmp_p%03d_%d" % (page, threading.get_ident()))
        for old in glob.glob(tmp_prefix + "*.png"):
            try:
                os.remove(old)
            except OSError:
                pass
        subprocess.run([self.exe, "-png", "-r", str(dpi), "-f", str(page),
                        "-l", str(page), pdf, tmp_prefix],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=300, creationflags=_NO_WINDOW)
        made = sorted(glob.glob(tmp_prefix + "*.png"))
        if not made:
            raise RuntimeError("渲染没有产出文件")
        os.replace(made[0], out)
        for extra in made[1:]:
            try:
                os.remove(extra)
            except OSError:
                pass
        return out


def settings_path():
    """设置文件放 %LOCALAPPDATA%，不放在程序旁边。

    打包成文件夹版放到桌面后，程序旁边就是桌面 —— 设置和渲染缓存写在程序目录，
    桌面上就会冒出一堆文件和文件夹。老位置里的设置会自动搬过来。
    """
    return os.path.join(work_dir(), "annotator_settings.json")


def work_dir():
    """放设置 / 渲染缓存的地方（不动程序目录）。自检模式走临时目录，绝不碰用户的。"""
    if TEST_MODE:
        d = os.path.join(tempfile.gettempdir(), "LBD标注工具_selftest")
        try:
            os.makedirs(d, exist_ok=True)
            return d
        except Exception:               # noqa: BLE001
            pass
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        d = os.path.join(base, "LBD标注工具")
        try:
            os.makedirs(d, exist_ok=True)
            return d
        except Exception:
            pass
    return app_dir()


def cache_dir():
    return os.path.join(work_dir(), "render_cache")


def load_settings():
    old = os.path.join(app_dir(), "annotator_settings.json")
    try:
        with open(settings_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass
    # 老版本把设置放在程序旁边：读得到就搬过来（PDF/标签表路径不用重选）
    try:
        if os.path.exists(old):
            with open(old, "r", encoding="utf-8") as f:
                d = json.load(f)
            save_settings(d)
            return d
    except Exception:
        pass
    return {}


def save_settings(d):
    try:
        with open(settings_path(), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _sp_key(path):
    """起始页按"文件绝对路径"记（大小写不敏感），换机器/换目录互不影响。"""
    try:
        return os.path.abspath(str(path)).lower()
    except Exception:
        return str(path or "").lower()


def start_page_of(path):
    """这份 JSON 记住的"起始页"（打开时停在哪一页）；没记过返回 0。

    为什么需要它：有些册子前面十几页是封面/说明，图纸从第 16、17 页才开始；
    默认会停在"第一个有框的页"，但用户往往想直接从某一页开始看/标。
    """
    try:
        d = load_settings().get("start_pages") or {}
        return int(d.get(_sp_key(path)) or 0)
    except Exception:
        return 0


def set_start_page(path, page):
    """记下/清掉这份 JSON 的起始页（page<=0 = 清掉）。返回设置后的页码（0=已清除）。"""
    try:
        page = int(page or 0)
    except Exception:
        page = 0
    d = load_settings()
    sp = dict(d.get("start_pages") or {})
    k = _sp_key(path)
    if page > 0:
        sp[k] = page
    else:
        sp.pop(k, None)
    d["start_pages"] = sp
    save_settings(d)
    return page if page > 0 else 0


def json_start_page(path):
    """JSON 自己带的起始页（顶层字段 "lbd_start_page": N）；没有返回 0。

    比"记在本机设置里"强的地方：跟着文件走 —— 换机器、发给别人、拷到别的目录都还在。
    只读文件开头几 KB 就够（这个字段写在最外层）。
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(8192)
    except Exception:
        return 0
    m = re.search(r'"lbd_start_page"\s*:\s*(\d+)', head)
    try:
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


def write_json_start_page(path, page):
    """把起始页写进 JSON 本身（顶层字段 lbd_start_page）；page<=0 = 删掉这个字段。

    只动这一个字段，其余内容逐字节照搬（和标注工具"保存某几页"那套做法一致）。
    返回写进去的页码（0 = 已删除）。
    """
    try:
        page = int(page or 0)
    except Exception:
        page = 0
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    txt = re.sub(r'"lbd_start_page"\s*:\s*\d+\s*,\s*', "", txt, count=1)
    txt = re.sub(r',?\s*"lbd_start_page"\s*:\s*\d+', "", txt, count=1)
    if page > 0:
        i = txt.find("{")
        if i < 0:
            return 0
        txt = txt[:i + 1] + '\n  "lbd_start_page": %d,' % page + txt[i + 1:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return page if page > 0 else 0


# ------------------------------------------------------- PDF 文字层：框内找 LBD 标签
# 这一套和主程序 app.py 里的 _pdf_text_items 等价，搬到标注工具里是为了
# "从已经框好的 LBD 区域里取标签文字"，不依赖任何识别模型。
def _pdf_rotate(page):
    try:
        return int(page.get("/Rotate") or 0) % 360
    except Exception:
        return 0


def _pdf_norm_pt(page, x, y):
    """PDF 用户坐标 -> 底图（渲染图）归一化坐标 (fx, fy)，fy 从下往上。

    PDF 带 /Rotate 时文字层坐标是没转过的，而底图是按转过的样子渲染的，
    不换算整片文字会跑到错位置。
    """
    try:
        cb = page.cropbox
        x0, y0 = float(cb.left), float(cb.bottom)
        pw = float(cb.right) - x0
        ph = float(cb.top) - y0
    except Exception:
        x0, y0, pw, ph = 0.0, 0.0, 1.0, 1.0
    pw = pw or 1.0
    ph = ph or 1.0
    x = float(x) - x0
    y = float(y) - y0
    rot = _pdf_rotate(page)
    if rot == 90:
        return (y / ph, (pw - x) / pw)
    if rot == 180:
        return ((pw - x) / pw, (ph - y) / ph)
    if rot == 270:
        return ((ph - y) / ph, x / pw)
    return (x / pw, y / ph)


# Helvetica / Arial 标准字宽表（单位 1/1000 em，来自 PDF 规范里那套内置字体度量）。
# CAD 图纸上的文字基本都是 Arial/Helvetica 系的，有这张表就能算出**真实字宽**，
# 不用再按"0.5×字号×字数"估 —— 实测那套估算会把编号框压短 ~10%。
_HELV_W = {}
for _ch, _w in ((" ", 278), ("!", 278), ('"', 355), ("#", 556), ("$", 556),
                ("%", 889), ("&", 667), ("'", 191), ("(", 333), (")", 333),
                ("*", 389), ("+", 584), (",", 278), ("-", 333), (".", 278),
                ("/", 278), (":", 278), (";", 278), ("<", 584), ("=", 584),
                (">", 584), ("?", 556), ("@", 1015), ("[", 278), ("\\", 278),
                ("]", 278), ("^", 469), ("_", 556), ("`", 333), ("{", 334),
                ("|", 260), ("}", 334), ("~", 584)):
    _HELV_W[_ch] = _w
for _d in "0123456789":
    _HELV_W[_d] = 556
for _c, _w in (("A", 667), ("B", 667), ("C", 722), ("D", 722), ("E", 667),
               ("F", 611), ("G", 778), ("H", 722), ("I", 278), ("J", 500),
               ("K", 667), ("L", 556), ("M", 833), ("N", 722), ("O", 778),
               ("P", 667), ("Q", 778), ("R", 722), ("S", 667), ("T", 611),
               ("U", 722), ("V", 667), ("W", 944), ("X", 667), ("Y", 667),
               ("Z", 611), ("a", 556), ("b", 556), ("c", 500), ("d", 556),
               ("e", 556), ("f", 278), ("g", 556), ("h", 556), ("i", 222),
               ("j", 222), ("k", 500), ("l", 222), ("m", 833), ("n", 556),
               ("o", 556), ("p", 556), ("q", 556), ("r", 333), ("s", 500),
               ("t", 278), ("u", 556), ("v", 500), ("w", 722), ("x", 500),
               ("y", 500), ("z", 500)):
    _HELV_W[_c] = _w


def text_advance(text, size, font_name=""):
    """这段文字在**文本空间**里有多长（还没乘矩阵）。返回 (宽度, 是不是按真字宽算的)。

    字体是 Arial / Helvetica / Liberation / Nimbus 这类标准字时，用标准字宽表算真值；
    其它字体（或表里没有的字符）退回"0.5×字号×字数"的估算。CAD 图纸基本都是前一类。
    """
    try:
        cs = float(size)
    except Exception:
        cs = 0.0
    nm = re.sub(r"^[A-Z]{6}\+", "", str(font_name or "")).lower()
    known = ("arial" in nm or "helvetica" in nm or "liberation" in nm
             or "nimbus" in nm or "helv" in nm)
    if known and cs > 0 and text:
        total = 0.0
        for ch in text:
            u = _HELV_W.get(ch)
            if u is None:
                return 0.5 * cs * len(text), False     # 有表里没有的字 -> 退回估算
            total += u
        return total / 1000.0 * cs, True
    return 0.5 * cs * len(text), False


def _pdf_text_items(page):
    """一页 -> [(文字, fx中心, fy中心, fx1, fy1, fx2, fy2)]（底图归一化坐标）。"""
    items = []

    def visit_text(text, cm, tm, font, size):
        if not text or not text.strip():
            return
        try:
            m0 = cm[0] * tm[0] + cm[2] * tm[1]
            m1 = cm[1] * tm[0] + cm[3] * tm[1]
            m2 = cm[0] * tm[2] + cm[2] * tm[3]
            m3 = cm[1] * tm[2] + cm[3] * tm[3]
            m4 = cm[0] * tm[4] + cm[2] * tm[5] + cm[4]
            m5 = cm[1] * tm[4] + cm[3] * tm[5] + cm[5]
        except Exception:
            m0, m1, m2, m3, m4, m5 = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
        try:
            cs = float(size)
        except Exception:
            cs = 0.0
        try:
            fname = str(font.get("/BaseFont") or "") if font else ""
        except Exception:
            fname = ""
        w, exact = text_advance(text, cs, fname)
        h = cs
        xs, ys = [], []
        for tx, ty in ((0.0, 0.0), (w, 0.0), (0.0, h), (w, h)):
            xs.append(m0 * tx + m2 * ty + m4)
            ys.append(m1 * tx + m3 * ty + m5)
        pts = [_pdf_norm_pt(page, px, py)
               for px, py in ((min(xs), min(ys)), (max(xs), min(ys)),
                              (min(xs), max(ys)), (max(xs), max(ys)))]
        fx1, fx2 = min(p[0] for p in pts), max(p[0] for p in pts)
        fy1, fy2 = min(p[1] for p in pts), max(p[1] for p in pts)
        items.append((text.strip(), (fx1 + fx2) * 0.5, (fy1 + fy2) * 0.5,
                      fx1, fy1, fx2, fy2, exact))

    page.extract_text(visitor_text=visit_text)
    return items


class PdfText:
    """按页取 PDF 文字块（复用同一个 reader + 缓存：整册跑才不至于太慢）。"""

    def __init__(self, pdf):
        from pypdf import PdfReader
        self.pdf = pdf
        self.reader = PdfReader(pdf)
        self._cache = {}

    def count(self):
        return len(self.reader.pages)

    def items(self, page_number):
        if page_number not in self._cache:
            try:
                self._cache[page_number] = _pdf_text_items(
                    self.reader.pages[int(page_number) - 1])
            except Exception:
                self._cache[page_number] = []
        return self._cache[page_number]


# ------------------------------------------------------- LBD 标签表（xlsx）
_XL_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_XL_RN = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _xlsx_sheets(path, want_cols=("A", "C")):
    """xlsx -> 有序的 [(分表名, {行号: {列: 文字}})]。不用 Excel，直接解 zip。"""
    import zipfile
    import xml.etree.ElementTree as ET
    out = []
    with zipfile.ZipFile(path) as z:
        have = set(z.namelist())

        def _read(name):
            return z.read(name) if name in have else None

        wb = ET.fromstring(_read("xl/workbook.xml"))
        rels = ET.fromstring(_read("xl/_rels/workbook.xml.rels"))
        rid2t = {r.get("Id"): r.get("Target") for r in rels}
        shared = []
        sst = _read("xl/sharedStrings.xml")
        if sst is not None:
            for si in ET.fromstring(sst):
                shared.append("".join(t.text or "" for t in si.iter(_XL_NS + "t")))
        for el in wb.iter():
            if not el.tag.endswith("}sheet"):
                continue
            sheet = (el.get("name") or "").strip()
            tgt = (rid2t.get(el.get(_XL_RN + "id")) or "").lstrip("/")
            if not sheet or not tgt:
                continue
            if not tgt.startswith("xl/"):
                tgt = "xl/" + tgt
            data = _read(tgt)
            if data is None:
                continue
            cells = {}
            for row in ET.fromstring(data).iter(_XL_NS + "row"):
                rn = int(row.get("r") or 0)
                for c in row:
                    if c.tag != _XL_NS + "c":
                        continue
                    ref = c.get("r") or ""
                    m = re.match(r"[A-Z]+", ref)
                    col = m.group(0) if m else ""
                    if col not in want_cols:
                        continue
                    typ = c.get("t")
                    v = c.find(_XL_NS + "v")
                    val = ""
                    if typ == "s" and v is not None:
                        try:
                            val = shared[int(v.text)]
                        except Exception:
                            val = ""
                    elif typ == "inlineStr":
                        ins = c.find(_XL_NS + "is")
                        val = ("".join(x.text or "" for x in ins.iter(_XL_NS + "t"))
                               if ins is not None else "")
                    else:
                        val = v.text if v is not None else ""
                    cells.setdefault(rn, {})[col] = (val or "").strip()
            out.append((sheet, cells))
    return out


def xlsx_sheet_names(path):
    """标签表的分表名（按表顺序）。"""
    try:
        return [nm for nm, _c in _xlsx_sheets(path)]
    except Exception:
        return []


def xlsx_lbd_rows(path, sheet_name):
    """一个分表的 LBD 行（按表里的行顺序）-> [(编号, 名称串)]。

    编号规则和 CAD 插件对齐：A 列写了 "LBD-15" 就用 15；A 列没有纯编号
    （比如 Yellow Viking 的 "1.C.1"）就按"这个分表里第几个 LBD 行"算 1,2,3…
    —— 两边同一套键，标签才对得上。
    """
    cells_all = None
    for name, cells in _xlsx_sheets(path):
        if name.strip().upper() == str(sheet_name).strip().upper():
            cells_all = cells
            break
    if not cells_all:
        return []
    order = sorted(cells_all)
    # 表头（C 列 "Item Code" / A 列 "LBD NO."）之后才开始算 LBD 行，
    # 不然像 "XXX Plus 1.01" 这种标题行会被当成第一行
    start = 0
    for k, rn in enumerate(order):
        a = cells_all[rn].get("A", "")
        c = cells_all[rn].get("C", "")
        if re.search(r"item\s*code", c, re.I) or re.match(r"^LBD\s*NO", a, re.I):
            start = k + 1
            break
    rows = []
    for rn in order[start:]:
        a = cells_all[rn].get("A", "")
        c = cells_all[rn].get("C", "")
        if a:
            rows.append([a, []])
        if c and rows:
            rows[-1][1].append(c)
    numbered = [_lbd_num_in(a) for a, _l in rows]
    # 只要有一行 A 列写了 "LBD-15" 这种真编号，就按真编号走；
    # 否则按"这个分表里的第几行"编号（"1.C.1" 这种）
    use_num = any(n is not None for n in numbered)
    out = []
    k = 0
    for (a, labels), n in zip(rows, numbered):
        if not labels:
            continue
        k += 1
        num = n if (use_num and n is not None) else k
        out.append((num, "/".join(labels)))
    return out


# ------------------------------------------------------- 框内取标签 / 编号
LBD_NUM_RE = re.compile(r"LBD\s*[_\-–—]?\s*0*(\d+)", re.I)
# 有些图里框内只印一个裸编号（"07" / "#7"），文字层里并没有 "LBD" 三个字母。
# 只认"整条就是数字"的，别把 3.5、1:100 这种尺寸/比例当编号。
BARE_NUM_RE = re.compile(r"^#?\s*0*(\d{1,3})\.?$")
# 兜底：框里任何文字里的第一个数字（"1.C.1"、"LBD-07"、"07" 都行）
ANY_NUM_RE = re.compile(r"(\d{1,3})")
INV_RE = re.compile(r"INV\s*0*(\d+)\s*([A-Za-z])\s*0*(\d+)", re.I)
SHEET_CLEAN_RE = re.compile(r"[^0-9A-Za-z.]")


def _clean_key(s):
    return SHEET_CLEAN_RE.sub("", str(s or "")).upper()


def _lbd_num_in(text):
    m = LBD_NUM_RE.search(str(text or ""))
    return int(m.group(1)) if m else None


def _clean_lbd_label(txt):
    """框里那串编号：去掉分表名和 LBD 标记，**保留前面的序号段**。

    '1.01.1.C.5'        -> '1.01.1.C.5'
    'INV31B102_LBD_07'  -> '07'
    'LBD 1.01.1.C.5'    -> '1.01.1.C.5'
    """
    t = re.sub(r"\s+", "", str(txt or ""))
    t = re.sub(r"INV\s*\d+\s*[A-Za-z]\s*\d+", "", t, flags=re.I)
    m = re.search(r"LBD[\s_\-–—:]*([0-9][0-9A-Za-z._\-]*)", t, re.I)
    if m:
        t = m.group(1)
    t = re.split(r"[()\[\]{}]", t)[0]          # 后面跟的 "(2)" 这种注解不要
    return t.strip("-_ .|()[]#")


LBD_LABEL_RE = re.compile(r"INV\s*\d+\s*[A-Z]\s*\d+\s*[-_ ]?LBD[-_ ]?\s*\d+", re.I)
LBDISH_RE = re.compile(r"LBD|INV\s*\d+\s*[A-Z]\s*\d+", re.I)


def ocr_text_to_name(txt):
    """OCR 读到的一行字 -> 要不要写成名字。返回 (名字, 状态)。

    状态：'ok'    框里就是标准编号（INV..-LBD-..）
          'check' 像编号但不是标准写法（少了前缀、被截断…）-> 写成名字，标紫让人核
          'noise' 读出来的是别的字（旁注、比例尺、图号…）-> **不写名字**，标红等人填
          'miss'  什么都没读到

    为什么要分 noise：OCR 读错时最容易吐出来的就是框边上那些旁注（"LBD CLUSTER, TYP."、
    "MV1A"、"TCH, TYP."）。以前一律写进名字，用户看到的就是"识别到的和里面的内容完全
    没有关系"。
    """
    t = str(txt or "").strip()
    if not t:
        return "", "miss"
    up = t.upper().replace(" ", "")
    if LBD_LABEL_RE.search(up):
        # 名字用原文（保留 1.01.1.C.5 这种整段），只砍掉尾巴上的 "(2)" 之类注解
        return re.split(r"[()\[\]{}]", t)[0].strip().replace(" ", "").upper(), "ok"
    if LBDISH_RE.search(up):
        return t[:40], "check"
    return "", "noise"


def _full_lbd_name(txt):
    """文字里带整段编号（INV01A01-LBD-01）就把它原样当名字用；没有返回 None。

    文字层里这条文字本身就是"分表+编号"的完整写法，比事后拿分表名拼一遍更靠得住。
    """
    nm, kind = ocr_text_to_name(txt)
    return nm if kind == "ok" else None


# --------------------------------------------------- 每页比例尺（1 IN = ? FT）
SCALE_ONE_RE = re.compile(
    r"1\s*(?:\"|''|IN\b|INCH(?:ES)?)\s*[=:]\s*(\d+(?:\.\d+)?)\s*(?:FT\b|FEET|')", re.I)
SCALE_FT_PER_IN_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:FT\b|FEET|')\s*(?:PER|/|每)\s*(?:\"|IN\b|INCH)", re.I)
SHEET_SIZE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*"\s*[xX×]\s*(\d+(?:\.\d+)?)\s*"')


def read_page_scale(items):
    """从一页的文字层里读出「1 IN = ? FT」。返回 (每英寸英尺数, 说明)。

    读不到返回 (None, 原因)。图纸上比例尺就两种写法，都认：
      ① 一句话：SCALE: 1" = 100'-0" / 1 IN = 100 FT
      ② 标尺表格（Steel River 这种）：标题栏里一行 "1  2  IN"、下一行 "100  200  FT"，
         数字上下对齐 —— 按 x 对齐把 1↔100、2↔200 配成对，用它们的比值（防止只看一行取错）。
    """
    txt = [((it[0] or "").strip(), float(it[1]), float(it[2])) for it in items]
    txt = [t for t in txt if t[0]]
    for s, _x, _y in txt:                       # ① 一句话
        m = SCALE_ONE_RE.search(s) or SCALE_FT_PER_IN_RE.search(s)
        if m:
            v = float(m.group(1))
            if 0 < v < 100000:
                return v, "文字里写着 1 IN = %g FT" % v
    # ② 标尺表格
    inch_lbl = [t for t in txt if re.fullmatch(r"IN|INCH(?:ES)?|\"", t[0], re.I)]
    ft_lbl = [t for t in txt if re.fullmatch(r"FT|FEET|'", t[0], re.I)]
    ratios = []
    for _s, ix, iy in inch_lbl:
        for _s2, fx, fy in ft_lbl:
            if abs(fy - iy) > 0.06 or fx < ix:
                continue                        # FT 得在同一块表里偏下/偏右
            nums_i = [t for t in txt
                      if abs(t[2] - iy) <= 0.012 and t[1] < ix + 0.02
                      and re.fullmatch(r"\d+(?:\.\d+)?", t[0])]
            nums_f = [t for t in txt
                      if abs(t[2] - fy) <= 0.012 and t[1] < fx + 0.02
                      and re.fullmatch(r"\d+(?:\.\d+)?", t[0])]
            for s3, x3, _y3 in nums_i:
                v_i = float(s3)
                if v_i <= 0:
                    continue
                near = min(nums_f, key=lambda t: abs(t[1] - x3), default=None)
                if near is None or abs(near[1] - x3) > 0.02:
                    continue
                v_f = float(near[0])
                if v_f > v_i:
                    ratios.append(v_f / v_i)
    if ratios:
        ratios.sort()
        v = ratios[len(ratios) // 2]
        return v, "标题栏标尺表（%d 对数字）" % len(ratios)
    return None, "这页文字层里没找到比例尺"


def read_sheet_width_in(items):
    """从"22\" x 34\" SHEETS"这种说明里读出图纸幅面（取大的一边，单位英寸）。"""
    for it in items:
        m = SHEET_SIZE_RE.search(str(it[0] or ""))
        if m:
            try:
                a, b = float(m.group(1)), float(m.group(2))
            except ValueError:
                continue
            if a > 1 and b > 1:
                return max(a, b)
    return None


def _name_num(name):
    """名字里的"号"：取最后一段数字。'…-LBD-1.01.1.C.5' -> 5，'…-LBD-07' -> 7。"""
    segs = re.findall(r"\d+", str(name or ""))
    return int(segs[-1]) if segs else None


def infer_lbd_names(shapes, max_gap=3, only_label=None):
    """给"没读到编号"的框按「位置顺序 + 编号连续」推编号，推出来的标黄（_auto）。

    只推能自证的情况，宁可少推也不乱推：
      ① 同一分表里已读到的编号，沿"位置顺序"（一排排、行内从左到右）单调递增或递减；
      ② 两个已知编号之间缺几个，正好等于这一段里没读到号的框的个数；
      ③ 缺的个数 ≤ max_gap（默认 3），且两头的编号本身是相邻段里的（不跨大空档）。
    推出来的名字沿用已读到的那种写法（INV10A06-LBD-07），前缀取自已读到的多数派。
    返回推出来几个。
    """
    rows = []
    for i, s in enumerate(shapes):
        if s.get("label") not in ("Node", "Tracker"):
            continue
        if only_label and s.get("label") != only_label:
            continue
        m = re.search(r"([A-Z]{2,}\d+[A-Z]?\d*)[-_ ]*LBD[-_ ]*0*(\d{1,3})",
                      (s.get("name") or ""), re.I)
        cx, cy = _box_center(s["bbox"])
        rows.append({"i": i, "s": s, "key": m.group(1).upper() if m else "",
                     "num": int(m.group(2)) if m else None,
                     "cy": cy, "cx": cx, "known": bool(m)})
    known = [r for r in rows if r["known"]]
    todo = [r for r in rows if not r["known"] and not (r["s"].get("name") or "").strip()]
    if len(known) < 2 or not todo:
        return 0
    cnt = {}
    for r in known:
        cnt[r["key"]] = cnt.get(r["key"], 0) + 1
    key = max(cnt, key=lambda k: cnt[k])
    known = [r for r in known if r["key"] == key]
    if len(known) < 2 or not todo:
        return 0
    seq = sorted(known + todo, key=lambda r: (round(r["cy"], 1), round(r["cx"], 1)))
    knums = [r["num"] for r in known]
    if len(set(knums)) != len(knums):
        return 0                          # 有重号，说明读得不准，别推
    if all(knums[i] < knums[i + 1] for i in range(len(knums) - 1)):
        inc = True
    elif all(knums[i] > knums[i + 1] for i in range(len(knums) - 1)):
        inc = False
    else:
        return 0                          # 位置顺序和编号顺序对不上，别推
    got = 0
    for a in range(len(seq)):
        if seq[a]["known"]:
            continue
        # 往左、往右各找一个已知编号，中间全是没读到的
        l = a - 1
        while l >= 0 and not seq[l]["known"]:
            l -= 1
        r = a + 1
        while r < len(seq) and not seq[r]["known"]:
            r += 1
        if l < 0 or r >= len(seq):
            continue                      # 缺在开头/结尾：两头没有参照，不推
        num_l, num_r = seq[l]["num"], seq[r]["num"]
        span = abs(num_r - num_l) - 1
        misses = r - l - 1
        if span <= 0 or span > max_gap or span != misses:
            continue                      # 缺的个数对不上（可能真跳号/漏图），不推
        step = 1 if num_r > num_l else -1
        seq[a]["s"]["name"] = _lbd_name(key, "", num_l + step * (a - l))
        seq[a]["s"]["_auto"] = True
        seq[a]["s"]["_miss"] = False
        seq[a]["s"]["_check"] = False
        got += 1
    return got


def _lbd_name(sheet, label, num):
    """拼 LBD 名字：<分表名>-LBD-<编号>。

    整段编号原样用（1.01.1.C.5）；只有一段数字就补成两位（07）。
    """
    lab = str(label or "").strip("-_ .|()[]#")
    if not lab and num is not None:
        lab = "%02d" % int(num)
    elif re.fullmatch(r"0*\d{1,3}", lab):
        lab = "%02d" % int(lab)
    return ("%s-LBD-%s" % (sheet, lab)) if sheet else ("LBD-%s" % lab)


def _inv_in(text):
    """文字里自带的 INV 分表名（如 INV31B102_LBD_07 -> INV31B102），没有返回 None。"""
    m = INV_RE.search(str(text or ""))
    if not m:
        return None
    return ("INV%s%s%s" % (m.group(1), m.group(2).upper(), m.group(3))).upper()


def norm_sheet(s):
    """分表名归一化：INV31B102 / inv31b102 / INV31B102_LBD_07 都能对上。"""
    return _clean_key(s)


def sheet_in_text(text, sheet_names):
    """文字里命中的分表名（取最长的那个，避免 "1.0" 抢 "1.01"）。"""
    key = _clean_key(text)
    if not key:
        return None
    best = None
    for nm in sheet_names:
        k = _clean_key(nm)
        if k and k in key and (best is None or len(k) > len(_clean_key(best))):
            best = nm
    return best


def page_sheet_by_text(text_items, sheet_names):
    """页面文字里出现次数最多的分表名 + 次数（用来校验"按顺序"推断的对不对）。"""
    counts = {}
    for it in text_items:
        nm = sheet_in_text(it[0], sheet_names)
        if nm:
            counts[nm] = counts.get(nm, 0) + 1
    if not counts:
        return None, 0
    nm = max(counts, key=lambda k: counts[k])
    return nm, counts[nm]


def _box_dist(b, x, y):
    dx = max(b[0] - x, 0.0, x - b[2])
    dy = max(b[1] - y, 0.0, y - b[3])
    return (dx * dx + dy * dy) ** 0.5


def _box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _flag_suspicious(shapes):
    """同一行里相邻两个框的号不连续（比如 3、7）-> 标 _check，返回清单。

    自动补号最常见的错法就是"这一排的号跳了"，拿它当"位置可能错"的提示。
    返回 [(名字, x, y), ...]，坐标是底图像素（人照着这个去图上找）。
    """
    nodes = []
    for s in shapes:
        if s.get("label") != "Node":
            continue
        # 名字里的"号"= 最后一段数字：既能认 "…-LBD-07"（7），
        # 也能认 "…-LBD-1.01.1.C.5"（5）—— 跳号检查就靠它
        n = _name_num(s.get("name") or "")
        if n is None:
            continue
        cx, cy = _box_center(s["bbox"])
        nodes.append((cy, cx, n, s))
    if len(nodes) < 2:
        return []
    rows = _rows_of(nodes, _row_tol(shapes))
    out = []
    for row in rows:
        for k in range(len(row) - 1):
            if abs(row[k][2] - row[k + 1][2]) != 1:
                for it in (row[k], row[k + 1]):
                    if not it[3].get("_check"):
                        it[3]["_check"] = True
                        out.append((it[3].get("name"), round(it[1]), round(it[0])))
    return out


def _row_tol(shapes):
    """分行容差：按 Node 框高度的中位数取 60%（条带框长、中心点浮动大，容差得放宽）。"""
    hs = sorted(abs(s["bbox"][3] - s["bbox"][1])
                for s in shapes if s.get("label") == "Node")
    med = hs[len(hs) // 2] if hs else 1.0
    return max(2.0, med * 0.6)


def _rows_of(items, tol):
    """[(y中点, x中点, ...)] -> 按 y 分成一排排（行内从左到右排好）。

    注意不能直接按 (y, x) 排序：同一条带排里的框高度不一样，y 中点能差好几百像素，
    直接排会把同一排的号打乱（补号顺序就错了）。
    """
    ps = sorted(items, key=lambda t: t[0])
    rows, cur = [], []
    for it in ps:
        if not cur or abs(it[0] - cur[-1][0]) <= tol:
            cur.append(it)
        else:
            rows.append(cur)
            cur = [it]
    if cur:
        rows.append(cur)
    for r in rows:
        r.sort(key=lambda t: t[1])
    return rows


def drawing_pages(dbg):
    """「有框的页」按页码升序 —— 和主程序 debug_page_map 同一套规则。

    分表顺序按这个列表排（不是 PDF 页码）：第 1 张图纸对标签表第 1 个分表。
    封面/说明页、以及"整本都写了空记录"的空页都不算，否则一律错位一格。
    """
    got = set()
    for key in SECTION_ORDER:
        for el in dbg._elements(key):
            try:
                data = dbg.page_data(key, el["page"])
            except Exception:
                continue
            if isinstance(data, dict):
                if data.get("detections"):
                    got.add(el["page"])
            elif isinstance(data, list) and data:
                got.add(el["page"])
    return sorted(got)


def _page_candidates(text_items, sheet, num_set, width, height):
    """这一页可用的编号文字 -> (强候选, 裸数字候选, 分表名对不上的条数)。

    强候选：文字里有 LBD+数字（如 INV31B102_LBD_07），且号码在本分表号码表里，
            再把"图例/引线名单"（同列密排）滤掉。
    裸数字候选：整条文字就是个号（"07"），有些图框里只印这个，没有 "LBD" 字样 —
            这条通道以前没有，所以"每个框里都有号、有的却取不到"。
    """
    raw, full, bare, wrong = [], [], [], 0
    for it in text_items:
        num = _lbd_num_in(it[0])
        inv = _inv_in(it[0])
        if inv and _clean_key(sheet) and _clean_key(inv) != _clean_key(sheet):
            wrong += 1
            continue
        px = it[1] * width
        py = (1.0 - it[2]) * height
        # 整段编号（INV01A01-LBD-01）单独归一类：这种文字本身就把分表和号写全了，
        # 绝不能当"图例/引线名单"丢掉 —— 用户那本 Steel River 上，全页 20 个编号
        # 就是被 _drop_legend 当图例清掉，然后兜底抓了旁边的 "MV1A" 当编号。
        if _full_lbd_name(it[0]):
            full.append((px, py, num, it[0]))
            continue
        if num is not None:
            if sheet and num_set and num not in num_set:
                continue
            raw.append((px, py, num, it[0]))
        else:
            m = BARE_NUM_RE.match(it[0].strip())
            if m:
                bare.append((px, py, int(m.group(1)), it[0]))
    return full + _drop_legend(raw, height), bare, wrong


def _box_texts(shapes, text_items, width, height, tol_ratio=0.006):
    """框里 / 框边上找到的**所有文字**（先全拿出来，谁是什么号后面再判断）。

    返回 {框下标: [(距离, 文字, x, y)]}；每条文字只归给离它最近的那个框。
    x/y 是这条文字的中心（页面像素），补编号时拿它当"编号印在哪儿"。
    """
    nodes = [(i, s) for i, s in enumerate(shapes) if s.get("label") == "Node"]
    tol = max(6.0, float(tol_ratio) * max(width, height))
    out = {}
    for it in text_items:
        txt = (it[0] or "").strip()
        if not txt:
            continue
        px, py = it[1] * width, (1.0 - it[2]) * height
        best, bd = None, None
        for i, s in nodes:
            d = _box_dist(s["bbox"], px, py)
            if bd is None or d < bd:
                best, bd = i, d
        if best is not None and bd is not None and bd <= tol:
            out.setdefault(best, []).append((round(bd), txt, px, py))
    for k in out:
        out[k].sort()
    return out


def _drop_legend(cands, page_h):
    """滤掉"引线名单/图例"：同一列上密集挤着 ≥3 条编号的那种。

    图纸上每个区域自己的号是孤零零一条；名单是十几条一列排下来 —— 靠这个区分。
    """
    if len(cands) < 4:
        return list(cands)
    gx = max(4.0, page_h * 0.004)
    gy = max(8.0, page_h * 0.02)
    out = []
    for i, (px, py, num, txt) in enumerate(cands):
        n = 0
        for j, (qx, qy, _n, _t) in enumerate(cands):
            if i != j and abs(qx - px) <= gx and abs(qy - py) <= gy:
                n += 1
        if n < 3:
            out.append((px, py, num, txt))
    return out


def check_rows(shapes, text_items, sheet, num_set, width, height):
    """核对表：每个 Node 框一行 —— 现有名字 + 框内找到的候选 + 按现在规则重算的建议。

    返回 (rows, dry_stat)。rows 里每个 dict：
      ix 框序号 / cx,cy 框中心（底图像素）/ now 现有名字 / sug 建议名字
      cand 框内候选（"号@距离"）/ src 建议来源（框内 / 推 / 缺）
    """
    cands, bare, _wrong = _page_candidates(text_items, sheet, num_set, width, height)
    alltext = _box_texts(shapes, text_items, width, height)
    tol = max(6.0, 0.006 * max(width, height))
    nodes = [(i, s) for i, s in enumerate(shapes) if s.get("label") == "Node"]
    got = {}
    for px, py, num, txt in cands:
        best, bd = None, None
        for i, s in nodes:
            d = _box_dist(s["bbox"], px, py)
            if bd is None or d < bd:
                best, bd = i, d
        if best is not None and bd is not None and bd <= tol:
            got.setdefault(best, []).append((bd, num, txt))
    # 只印裸编号的（文字层里没有 "LBD" 字样）也一起列出来，标个 (裸)
    for px, py, val, txt in bare:
        best, bd = None, None
        for i, s in nodes:
            d = _box_dist(s["bbox"], px, py)
            if bd is None or d < bd:
                best, bd = i, d
        if best is not None and bd is not None and bd <= tol:
            got.setdefault(best, []).append((bd, val, txt + "(裸)"))
    tmp = [dict(s) for s in shapes]
    for s in tmp:
        s.pop("_auto", None)
        s.pop("_miss", None)
        s.pop("_check", None)
        if s.get("label") == "Node":
            s["name"] = ""
    dry = autofill_shapes(tmp, text_items, sheet, num_set, width, height)
    rows = []
    for i, s in nodes:
        cx, cy = _box_center(s["bbox"])
        cand = sorted(got.get(i) or [])[:3]
        raw = []
        for d, txt, _tx, _ty in (alltext.get(i) or [])[:6]:
            mark = ""
            # 标 × 的：这条文字被过滤掉了（号不在本分表号码表里 / 结尾像是被截断）
            mm = ANY_NUM_RE.search(re.sub(r"INV\s*\d+\s*[A-Za-z]\s*\d+", " ", txt, flags=re.I))
            if (txt.rstrip().endswith(("-", "_", ".", "(", "#"))
                    or (mm and sheet and num_set and int(mm.group(1)) not in num_set)):
                mark = "×"
            raw.append((d, txt + mark))
        t = tmp[i]
        src = ("框内" if raw else ("推" if t.get("_auto") else
                                    ("缺" if t.get("_miss") else "已有")))
        rows.append({"ix": i, "cx": round(cx), "cy": round(cy),
                     "now": s.get("name") or "", "sug": t.get("name") or "",
                     "src": src,
                     "cand": " | ".join("%s@%d" % (txt, d) for d, txt in raw)
                             or " / ".join("%02d@%d" % (n, round(d)) for d, n, _t in cand)})
    return rows, dry


def autofill_shapes(shapes, text_items, sheet, num_set, width, height,
                    tol_ratio=0.006):
    """给一页的 Node 框补 LBD 名字。返回统计 dict。

    shapes: 这一页的框（就地改 name / _auto / _miss 标记，只动 label == "Node"）
    text_items: [(文字, fx中心, fy中心, fx1, fy1, fx2, fy2)]（底图归一化坐标）
    sheet: 本页分表名（来自标签表）
    num_set: 这个分表可用的 LBD 编号集合（用来滤掉框里的干扰文字）
    规则：
      1) 每条文字只归给离它最近的那个框（不是一个框把周围文字全拿走）
      2) 文字里的编号必须在这个分表的号码表里
      3) 文字自带 INV 分表名、且和本页分表不一致 -> 当干扰丢掉
      4) 框里没有可用文字的，按标签表的号（还没被用掉的）按位置顺序补，标黄
      5) 连标签表的号都没有了 -> 标红，等人手填
    """
    # 锁上的 Node **也补编号**：锁是防"误拖/误删/框选到"的，不是不让补号
    # （用户反馈：锁定 Node 之后跑「识别框内文字」编号写不进去 —— 就是这里被跳过了）
    locked_n = sum(1 for s in shapes
                   if s.get("label") == "Node" and s.get("locked"))
    nodes = [(i, s) for i, s in enumerate(shapes) if s.get("label") == "Node"]
    tol = max(6.0, float(tol_ratio) * max(width, height))
    stat = {"total": len(nodes), "filled": 0, "auto": 0, "missed": 0,
            "wrong_sheet": 0, "kept": 0, "locked": locked_n, "pos_added": 0}
    if not nodes:
        return stat

    def _find_label_mark(s):
        """框里那条编号文字印在哪儿：(中心, 小框) —— 都是页面像素坐标。

        优先"整段文字和名字一样"的，其次"号一样"的。名字已经有了（以前跑过、或者
        人填的）时也要能拿到 —— 用户反馈的"导出 JSON 里没有 lbd 编号位置"就是因为
        老逻辑见到已有名字就整框跳过。
        """
        name = (s.get("name") or "").strip()
        if not name:
            return None
        want = _name_num(name)
        nm = re.sub(r"\s+", "", name).upper()
        hit = None
        b = s["bbox"]
        for it in text_items:
            px, py = it[1] * width, (1.0 - it[2]) * height
            if not (b[0] <= px <= b[2] and b[1] <= py <= b[3]):
                continue
            up = re.sub(r"\s+", "", str(it[0] or "")).upper()
            same = (up == nm)
            if not same and (want is None or _lbd_num_in(it[0]) != want):
                continue
            # 文字项的框：x1,x2 直接换算；y 是自下而上的，要翻过来
            bb = (it[3] * width, (1.0 - it[6]) * height,
                  it[5] * width, (1.0 - it[4]) * height)
            if same:
                return (px, py), bb, (it[7] if len(it) > 7 else False)
            if hit is None:
                hit = ((px, py), bb, (it[7] if len(it) > 7 else False))
        return hit

    def _mark_positions():
        """给已经定了名字的框补上"编号印在哪儿"：label_pos（中心）+ label_bbox（小框）。

        已经有名字、只有中心没框的（老文件）也在这里补齐。按标签表顺序推出来的名字
        没有对应文字，给不出框，跳过。
        """
        for _i, s in nodes:
            nm = (s.get("name") or "").strip()
            if not nm or s.get("_auto"):
                continue
            raw = s.get("raw") if isinstance(s.get("raw"), dict) else {}
            if raw.get("label_pos") and raw.get("label_bbox"):
                continue
            mk = _find_label_mark(s)
            if not mk:
                continue
            (cx, cy), bb, exact = mk
            raw["label_pos"] = [int(round(cx)), int(round(cy))]
            raw["label_bbox"] = [int(round(v)) for v in bb]
            raw["label_src"] = raw.get("label_src") or "text_layer"
            raw["label_box_kind"] = "metrics" if exact else "estimated"
            s["raw"] = raw
            stat["pos_added"] = stat.get("pos_added", 0) + 1

    # 已经有人写过的名字：不动（免得把手工改的冲掉）
    todo = []
    for i, s in nodes:
        if (s.get("name") or "").strip() and not s.get("_auto"):
            stat["kept"] += 1
            s["_miss"] = False
        else:
            s["name"] = ""
            s["_miss"] = False
            s["_check"] = False
            todo.append((i, s))
    if not todo:
        _mark_positions()       # 名字都齐了，但可能缺位置/小框 -> 补上再返回
        return stat

    # 1) 候选文字：编号 + 分表名过滤 + 去掉图例/引线名单
    cands, bare, n_wrong = _page_candidates(text_items, sheet, num_set, width, height)
    stat["wrong_sheet"] = n_wrong

    # 2) 归属：① 文字中心落在哪个框里，就归那个框（谁的框谁拿字）
    #          ② 落在框外的，才按"最近 + 那个框还没拿到字"分给别人
    #    （老写法是"每条文字只给最近的框"，两个框抢同一条字时多的那条直接被丢掉，
    #      结果本该拿到它的框就空了 —— 表现就是"4 个框只读到 2 个"。）
    got = {}
    left_c = []
    pos_of = {}                       # 框下标 -> 那条编号文字的中心（页面像素）
    for px, py, num, _txt in cands:
        inside = None
        for i, s in todo:
            b = s["bbox"]
            if b[0] <= px <= b[2] and b[1] <= py <= b[3]:
                inside = i
                break
        if inside is None:
            left_c.append((px, py, num, _txt))   # 文字带上，别丢
            continue
        old = got.get(inside)
        if old is None or 0.0 < old[0]:
            got[inside] = (0.0, num, _txt)
            pos_of[inside] = (px, py)
    for px, py, num, _ltxt in left_c:
        best, bd = None, None
        for i, s in todo:
            if i in got:                 # 已经有字了，别再抢
                continue
            d = _box_dist(s["bbox"], px, py)
            if bd is None or d < bd:
                best, bd = i, d
        if best is None or bd is None or bd > tol:
            continue
        # 这里原来用的是外层循环残留的 txt（早就不在作用域了）——一走到这条路就崩
        got[best] = (bd, num, _ltxt)
        pos_of[best] = (px, py)

    # 3) 抢号：同一个号只留最近的那个框
    keep, keep_lab, full_name = {}, {}, {}
    for i, (d, num, txt) in sorted(got.items(), key=lambda kv: kv[1][0]):
        if num in keep:
            continue
        keep[i] = num
        keep_lab[i] = _clean_lbd_label(txt)
        nm = _full_lbd_name(txt)          # 整段编号（INV01A01-LBD-01）就原样用
        if nm:
            full_name[i] = nm

    used = set(keep.values())
    for i, s in todo:
        if i in keep:
            # 文字层里本身就是整段编号（INV01A01-LBD-01）-> 原样用，别拿分表名重拼
            # 编号整段带上（框里印的是 1.01.1.C.5 就写 1.01.1.C.5，不是只留 5）
            s["name"] = (full_name.get(i)
                         or _lbd_name(sheet, keep_lab.get(i, ""), keep[i]))
            # 把"这个编号印在图上的哪儿"一起记下来（主程序/核对都用得上）
            bb = pos_of.get(i)
            if bb:
                raw = s.get("raw") if isinstance(s.get("raw"), dict) else {}
                raw["label_pos"] = [int(round(bb[0])), int(round(bb[1]))]
                raw["label_src"] = "text_layer"
                s["raw"] = raw
            s["_auto"] = False
            stat["filled"] += 1

    # 3b) 框里只印裸编号的（文字层里没有 "LBD" 字样）：按"离框最近 + 号在号码表里优先"补。
    #     这正是"每个框里都有号、有的却取不到"那批 —— 以前只认带 LBD 字样的文字。
    if bare and len(keep) < len(todo):
        left = [t for t in todo if t[0] not in keep]
        bare_got = {}
        for px, py, val, txt in bare:
            best, bd = None, None
            for i, s in left:
                d = _box_dist(s["bbox"], px, py)
                if bd is None or d < bd:
                    best, bd = i, d
            if best is None or bd is None or bd > tol or best in bare_got:
                continue
            if val in used and any(v == val for _d, v, _t in [bare_got.get(best) or (0, -1, "")]):
                continue
            bare_got[best] = (bd, val, txt)
        # 同一号码只留一个框
        seen_num = set()
        for i in sorted(bare_got, key=lambda k: bare_got[k][0]):
            _d, val, _t = bare_got[i]
            if val in seen_num or val in used:
                continue
            # 号码不在这个分表号码表里 -> 当"文字不全/被截断"过滤掉，不填
            if sheet and num_set and val not in num_set:
                stat["dropped"] = stat.get("dropped", 0) + 1
                continue
            seen_num.add(val)
            s = dict(todo)[i]
            s["name"] = _lbd_name(sheet, "", val)
            s["_auto"] = False
            s["_check"] = False
            keep[i] = val
            used.add(val)
            stat["bare"] = stat.get("bare", 0) + 1
            stat["filled"] += 1

    # 3c) 最后兜底：框里的文字先全拿出来，只要里面带数字就试一把
    #     （"1.C.1"、"LBD-07"、"07" 都行；号码不在本分表表里的标紫，让人核）
    if len(keep) < len(todo):
        bt = _box_texts(shapes, text_items, width, height, tol_ratio)
        left = [t for t in todo if t[0] not in keep]
        picks = {}
        for i, _s in left:
            for d, txt, _tx, _ty in (bt.get(i) or []):
                # ★ 先把分表名（INV31B101 这种）从文字里剔掉再找数字，
                #   不然"任意数字"会抓到分表名里的 31，填出 INV31B101-LBD-31 这种鬼东西
                if txt.rstrip().endswith(("-", "_", ".", "(", "#", "|")):
                    continue           # 结尾是分隔符 -> 文字被截断了，不拿它猜号
                t2 = re.sub(r"INV\s*\d+\s*[A-Za-z]\s*\d+", " ", txt, flags=re.I)
                # "1.01.1.C.5" 这种：把所有数字段都拆出来，从**最后往前**找，
                # 只要有一个能在本分表号码表里对上就用它（不再要求带 "LBD" 字样）
                segs = [int(g) for g in re.findall(r"\d{1,3}", t2)]
                if not segs:
                    continue
                # 旁边那些旁注（"MV1A"、"TCH, TYP."、"LBD CLUSTER, TYP."）不是编号：
                # 以前会拿它们里的数字硬凑一个名字（用户看到 "LBD-MV1A" 就是这么来的）
                nm_ok, kind = ocr_text_to_name(txt)
                if kind == "noise" and not BARE_NUM_RE.match(txt.strip()):
                    stat["dropped"] = stat.get("dropped", 0) + 1
                    continue
                val = None
                for v in reversed(segs):
                    if not sheet or not num_set or v in num_set:
                        val = v
                        break
                if val is None:
                    stat["dropped"] = stat.get("dropped", 0) + 1
                    continue
                in_set = (not sheet or not num_set or val in num_set)
                picks.setdefault(i, []).append((0 if in_set else 1, d, val, txt, _tx, _ty))
        for i, lst in picks.items():
            if i in keep:
                continue
            lst.sort()
            pref = [t for t in lst if t[0] == 0]
            if not pref:
                # 框里那些字里的数字都不在本分表号码表里（多半是被拆开/截断的文字）
                # -> 过滤掉，不猜号
                stat["dropped"] = stat.get("dropped", 0) + 1
                continue
            flag, _d, val, pick_txt, pick_x, pick_y = pref[0]
            if val in used:
                continue
            s = dict(todo)[i]
            # 注意：只能用"挑中的那条文字"（pick_txt）拼名字。
            # 原来这里用的是外层循环残留的 txt，于是框里明明写着 INV01A01-LBD-06，
            # 名字却被拼成框边那条旁注（"LBD-MV1A"）—— 用户看到的就是这个。
            s["name"] = (_full_lbd_name(pick_txt)
                         or _lbd_name(sheet, _clean_lbd_label(pick_txt), val))
            raw = s.get("raw") if isinstance(s.get("raw"), dict) else {}
            raw["label_pos"] = [int(round(pick_x)), int(round(pick_y))]
            raw["label_src"] = "text_layer"
            s["raw"] = raw
            s["_auto"] = False
            s["_check"] = False
            keep[i] = val
            used.add(val)
            stat["any"] = stat.get("any", 0) + 1
            stat["filled"] += 1

    # 4) 框里没取到号的：按标签表的号顺序补（标黄，提示人工核）
    rest = [t for t in todo if t[0] not in keep]
    if rest and sheet and num_set:
        free = [n for n in sorted(num_set) if n not in used]
        # 位置顺序 = 一排排（先分行，行内从左到右）——和图纸上读的顺序一致
        pairs = [(round(_box_center(s["bbox"])[1], 3), round(_box_center(s["bbox"])[0], 3), i, s)
                 for i, s in rest]
        k = 0
        for row in _rows_of(pairs, _row_tol(shapes)):
            for _cy, _cx, i, s in row:
                if k >= len(free):
                    break
                s["name"] = _lbd_name(sheet, "", free[k])
                s["_auto"] = True
                stat["auto"] += 1
                k += 1
    for i, s in rest:
        if not (s.get("name") or "").strip():
            s["_miss"] = True
            stat["missed"] += 1
    # 4b) 把"编号印在哪儿"补全：位置 + 小框（画在图上给人核对用）
    _mark_positions()
    # 5) 位置可疑：同一行里号码跳号 -> 标紫（紫 = 从框内取到号了，但和同排的号对不上，
    #    也可能是框画错/取到了隔壁）；黄 = 按标签表顺序推的，同样要人核一眼。
    stat["sus"] = _flag_suspicious(shapes)
    stat["auto_list"] = [(s.get("name") or "") for _i, s in todo if s.get("_auto")]
    stat["sus_list"] = [(s.get("name") or "", round(_box_center(s["bbox"])[0]),
                         round(_box_center(s["bbox"])[1]))
                        for _i, s in todo if s.get("_check")]
    return stat


# ------------------------------------------------------- 字节级 JSON 定位
def skip_ws(b, i):
    while i < len(b) and b[i] in WS:
        i += 1
    return i


def skip_str(b, i):
    """b[i] 是开引号，返回闭引号之后的下标。"""
    i += 1
    while i < len(b):
        c = b[i]
        if c == 0x5C:
            i += 2
            continue
        if c == 0x22:
            return i + 1
        i += 1
    raise ValueError("字符串没有收尾")


def skip_val(b, i):
    """返回值的结束下标（不含）。"""
    i = skip_ws(b, i)
    c = b[i]
    if c == 0x22:
        return skip_str(b, i)
    if c in b"{[":
        depth = 0
        while i < len(b):
            c = b[i]
            if c == 0x22:
                i = skip_str(b, i)
                continue
            if c in b"{[":
                depth += 1
            elif c in b"}]":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        raise ValueError("容器没有收尾")
    j = i
    while j < len(b) and b[j] not in b",}] \t\r\n":
        j += 1
    return j


def top_spans(b):
    """对象 -> {键: (值起, 值止)}，只扫一层。"""
    out = {}
    i = skip_ws(b, 0)
    if b[i] != 0x7B:
        raise ValueError("不是对象")
    i += 1
    while True:
        i = skip_ws(b, i)
        if b[i] == 0x7D:
            return out
        ke = skip_str(b, i)
        key = json.loads(b[i:ke])
        i = skip_ws(b, ke)
        if b[i] != 0x3A:
            raise ValueError("缺少冒号")
        vs = skip_ws(b, i + 1)
        ve = skip_val(b, vs)
        out[key] = (vs, ve)
        i = skip_ws(b, ve)
        if b[i] == 0x2C:
            i += 1
            continue
        return out


def round_box_key(b):
    """和 lbd_regions.py 一样按 0.1 像素取整对齐 node_bbox。"""
    return (round(float(b["x1"]), 1), round(float(b["y1"]), 1),
            round(float(b["x2"]), 1), round(float(b["y2"]), 1))


class DebugJson:
    """识别结果 JSON 的字节级索引：图片按需取，段落按需解析，保存时定点替换。"""

    def __init__(self, path):
        self.path = path
        t0 = time.time()
        with open(path, "rb") as f:
            self.raw = f.read()
        self.load_seconds = time.time() - t0
        self.size = len(self.raw)
        self.sections = {}
        for key in SECTION_ORDER:
            span = self._find_array_span(key)
            if span:
                self.sections[key] = {"span": span, "elements": None}
        self.pages = {}
        self._index_pages()

    def _find_key(self, key):
        pat = ('"%s"' % key).encode()
        i = self.raw.find(pat)
        return None if i < 0 else i + len(pat)

    def _find_array_span(self, key):
        """返回 (数组 '[' 下标, ']' 之后下标)。用后一个顶层键兜底找收尾方括号。"""
        d = self.raw
        at = self._find_key(key)
        if at is None:
            return None
        i = skip_ws(d, at)
        if d[i] != 0x3A:
            return None
        i = skip_ws(d, i + 1)
        if d[i] != 0x5B:
            return None
        start = i
        limit = len(d)
        for other in ("yolo_tracker_detection_results", "yolo_box_detection_results",
                      "ocr_node_name_results", "claude_node_name_review_results"):
            if other == key:
                continue
            o = self._find_key(other)
            if o is not None and o > at:
                limit = min(limit, o)
        end = d.rfind(b"]", start, limit)
        if end < 0:
            end = skip_val(d, start) - 1
        return (start, end + 1)

    def _index_pages(self):
        """用 media_type 当锚点找每页元素（纯 C 速度的 find，不整份扫）。"""
        d = self.raw
        pos = 0
        while True:
            mt = d.find(b'"media_type"', pos)
            if mt < 0:
                break
            pos = mt + 1
            pn_at = d.rfind(b'"page_number"', max(0, mt - 400), mt)
            if pn_at < 0:
                continue
            colon = d.find(b":", pn_at, mt)
            m = re.match(rb"\s*(\d+)", d[colon + 1:colon + 32])
            if not m:
                continue
            number = int(m.group(1))
            width = height = 0
            for key, which in ((b'"width"', "w"), (b'"height"', "h")):
                at = d.find(key, mt, mt + 600)
                if at < 0:
                    continue
                c2 = d.find(b":", at, at + 20)
                m2 = re.match(rb"\s*(\d+)", d[c2 + 1:c2 + 32])
                if m2:
                    if which == "w":
                        width = int(m2.group(1))
                    else:
                        height = int(m2.group(1))
            pk = d.find(b'"png_base64"', mt, mt + 4000)
            q1 = q2 = -1
            if pk >= 0:
                q1 = d.find(b'"', pk + 12)
                q2 = d.find(b'"', q1 + 1)
            # 没有内嵌图片也要登记这一页（只存检测结果的 JSON 就是这样），
            # 底图到时候用 PDF 现渲染。
            self.pages[number] = {"page_number": number, "width": width,
                                  "height": height,
                                  "png_span": ((q1 + 1, q2) if 0 <= q1 < q2 else None)}

    def page_numbers(self):
        return sorted(self.pages)

    def png_bytes(self, number):
        import base64
        span = self.pages[number].get("png_span")
        if not span:
            raise RuntimeError("第 %s 页没有内嵌图片" % number)
        return base64.b64decode(self.raw[span[0]:span[1]])

    def _elements(self, key):
        """[{page, span, data_span}]，第一次用到才建。"""
        sec = self.sections.get(key)
        if sec is None:
            return []
        if sec["elements"] is None:
            d = self.raw
            start, end = sec["span"]
            out = []
            i = start + 1
            while True:
                i = skip_ws(d, i)
                if i >= end or d[i] == 0x5D:
                    break
                s = i
                e = skip_val(d, i)
                sub = d[s:e]
                try:
                    spans = top_spans(sub)
                    pn = int(json.loads(sub[spans["page_number"][0]:spans["page_number"][1]]))
                    ds, de = spans["data"]
                    out.append({"page": pn, "span": (s, e),
                                "data_span": (s + ds, s + de)})
                except Exception:
                    pass
                i = skip_ws(d, e)
                if i < len(d) and d[i] == 0x2C:
                    i += 1
            sec["elements"] = out
        return sec["elements"]

    def element_for(self, key, page_number):
        for el in self._elements(key):
            if el["page"] == page_number:
                return el
        return None

    def page_data(self, key, page_number):
        el = self.element_for(key, page_number)
        if not el:
            return None
        a, b = el["data_span"]
        return json.loads(self.raw[a:b])

    @staticmethod
    def _field_of(section):
        return {TRACKER_SECTION: "tracker", BOX_SECTION: "box",
                OCR_SECTION: "ocr"}[section]

    def save(self, dest, modified):
        """modified: {页号: {"tracker": dict, "box": dict, "ocr": list}}
        只替换这些页所在的段落，其余逐字节照搬。"""
        jobs = []
        for key in SECTION_ORDER:
            field = self._field_of(key)
            pages = {p: m for p, m in modified.items() if m.get(field) is not None}
            sec = self.sections.get(key)
            if not pages or not sec:
                continue
            jobs.append((sec["span"][0], key, field, pages, sec))
        # 必须从后往前替换：前面的数组一变长，后面记录的偏移量就作废了。
        jobs.sort(key=lambda j: -j[0])
        out = self.raw
        for a, key, field, pages, sec in jobs:
            parts = []
            seen = set()
            for el in self._elements(key):
                seen.add(el["page"])
                if el["page"] in pages:
                    elem = {"page_number": el["page"], "data": pages[el["page"]][field]}
                    parts.append(json.dumps(elem, ensure_ascii=False).encode("utf-8"))
                else:
                    s, e = el["span"]
                    parts.append(self.raw[s:e])
            # 原文件里没有这一页的记录（比如原来没检测到东西的页，现在手工画了框）：
            # 补一条，否则新画的框存不进去。
            for p in sorted(p for p in pages if p not in seen):
                elem = {"page_number": p, "data": pages[p][field]}
                parts.append(json.dumps(elem, ensure_ascii=False).encode("utf-8"))
            new_arr = b"[\n    " + b",\n    ".join(parts) + b"\n  ]"
            b = sec["span"][1]
            out = out[:a] + new_arr + out[b:]
        tmp = dest + ".part"
        with open(tmp, "wb") as f:
            f.write(out)
        os.replace(tmp, dest)


class PageModel:
    """一页的框 + 名字。shapes 是纯 dict，界面和自检共用。"""

    def __init__(self, dbg, page_number):
        self.dbg = dbg
        self.page = page_number
        info = dbg.pages[page_number]
        self.width = info["width"] or 6000
        self.height = info["height"] or 4000
        self.shapes = []
        self.dirty = False
        self._orig_tracker = None
        self._orig_box = None
        self._orig_ocr = None
        self.load()

    def load(self):
        dbg = self.dbg
        self._orig_tracker = dbg.page_data(TRACKER_SECTION, self.page) or {}
        self._orig_box = dbg.page_data(BOX_SECTION, self.page) or {}
        self._orig_ocr = dbg.page_data(OCR_SECTION, self.page) or []

        ocr_by_key = {}
        for idx, rec in enumerate(self._orig_ocr):
            nb = rec.get("node_bbox") or {}
            if all(k in nb for k in ("x1", "y1", "x2", "y2")):
                ocr_by_key[round_box_key(nb)] = idx

        self.shapes = []
        for det in (self._orig_tracker.get("detections") or []):
            b = det.get("bbox") or {}
            if not all(k in b for k in ("x1", "y1", "x2", "y2")):
                continue
            label = (det.get("label") or "").strip() or "Tracker"
            ocr_index, name = None, ""
            if label.lower() == "node":
                ocr_index = ocr_by_key.get(round_box_key(b))
                if ocr_index is not None:
                    rec = self._orig_ocr[ocr_index]
                    name = (rec.get("final_node_name")
                            or rec.get("preliminary_node_name") or "").strip()
            self.shapes.append({
                "label": label, "name": name,
                "bbox": [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])],
                "confidence": det.get("confidence"),
                "class_id": det.get("class_id"),
                "source": det.get("source") or "manual",
                "raw": det.get("raw") or {},
                "ocr_index": ocr_index,
                "locked": bool((det.get("raw") or {}).get("lbd_locked")),
            })
        for det in (self._orig_box.get("detections") or []):
            b = det.get("bbox") or {}
            if not all(k in b for k in ("x1", "y1", "x2", "y2")):
                continue
            self.shapes.append({
                "label": (det.get("label") or "Box").strip(),
                "name": (det.get("description") or "").strip(),
                "bbox": [float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])],
                "confidence": det.get("confidence"),
                "class_id": det.get("class_id"),
                "source": det.get("source") or "manual",
                "raw": det.get("raw") or {},
                "ocr_index": None,
                "locked": bool((det.get("raw") or {}).get("lbd_locked")),
            })

    def counts(self):
        c = {k: 0 for k in CLASSES}
        for s in self.shapes:
            c[s["label"]] = c.get(s["label"], 0) + 1
        return c

    def _det(self, s):
        cid = s.get("class_id")
        # 锁定状态记在 raw 里：raw 是原样透传的字典，存进去重开还在，
        # 也不用给下游 JSON 多塞一个新字段（下游只认 label/bbox 那几个）。
        raw = dict(s.get("raw") or {})
        if s.get("locked"):
            raw["lbd_locked"] = True
        else:
            raw.pop("lbd_locked", None)
        return {
            "label": s["label"],
            "confidence": s.get("confidence"),
            "class_id": cid if cid is not None else DEFAULT_CLASS_ID.get(s["label"], 0),
            "bbox": {"x1": s["bbox"][0], "y1": s["bbox"][1],
                     "x2": s["bbox"][2], "y2": s["bbox"][3]},
            "source": s.get("source") or "manual",
            "raw": raw,
        }

    def tracker_data(self):
        data = copy.deepcopy(self._orig_tracker) if self._orig_tracker else {
            "model_type": "tracker", "coordinates": "original_page_pixels",
            "detections": [], "error": None}
        data["detections"] = [self._det(s) for s in self.shapes
                              if s["label"] in ("Node", "Tracker")]
        return data

    def box_data(self):
        data = copy.deepcopy(self._orig_box) if self._orig_box else {
            "model_type": "box", "detections": [], "error": None}
        data["detections"] = [self._det(s) for s in self.shapes if s["label"] == "Box"]
        return data

    def ocr_data(self):
        """Node 框 -> ocr 记录一一对应；人改的名字写进 final_node_name。"""
        orig = self._orig_ocr or []
        used = set()
        for rec in orig:
            try:
                used.add(int(rec.get("node_index")))
            except Exception:
                pass
        nxt = 1
        out = []
        for s in self.shapes:
            if s["label"] != "Node":
                continue
            idx = s.get("ocr_index")
            if idx is not None and 0 <= idx < len(orig):
                rec = copy.deepcopy(orig[idx])
            else:
                while nxt in used:
                    nxt += 1
                used.add(nxt)
                rec = {
                    "node_index": nxt,
                    "entity_id": "p%d:node:%d" % (self.page, nxt),
                    "readings": [],
                    "selected": {"text": s["name"], "angle_deg": 0, "confidence": None},
                    "matched_table_name": None,
                    "preliminary_node_name": s["name"],
                    "error": None,
                }
            rec["node_bbox"] = {"x1": s["bbox"][0], "y1": s["bbox"][1],
                                "x2": s["bbox"][2], "y2": s["bbox"][3]}
            rec["final_node_name"] = s["name"]
            # 编号印在哪儿：名字旁边再放一份，省得下游为了拿位置还得去翻 detections.raw
            raw = s.get("raw") if isinstance(s.get("raw"), dict) else {}
            if raw.get("label_pos"):
                rec["label_pos"] = [int(raw["label_pos"][0]), int(raw["label_pos"][1])]
                rec["label_src"] = raw.get("label_src") or ""
                if raw.get("label_bbox"):
                    rec["label_bbox"] = [int(v) for v in raw["label_bbox"]]
            else:
                rec.pop("label_pos", None)
                rec.pop("label_src", None)
                rec.pop("label_bbox", None)
            out.append(rec)
        return out

    def result(self):
        return {"tracker": self.tracker_data(), "box": self.box_data(),
                "ocr": self.ocr_data()}


def selftest(src, out=None):
    """无界面走一遍：载入 -> 改框改名 -> 另存 -> 用下游代码验证。"""
    here = os.path.dirname(os.path.abspath(__file__))
    out = out or os.path.join(app_dir(), "selftest_annotated.json")
    print("源文件:", src)
    dbg = DebugJson(src)
    pgs = dbg.page_numbers()
    print("页数 %d（%d..%d），读入 %.2fs，%.1f MB"
          % (len(pgs), pgs[0], pgs[-1], dbg.load_seconds, dbg.size / 1048576))

    page = pgs[0]
    t0 = time.time()
    pm = PageModel(dbg, page)
    c0 = pm.counts()
    print("第 %d 页解析 %.2fs：Node %d / Tracker %d / Box %d"
          % (page, time.time() - t0, c0["Node"], c0["Tracker"], c0["Box"]))

    nodes = [s for s in pm.shapes if s["label"] == "Node"]
    trks = [s for s in pm.shapes if s["label"] == "Tracker"]
    if nodes:
        n0 = nodes[0]
        print("  原名字:", n0["name"], "原位置: %.0f,%.0f" % (n0["bbox"][0], n0["bbox"][1]))
        n0["bbox"] = [n0["bbox"][0] + 12, n0["bbox"][1] + 12,
                      n0["bbox"][2] + 12, n0["bbox"][3] + 12]
        n0["name"] = "SELFTEST-LBD-99"
    if len(trks) > 1:
        t = trks[-1]
        pm.shapes.append({"label": "Tracker", "name": "", "confidence": None,
                          "class_id": 0, "source": "manual", "raw": {},
                          "ocr_index": None,
                          "bbox": [t["bbox"][0], t["bbox"][1] + 5,
                                   t["bbox"][2], t["bbox"][3] + 5]})
        pm.shapes.remove(trks[0])
    c1 = pm.counts()
    print("改动：Node %d->%d（首框平移 12px 并改名），Tracker %d->%d（末尾加一个、删首一个）"
          % (c0["Node"], c1["Node"], c0["Tracker"], c1["Tracker"]))

    t0 = time.time()
    dbg.save(out, {page: pm.result()})
    print("另存 %.2fs -> %s（%.1f MB）"
          % (time.time() - t0, out, os.path.getsize(out) / 1048576))

    sys.path.insert(0, os.path.join(os.path.dirname(here), "CAD-MAP-main", "编排器"))
    import lbd_regions as lr
    lines_path = os.path.join(here, "selftest_lines.txt")
    r = lr.extract_lines_from_debug(out, lines_path, prefix="STR")
    rows = open(lines_path, encoding="utf-8").read().splitlines()
    names = [ln.split("\t")[4] for ln in rows if "-LBD-" in ln.upper()]
    # 下游是按「名字以 STR 开头」认支架号的，LBD 行就是其余那些
    pcols = [ln.split("\t") for ln in rows if ln.split("\t")[1] == str(page)]
    page_str = [c[4] for c in pcols if c[4].upper().startswith("STR")]
    page_lbd = [c[4] for c in pcols if not c[4].upper().startswith("STR")]
    print("下游 extract_lines_from_debug：ok=%s lines=%s lbd=%s str=%s pages=%s"
          % (r.get("ok"), r.get("lines"), r.get("lbd"), r.get("str"), r.get("pages")))
    ok_name = "SELFTEST-LBD-99" in names
    ok_lbd = len(page_lbd) == c1["Node"]
    ok_str = len(page_str) == c1["Tracker"]
    print("改的名字被下游读到:", ok_name)
    print("第 %d 页 LBD 行数 == 该页 Node 数: %s (%d vs %d)"
          % (page, ok_lbd, len(page_lbd), c1["Node"]))
    print("第 %d 页 支架行数 == 该页 Tracker 数: %s (%d vs %d)"
          % (page, ok_str, len(page_str), c1["Tracker"]))
    print("名字示例:", names[:3])
    good = bool(r.get("ok")) and ok_name and ok_lbd and ok_str
    print("自检:", "通过" if good else "失败")
    return 0 if good else 1


def _kmeans_cuts(values, k, iters=80):
    """把一堆长度分成 k 档，返回 k-1 个切点（切在长度间隙大的地方）。

    支架分档用：不能用"等分位"硬切 —— 那样长度几乎一样的支架，只要一个落在切线
    左边、一个落在右边，就会被分到两类（用户看到的就是"明明一样长却分了两种"）。
    这里做一维 k-means，切点落在两类中心之间，长度接近的必然是同一类。
    """
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if k <= 1 or n <= k:
        return []
    cent = [vals[min(n - 1, int(n * (i + 0.5) / k))] for i in range(k)]
    for _ in range(iters):
        groups = [[] for _ in range(k)]
        for v in vals:
            groups[min(range(k), key=lambda i: abs(v - cent[i]))].append(v)
        new = [sum(g) / len(g) if g else cent[i] for i, g in enumerate(groups)]
        if new == cent:
            break
        cent = new
    cent = sorted(cent)
    return [(cent[i] + cent[i + 1]) / 2.0 for i in range(k - 1)]


def _as_pages(entry):
    """撤销栈里的一步：老写法是 (页号, 快照)，新写法是 [(页号, 快照), ...]（整册操作一步多页）。"""
    if isinstance(entry, tuple):
        return [entry]
    return list(entry)


def make_gui_classes():
    """延迟导入 Qt（自检模式不需要 Qt）。"""
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QImage, QKeySequence,
                               QPainter, QPen, QPixmap)
    from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout,
                                   QGraphicsItem, QGraphicsRectItem, QGraphicsScene,
                                   QGraphicsView, QHBoxLayout, QLabel, QLineEdit,
                                   QMainWindow, QMessageBox, QPushButton, QStatusBar,
                                   QToolBar, QVBoxLayout, QWidget)

    COLORS = {"Node": QColor(0, 120, 255), "Tracker": QColor(255, 40, 40),
              "Box": QColor(0, 200, 0)}
    HANDLE_CURSORS = [Qt.CursorShape.SizeFDiagCursor,
                      Qt.CursorShape.SizeVerCursor,
                      Qt.CursorShape.SizeBDiagCursor,
                      Qt.CursorShape.SizeHorCursor,
                      Qt.CursorShape.SizeHorCursor,
                      Qt.CursorShape.SizeBDiagCursor,
                      Qt.CursorShape.SizeVerCursor,
                      Qt.CursorShape.SizeFDiagCursor]

    class BoxItem(QGraphicsRectItem):
        # 名字字号（屏幕像素）：工具栏可调，默认小号不挡图
        name_px = 11.0
        # True = 只给"选中的那个框"画名字（默认，图上不会一堆名字压在一起）
        name_sel_only = True
        # 自动补的号（标黄）/ 没取到号（标红）用的颜色
        AUTO_COLOR = QColor(230, 150, 0)
        MISS_COLOR = QColor(255, 40, 40)
        CHECK_COLOR = QColor(150, 60, 220)
        # 补标/识别建议框：统一橙色 + 虚线（一眼就能和原有标注区分），强弱看框上的置信度数字
        SUG_COLOR = QColor(255, 120, 0)
        show_conf = True                               # 是否在建议框上标置信度（工具栏可切）

        def __init__(self, shape):
            super().__init__(0, 0, shape["bbox"][2] - shape["bbox"][0],
                             shape["bbox"][3] - shape["bbox"][1])
            self.shape_data = shape          # 不能叫 self.shape：会盖掉 Qt 的虚函数 shape()
            self.setPos(shape["bbox"][0], shape["bbox"][1])
            # 建议框压在上层：这样即使和原有标注重叠，虚线和置信度颜色也看得见
            self.setZValue(12 if self.is_suggest() else 10)
            self.apply_flags()
            self.apply_pen()

        def apply_pen(self):
            sug = self.is_suggest()
            ctx = self.shape_data.get("source") == "context"
            if self.shape_data.get("_miss"):
                c = self.MISS_COLOR
            elif self.shape_data.get("_check"):
                c = self.CHECK_COLOR
            elif self.shape_data.get("_auto"):
                c = self.AUTO_COLOR
            elif sug:
                c = BoxItem.SUG_COLOR
            else:
                c = COLORS.get(self.shape_data["label"], QColor(255, 0, 255))
            pen = QPen(c)
            pen.setCosmetic(True)
            pen.setWidthF(3.2 if sug else 2.0)
            alpha = 55 if sug else 26
            if ctx and not sug:
                # 补标包里带来的"原有标注"：只是参照，画细实线、不填充，
                # 让彩色虚线的建议框一眼就能挑出来（类别颜色仍然保留）。
                # 颜色再往白里调淡一档 —— 屏幕上只留建议框一个"重颜色"。
                base = COLORS.get(self.shape_data["label"], QColor(170, 170, 170))
                c = QColor((base.red() * 2 + 255 * 3) // 5,
                           (base.green() * 2 + 255 * 3) // 5,
                           (base.blue() * 2 + 255 * 3) // 5)
                pen.setColor(c)
                pen.setWidthF(1.2)
                alpha = 0
                self.setPen(pen)
                self.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 0)))
                return
            if self.is_locked() or sug:
                # 锁上的框：虚线 + 更淡的底，一眼看得出"这个不能动"
                # 建议框：同样虚线 —— 虚线＝"还没人工确认过"
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setWidthF(1.6)
                alpha = 10 if self.is_locked() else 14
            self.setPen(pen)
            self.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), alpha)))

        def is_suggest(self):
            """模型建议补的框（类别名带问号 / 标了 model_suggest）——单独配色，别和确认过的框混一起。"""
            if self.shape_data.get("source") == "suggest":
                return True
            raw = self.shape_data.get("raw") or {}
            return str(raw.get("xl_label") or "").endswith("?")

        def conf_value(self):
            """建议框的置信度数值（拿不到算 0）。"""
            try:
                return float(self.shape_data.get("confidence") or 0.0)
            except Exception:
                return 0.0

        def is_locked(self):
            return bool(self.shape_data.get("locked"))

        def apply_flags(self):
            """锁上 = 点不中、框选也框不到、更拖不动；解锁恢复。"""
            flag = QGraphicsItem.GraphicsItemFlag
            lock = self.is_locked()
            self.setFlag(flag.ItemIsSelectable, not lock)
            self.setFlag(flag.ItemIsMovable, not lock)
            if lock and self.isSelected():
                self.setSelected(False)

        def set_locked(self, flag):
            self.shape_data["locked"] = bool(flag)
            self.apply_flags()
            self.apply_pen()
            self.update()

        def scene_box(self):
            r, p = self.rect(), self.pos()
            return [p.x(), p.y(), p.x() + r.width(), p.y() + r.height()]

        def sync_shape(self):
            self.shape_data["bbox"] = self.scene_box()

        def itemChange(self, change, value):
            if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
                self.sync_shape()
            return super().itemChange(change, value)

        def view_scale(self):
            views = self.scene().views() if self.scene() else []
            return views[0].transform().m11() if views else 1.0

        def paint(self, painter, option, widget=None):
            super().paint(painter, option, widget)
            sc = self.view_scale()
            if self.is_suggest() and BoxItem.show_conf:
                self._paint_conf(painter, sc)
            name = self.shape_data.get("name") or ""
            if name and sc > 0.02 and (not BoxItem.name_sel_only or self.isSelected()):
                self._paint_name(painter, name, sc)
            if self.is_locked():
                self._paint_lock(painter, sc)
            if self.isSelected():
                h = 9.0 / max(sc, 1e-6)
                r = self.rect()
                pts = [(r.left(), r.top()), (r.center().x(), r.top()),
                       (r.right(), r.top()), (r.left(), r.center().y()),
                       (r.right(), r.center().y()), (r.left(), r.bottom()),
                       (r.center().x(), r.bottom()), (r.right(), r.bottom())]
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                pen = QPen(QColor(0, 0, 0))
                pen.setCosmetic(True)
                painter.setPen(pen)
                for x, y in pts:
                    painter.drawRect(QRectF(x - h / 2, y - h / 2, h, h))

        def _paint_lock(self, painter, sc):
            """锁上的框：右上角画个小锁（屏幕上恒定大小，不随缩放变形）。

            放右上角是为了不和框里那行名字（左对齐画的）打架。
            """
            r = self.rect()
            if min(r.width(), r.height()) * sc < 9.0:
                return
            k = 1.0 / max(sc, 1e-6)
            bw, bh = 8.0 * k, 6.5 * k
            x, y = r.right() - bw - 3.0 * k, r.top() + 2.0 * k
            painter.save()
            pen = QPen(QColor(40, 40, 40))
            pen.setCosmetic(True)
            pen.setWidthF(1.4)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 245, 180, 235)))
            painter.drawRect(QRectF(x, y + bh * 0.42, bw, bh * 0.58))
            painter.drawArc(QRectF(x + bw * 0.12, y, bw * 0.76, bh * 0.9), 0, 180 * 16)
            painter.restore()

        def _paint_conf(self, painter, sc):
            """建议框标出置信度数字（屏幕上恒定小字号）。

            框在屏幕上太小时不画（一页两三百个框，全画会糊成一片）；
            放大到能看清时自动出现，选中的那个框一律画。
            """
            conf = self.shape_data.get("confidence")
            if conf is None:
                return
            r = self.rect()
            w, h = max(r.width(), 1.0), max(r.height(), 1.0)
            # 屏幕上太挤就不画数字（一页两三百个框）：放大到长边 70px 以上才出现，
            # 选中的那个框无条件画。
            if max(w, h) * sc < 70.0 and not self.isSelected():
                return
            try:
                txt = "%.2f" % float(conf)
            except Exception:
                return
            size = float(BoxItem.name_px)
            if size * sc < 4.5:
                return
            f = QFont()
            f.setBold(True)
            f.setPixelSize(max(1, int(size)))
            painter.save()
            painter.setFont(f)
            box = QRectF(r.left() + 1.0, r.top() + 1.0,
                         max(12.0, len(txt) * size * 0.66), size * 1.3)
            painter.fillRect(box, QColor(255, 255, 255, 200))
            painter.setPen(QPen(BoxItem.SUG_COLOR))     # 统一橙色（强弱看数字就行）
            painter.drawText(box, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, txt)
            painter.restore()

        def _paint_name(self, painter, name, sc):
            """把 LBD 名字画在框里：屏幕上恒定小字号、永远不出自己的框，所以不会互相压。

            横放和竖放哪个能排得下就选哪个（图上那种细长条带框会走竖排）。
            """
            if BoxItem.name_px <= 0:
                return
            r = self.rect()
            w, h = max(r.width(), 1.0), max(r.height(), 1.0)
            need = max(1, len(name)) * 0.58
            # 字号上限：屏幕上恒定 name_px；再夹住"文字厚度"，保证永远不出自己的框
            base = min(BoxItem.name_px / sc, max(0.0, (min(w, h) - 2.0) / 1.4))
            if base <= 0:
                return
            horiz = min(base, (w - 2.0) / need)
            vert = min(base, (h - 2.0) / need)
            size = max(horiz, vert)
            if size * sc < 4.0:            # 屏幕上太小了，干脆不画
                return
            f = QFont()
            f.setBold(True)
            f.setPixelSize(max(1, int(size)))
            col = (self.MISS_COLOR if self.shape_data.get("_miss")
                   else (self.CHECK_COLOR if self.shape_data.get("_check")
                         else (self.AUTO_COLOR if self.shape_data.get("_auto")
                               else QColor(0, 140, 0))))
            painter.save()
            painter.setFont(f)
            if horiz >= vert:
                box = QRectF(r.left() + 1.0, r.top() + 1.0, w - 2.0, size * 1.35)
                painter.fillRect(box, QColor(255, 255, 255, 200))
                painter.setPen(QPen(col))
                painter.drawText(box, Qt.AlignmentFlag.AlignLeft
                                 | Qt.AlignmentFlag.AlignTop, name)
            else:
                # 竖排：从框的左下角往上排（和图纸上那些转 90° 的标注一样）
                box = QRectF(0.0, 0.0, h - 2.0, size * 1.4)
                painter.translate(r.left() + 1.0, r.bottom() - 1.0)
                painter.rotate(-90.0)
                painter.fillRect(box, QColor(255, 255, 255, 200))
                painter.setPen(QPen(col))
                painter.drawText(box, Qt.AlignmentFlag.AlignLeft
                                 | Qt.AlignmentFlag.AlignTop, name)
            painter.restore()

    class Canvas(QGraphicsView):
        def __init__(self, win):
            super().__init__()
            self.win = win
            self.mode = "select"
            self._pan = False
            self._pan_pt = None
            self._rubber = None
            self._origin = None
            self._resize = None
            self.setRenderHints(QPainter.RenderHint.Antialiasing
                                | QPainter.RenderHint.SmoothPixmapTransform)
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        def set_mode(self, mode):
            self.mode = mode
            if mode == "select":
                self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
                self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            else:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                self.viewport().setCursor(Qt.CursorShape.CrossCursor)

        def cancel_draw(self):
            """画到一半按 ESC：把临时框丢掉。返回是不是真的取消了。"""
            if self._rubber is not None:
                self.scene().removeItem(self._rubber)
                self._rubber = None
                self._origin = None
                return True
            return False

        def keyPressEvent(self, ev):
            # 键盘统一交给主窗口处理（ESC/Delete/A/D…），避免两处各一套
            self.win.keyPressEvent(ev)

        def zoom_step(self, f, anchor=None):
            """以 anchor（视口坐标；不给就用视口中心）为锚点缩放，返回是否真的缩放了。"""
            cur = self.transform().m11()
            if not 0.01 < cur * f < 40:
                return False
            # 缩放的锚点自己算，别用 AnchorUnderMouse：Qt 那个「鼠标在哪儿」只在
            # 基类的 mousePress/mouseMove 里更新，而画框（两点画法）、拖控制点、
            # 中键平移这几条路径我们都提前 return 了、没把事件交给基类 —— Qt 记的
            # 位置还停在旧的（甚至退回视图中心），于是「画 node/tracker 点了第一个
            # 点之后，滚轮缩放不跟十字光标走」。
            # 这里改成：缩放前后各算一次锚点下的场景坐标，差多少补多少。跟 Qt 记的
            # 位置无关，画框中、拖框时、fitInView / 100% 之后都永远贴着锚点。
            pos = anchor if anchor is not None else self.viewport().rect().center()
            keep = self.transformationAnchor()
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
            before = self.mapToScene(pos)
            self.scale(f, f)
            after = self.mapToScene(pos)
            # 注意顺序：translate 会改视图变换，锚点还没恢复成 AnchorUnderMouse 时
            # 调用，否则 Qt 会拿它记的（旧的）鼠标位置把视图再居中一次，白补。
            self.translate(after.x() - before.x(), after.y() - before.y())
            self.setTransformationAnchor(keep)
            self.win.update_state_label()
            return True

        def wheelEvent(self, ev):
            self.zoom_step(1.18 if ev.angleDelta().y() > 0 else 1 / 1.18,
                           ev.position().toPoint())

        def drawForeground(self, painter, rect):
            """把「识别到的编号框」画出来（工具栏「编号框」开关控制）。

            数据来自每个 Node 的 raw.label_pos / raw.label_bbox：
              · label_bbox 有 -> 画小方框（文字层给的是估算框，OCR 给的是实测框）
              · 只有 label_pos -> 画一个小十字
            线宽用 cosmetic，放多大都不变粗；只画当前页，不参与选中/拖动。
            """
            win = self.win
            if not getattr(win, "show_label_boxes", False) or not win.pm:
                return
            sc = max(self.transform().m11(), 1e-6)
            box_pen = QPen(QColor(255, 140, 0), 1.6)
            box_pen.setCosmetic(True)
            cross_pen = QPen(QColor(255, 60, 60), 1.6)
            cross_pen.setCosmetic(True)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for s in win.pm.shapes:
                if s.get("label") != "Node":
                    continue
                raw = s.get("raw") if isinstance(s.get("raw"), dict) else {}
                bb = raw.get("label_bbox")
                if bb and len(bb) == 4:
                    painter.setPen(box_pen)
                    painter.drawRect(QRectF(float(bb[0]), float(bb[1]),
                                            float(bb[2]) - float(bb[0]),
                                            float(bb[3]) - float(bb[1])))
                pos = raw.get("label_pos")
                if pos:
                    painter.setPen(cross_pen)
                    r = 7.0 / sc
                    painter.drawLine(QPointF(pos[0] - r, pos[1]), QPointF(pos[0] + r, pos[1]))
                    painter.drawLine(QPointF(pos[0], pos[1] - r), QPointF(pos[0], pos[1] + r))

        def mousePressEvent(self, ev):
            if ev.button() == Qt.MouseButton.MiddleButton:
                self._pan = True
                self._pan_pt = ev.position()
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                return
            if ev.button() == Qt.MouseButton.LeftButton and self.mode == "select":
                # 锁上的框点不中（selectable=False），所以单独解锁得留个入口：
                # Ctrl+点它 = 只解锁这一个。
                if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    hit = self.itemAt(ev.position().toPoint())
                    if isinstance(hit, BoxItem) and hit.is_locked():
                        self.win.unlock_item(hit)
                        return
                it = self.win.current_item()
                if it is not None:
                    sp = self.mapToScene(ev.position().toPoint())
                    idx = self.handle_at(it, sp)
                    if idx is not None:
                        self._resize = {"item": it, "idx": idx, "start": sp,
                                        "box": it.scene_box()}
                        self.win.begin_change()
                        return
            if ev.button() == Qt.MouseButton.LeftButton and self.mode != "select":
                sp = self.mapToScene(ev.position().toPoint())
                # 两点画法：已经点过第一角了，这次点击就是对角点，直接成框
                if self._rubber is not None:
                    r = self._rubber.rect()
                    self.scene().removeItem(self._rubber)
                    self._rubber = None
                    self._origin = None
                    if r.width() >= 3 and r.height() >= 3:
                        self.win.add_shape(self.mode,
                                           [r.left(), r.top(), r.right(), r.bottom()])
                    return
                self._origin = sp
                c = COLORS.get(self.mode, QColor(255, 0, 255))
                pen = QPen(c)
                pen.setCosmetic(True)
                pen.setWidthF(2.0)
                self._rubber = self.scene().addRect(
                    QRectF(sp, sp), pen,
                    QBrush(QColor(c.red(), c.green(), c.blue(), 40)))
                self._rubber.setZValue(50)
                return
            self.win.begin_change()
            super().mousePressEvent(ev)

        def mouseMoveEvent(self, ev):
            if self._pan and self._pan_pt is not None:
                d = ev.position() - self._pan_pt
                self._pan_pt = ev.position()
                self.horizontalScrollBar().setValue(
                    int(self.horizontalScrollBar().value() - d.x()))
                self.verticalScrollBar().setValue(
                    int(self.verticalScrollBar().value() - d.y()))
                return
            if self._rubber is not None and self._origin is not None:
                sp = self.mapToScene(ev.position().toPoint())
                self._rubber.setRect(QRectF(self._origin, sp).normalized())
                return
            if self._resize is not None:
                d = self._resize
                p = self.mapToScene(ev.position().toPoint())
                box = self.apply_handle(d["box"], d["idx"],
                                        p.x() - d["start"].x(), p.y() - d["start"].y())
                it = d["item"]
                it.setPos(box[0], box[1])
                it.setRect(QRectF(0, 0, box[2] - box[0], box[3] - box[1]))
                it.sync_shape()
                it.update()
                return
            if not self._pan and self.mode == "select":
                it = self.win.current_item()
                if it is not None:
                    sp = self.mapToScene(ev.position().toPoint())
                    i = self.handle_at(it, sp)
                    self.viewport().setCursor(HANDLE_CURSORS[i] if i is not None
                                              else Qt.CursorShape.ArrowCursor)
            super().mouseMoveEvent(ev)
            # 拖动框的时候实时同步坐标（Qt 不会给 itemChange 派发"位置变了"）
            if self.mode == "select" and not self._pan and self._resize is None:
                self.win.sync_shapes()

        def mouseReleaseEvent(self, ev):
            if ev.button() == Qt.MouseButton.MiddleButton:
                self._pan = False
                self.viewport().setCursor(
                    Qt.CursorShape.CrossCursor if self.mode != "select"
                    else Qt.CursorShape.ArrowCursor)
                return
            if self._rubber is not None and ev.button() == Qt.MouseButton.LeftButton:
                r = self._rubber.rect()
                # 只是点了一下（面积≈0）→ 留在"两点画法"里等第二次点击；
                # 拖着画出来的（有点面积）→ 按老习惯直接成框
                if r.width() < 3 or r.height() < 3:
                    return
                self.scene().removeItem(self._rubber)
                self._rubber = None
                self._origin = None
                if r.width() >= 3 and r.height() >= 3:
                    self.win.add_shape(self.mode,
                                       [r.left(), r.top(), r.right(), r.bottom()])
                return
            if self._resize is not None and ev.button() == Qt.MouseButton.LeftButton:
                self._resize = None
                self.win.end_change()
                self.win.on_selection()
                return
            super().mouseReleaseEvent(ev)
            # ★ 必须自己同步：这个 PySide6 版本的 QGraphicsItem::itemChange 收不到
            #   ItemPositionHasChanged，靠它同步的话"拖动框"永远不会写回数据，
            #   保存出来还是老坐标（用户反馈的"移动支架位置保存不生效"就是这个）。
            self.win.sync_shapes()
            self.win.end_change()

        @staticmethod
        def handle_points(box):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            return [(x1, y1), (cx, y1), (x2, y1), (x1, cy),
                    (x2, cy), (x1, y2), (cx, y2), (x2, y2)]

        @staticmethod
        def apply_handle(box, idx, dx, dy):
            """按第 idx 个控制点拖动 (dx,dy)，返回新的 [x1,y1,x2,y2]（允许拖过头翻转）。"""
            x1, y1, x2, y2 = box
            if idx in (0, 3, 5):
                x1 += dx
            if idx in (2, 4, 7):
                x2 += dx
            if idx in (0, 1, 2):
                y1 += dy
            if idx in (5, 6, 7):
                y2 += dy
            nx1, nx2 = (x1, x2) if x1 <= x2 else (x2, x1)
            ny1, ny2 = (y1, y2) if y1 <= y2 else (y2, y1)
            return [nx1, ny1, nx2, ny2]

        def handle_at(self, item, scene_pos):
            """鼠标是不是落在选中框的控制点上，返回控制点序号。"""
            sc = max(self.transform().m11(), 1e-6)
            tol = 9.0 / sc * 0.8
            for i, (x, y) in enumerate(self.handle_points(item.scene_box())):
                if abs(scene_pos.x() - x) <= tol and abs(scene_pos.y() - y) <= tol:
                    return i
            return None

    return BoxItem, Canvas, COLORS


def run_gui(path=None, smoke=False, memtest=0.0, roundtrip=False):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
    from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFormLayout,
                                   QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                                   QMessageBox, QPushButton, QStatusBar, QToolBar,
                                   QVBoxLayout, QWidget)

    BoxItem, Canvas, COLORS = make_gui_classes()
    MODE_TEXT = {"select": "选择", "Node": "画 Node", "Tracker": "画 Tracker",
                 "Box": "画 Box"}

    class Win(QMainWindow):
        def __init__(self, src=None):
            super().__init__()
            self.setWindowTitle("LBD 标注工具 v%s" % ANNOTATOR_VERSION)
            self.resize(1500, 950)
            self.dbg = None
            self.page = None
            self.pm = None
            self.items = []
            self.undo = []
            self.redo = []
            self._pending = None
            self.edited = {}
            self._pixmap = None
            self.clip = []                 # 复制/粘贴框用的内部剪贴板
            self._paste_n = 0
            self.settings = load_settings()
            # 不读"上次的 PDF"：路径不落盘，每次都是干净开局（要 PDF 自己点「选 PDF…」）
            self.pdf = ""
            self._ptext = None          # PDF 文字层缓存（换 PDF 时要清掉）
            self._folder_mode = False   # True = 正在审「补标文件夹」（底图来自 png）
            self.dpi = int(self.settings.get("dpi") or 0)
            self.xlsx = self.settings.get("xlsx") or ""
            self._sheet_cache = None
            self.poppler = find_pdftoppm()
            self.renderer = Renderer(self.poppler,
                                     cache_dir())
            self._img_note = ""
            # 窗口：给个合理的最小尺寸（不然侧栏会被挤没），大小记住上次的
            self.setMinimumSize(1180, 740)
            try:
                w = int(self.settings.get("win_w") or 0)
                h = int(self.settings.get("win_h") or 0)
                if w > 600 and h > 400:
                    self.resize(w, h)
            except Exception:
                pass

            self.scene = self._make_scene()
            self.canvas = Canvas(self)
            self.canvas.setScene(self.scene)

            tb = QToolBar("工具栏")
            tb.setMovable(False)
            self.addToolBar(tb)
            # 画图相关的工具栏放左边（竖排），保存/输出相关的留在上面
            tb2 = QToolBar("画图")
            tb2.setMovable(False)
            try:
                tb2.setOrientation(Qt.Orientation.Vertical)
                self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, tb2)
            except Exception:
                self.addToolBar(tb2)

            act = QAction("打开 JSON", self)
            act.triggered.connect(self.on_open)
            tb.addAction(act)
            act = QAction("打开 PDF（无 JSON，直接标注）", self)
            act.triggered.connect(self.on_open_pdf)
            tb.addAction(act)
            self.act_open_folder = QAction("打开补标文件夹…", self)
            self.act_open_folder.setToolTip(
                "打开「补标建议」这种包：一个文件夹里每个 json 配一张同名图片。\n"
                "底图直接用那张图片，审完的框就地写回同一个 json —— 不碰你的原始标注。")
            self.act_open_folder.triggered.connect(self.on_open_folder)
            tb.addAction(self.act_open_folder)
            tb.addSeparator()
            tb2.addSeparator()
            mode_tip = {"select": ("选择：点/框选，拖框内=移动，拖白点=改大小", "1"),
                        "Node": ("画 Node（阵列块）：点两个对角，或按住拖一个框", "2"),
                        "Tracker": ("画 Tracker（支架/板列）：点两个对角，或按住拖一个框", "3"),
                        "Box": ("画 Box（其它）：点两个对角，或按住拖一个框", "4")}
            for m in ("select",) + CLASSES:
                a = QAction(MODE_TEXT[m], self)
                a.setCheckable(True)
                tip, key = mode_tip[m]
                a.setToolTip("%s\n快捷键 %s；Esc 回到选择模式" % (tip, key))
                a.triggered.connect(lambda _c=False, mm=m: self.set_mode(mm))
                tb2.addAction(a)
                setattr(self, "act_" + m, a)
            tb2.addSeparator()
            self.act_lockbox = QAction("锁定/解锁", self)
            self.act_lockbox.setCheckable(True)
            self.act_lockbox.setShortcut(QKeySequence("Ctrl+L"))
            self.act_lockbox.setToolTip(
                "把选中的框锁上：点不中、框选框不到、拖不动、Delete 也删不掉，\n"
                "「清理多余框」不会删它。\n"
                "注意：「补全编号 / 自动补编号 / OCR」**照样会**给锁定的 LBD 框写编号\n"
                "（锁是防误拖误删的，不是不让补号）。（Ctrl+L）\n"
                "锁定状态存进 JSON，下次打开还在；再点一次＝解锁。\n"
                "单独解锁某一个：Ctrl+点那个框（或用「解锁本页」）。")
            self.act_lockbox.triggered.connect(self.on_toggle_lock)
            tb2.addAction(self.act_lockbox)
            self.act_show_conf = QAction("显示置信度", self)
            self.act_show_conf.setCheckable(True)
            self.act_show_conf.setChecked(bool(self.settings.get("show_conf", True)))
            BoxItem.show_conf = self.act_show_conf.isChecked()
            self.act_show_conf.setToolTip(
                "补标/识别建议框上标出置信度数字。\n"
                "建议框统一是橙色虚线（一眼就能和原有标注区分），强弱看框上的数字：\n"
                "≥0.7 基本可信、0.5~0.7 看一眼、<0.5 重点核。\n"
                "框在屏幕上太小时不画数字（放大就会出来），选中的框一定画。")
            self.act_show_conf.toggled.connect(self.on_toggle_show_conf)
            tb2.addAction(self.act_show_conf)
            self.act_lock_nodes = QAction("锁定本页 Node", self)
            self.act_lock_nodes.setCheckable(True)
            self.act_lock_nodes.setShortcut(QKeySequence("Ctrl+Shift+L"))
            self.act_lock_nodes.setToolTip(
                "一键把本页所有 Node（阵列块）锁上：点不中、框选框不到、拖不动、\n"
                "Delete 删不掉，「清理多余框」不会删它们。\n"
                "补编号 / OCR 照样会给它们写编号（锁只防误拖误删）。（Ctrl+Shift+L）\n"
                "本页 Node 全锁着的时候，再点一次＝全部解锁。")
            self.act_lock_nodes.triggered.connect(self.on_lock_page_nodes)
            tb2.addAction(self.act_lock_nodes)
            a = QAction("解锁本页", self)
            a.setToolTip("把本页锁上的框一次全解锁（Node / Tracker / Box 全算）")
            a.triggered.connect(self.on_unlock_page)
            tb2.addAction(a)
            tb2.addSeparator()
            a = QAction("撤销", self)
            a.setShortcut(QKeySequence("Ctrl+Z"))
            a.triggered.connect(self.on_undo)
            tb2.addAction(a)
            a = QAction("重做", self)
            a.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
            a.setToolTip("重做刚撤销的操作（Ctrl+Y 或 Ctrl+Shift+Z）")
            a.triggered.connect(self.on_redo)
            tb2.addAction(a)
            a = QAction("支架按长度分档", self)
            a.setToolTip("Tracker（支架）框按长边长度分档：短的算 2 串、长的算 3 串（可改），"
                         "结果写进 JSON 的 raw.strings")
            a.triggered.connect(self.on_rack_grade)
            tb2.addAction(a)
            a = QAction("清理多余框(本页)", self)
            a.setToolTip("删掉多余的框：已经有手工框的地方、同一个编号重复的、"
                         "互相重叠 90% 以上的、形状离谱的 Tracker。\n"
                         "只删模型画的框，手工框一个都不动；锁定的框也不动；"
                         "可用 Ctrl+Z 撤销")
            a.triggered.connect(self.on_clean_shapes)
            tb2.addAction(a)
            a = QAction("清理多余框(整册)", self)
            a.setToolTip("把整册所有页一起清一遍（同样的规则、同样不动锁定框和手工框）。\n"
                         "整册算一步：Ctrl+Z 一次就能把所有页一起撤回。")
            a.triggered.connect(self.on_clean_shapes_all)
            tb2.addAction(a)
            a = QAction("重载本页", self)
            a.setToolTip("把当前页恢复成上次打开/保存时的样子（画乱了的出口；可用 Ctrl+Z 撤销）")
            a.triggered.connect(self.on_reload_page)
            tb.addAction(a)
            # 复制框 / 粘贴框 / 放大 / 缩小 不再占工具栏（Ctrl+C、Ctrl+V、Ctrl+=、Ctrl+-、
            # 滚轮就够）。动作本身留着挂在窗口上，快捷键才有效。
            _zi = QAction("放大", self)
            _zi.setShortcut(QKeySequence("Ctrl+="))
            _zi.setToolTip("放大（Ctrl+= 或滚轮）")
            _zi.triggered.connect(lambda: self.zoom_by(1.18))
            self.addAction(_zi)
            _zo = QAction("缩小", self)
            _zo.setShortcut(QKeySequence("Ctrl+-"))
            _zo.setToolTip("缩小（Ctrl+- 或滚轮）")
            _zo.triggered.connect(lambda: self.zoom_by(1 / 1.18))
            self.addAction(_zo)
            a = QAction("适应窗口", self)
            a.setShortcut(QKeySequence("Ctrl+0"))
            a.setToolTip("整页缩放到刚好铺满窗口（Ctrl+0）")
            a.triggered.connect(self.fit)
            tb.addAction(a)
            a = QAction("100%", self)
            a.setToolTip("按原始像素 1:1 显示")
            a.triggered.connect(self.zoom_reset)
            tb.addAction(a)
            tb.addSeparator()
            tb.addSeparator()
            self.act_save_as = QAction("另存为…", self)
            self.act_save_as.setShortcut(QKeySequence("Ctrl+S"))
            self.act_save_as.setToolTip("另存一份 JSON（Ctrl+S）")
            self.act_save_as.triggered.connect(self.on_save_as)
            tb.addAction(self.act_save_as)
            self.act_save_over = QAction("覆盖原文件", self)
            self.act_save_over.setToolTip("直接覆盖原文件（不留备份）")
            self.act_save_over.triggered.connect(self.on_save_over)
            tb.addAction(self.act_save_over)

            tb.addSeparator()
            a = QAction("选标签表…", self)
            a.setToolTip("选 LBD 标签表(xlsx)：分表名、LBD 编号、要填的名字都从这张表来")
            a.triggered.connect(self.on_pick_xlsx)
            tb.addAction(a)
            a = QAction("导出核对表…", self)
            a.setToolTip("每个框一行：现有名字 / 框内找到的候选 / 建议名字 —— 导出 CSV 在 Excel 里核")
            a.triggered.connect(self.on_export_check)
            tb.addAction(a)
            # 用得不多的两个收进「更多…」，工具栏只留日常那几个
            from PySide6.QtWidgets import QMenu, QToolButton
            btn_more = QToolButton(self)
            btn_more.setText("更多…")
            btn_more.setToolTip("不常用的功能：导出 YOLO 数据集、训练环境")
            btn_more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            menu_more = QMenu(btn_more)
            a = menu_more.addAction("导出 YOLO 数据集…")
            a.setToolTip("导出成 YOLO 训练集（images/ + labels/ + data.yaml），可以丢给 yolo26 训练")
            a.triggered.connect(self.on_export_dataset)
            a = menu_more.addAction("训练环境…")
            a.setToolTip("检测 Python / 一键装 ultralytics（CPU 版）+ 一键训练")
            a.triggered.connect(self.on_train_panel)
            a = menu_more.addAction("更新诊断…")
            a.setToolTip("更新检查卡住/失败时点这个：DNS、curl、git、安装包地址、api 各测一遍，\n"
                         "把结果发我就能看出是哪一层被防火墙/杀毒软件挡了")
            a.triggered.connect(self.on_update_diagnose)
            btn_more.setMenu(menu_more)
            tb.addWidget(btn_more)
            a = QAction("检查更新", self)
            a.setToolTip("看 Release 里有没有新的标注工具包（名字里带 LBD 的那个）")
            a.triggered.connect(self.on_check_update)
            tb.addAction(a)
            a = QAction("用模型识别…", self)
            a.setToolTip("用训练好的 YOLO 模型（best.pt）识别图纸，本页/整册自动把框画上来")
            a.triggered.connect(self.on_ai_detect)
            tb.addAction(a)
            a = QAction("识别全册…", self)
            a.setToolTip("同上，但范围默认选好「整册」：整本图纸逐页跑模型，自动把框画上来")
            a.triggered.connect(lambda _c=False: self.on_ai_detect(True))
            tb.addAction(a)
            a = QAction("自动补编号(本页)", self)
            a.setToolTip("一键补编号：先用 PDF 文字层读（1~2 秒一页、字是原文不会认错），\n"
                         "只有文字层没给到号的框才自动去 OCR。\n"
                         "只写 LBD 区域（Node）框；支架框不碰。")
            a.triggered.connect(lambda: self.on_auto_name(False))
            tb.addAction(a)
            a = QAction("自动补编号(整册)", self)
            a.setToolTip("同上，整册逐页跑：有文字层的图纸基本不用装 OCR 组件。\n"
                         "整册算一步，Ctrl+Z 一次全撤回。")
            a.triggered.connect(lambda: self.on_auto_name(True))
            tb.addAction(a)
            # 两条单独的路线收进这个下拉里，工具栏就不会一排按钮挤着。
            # 平时点「自动补编号」就够了；要只走文字层 / 强制 OCR 时再来这儿。
            from PySide6.QtWidgets import (QCheckBox, QMenu, QToolButton,
                                           QWidgetAction)
            btn_more = QToolButton(self)
            # 名字里别用 '▾' 这类字符：控制台按 GBK 打印时会 UnicodeEncodeError
            # （工具栏上其它下拉都是"……"结尾，跟它们保持一致）
            btn_more.setText("补编号选项…")
            btn_more.setToolTip(
                "单独跑某一条路时用这里：\n"
                "· 补全编号 = 只读 PDF 文字层（快，不用装任何东西，号码可跟标签表核对）\n"
                "· OCR 读框内文字 = 强制用 OCR（图纸没有文字层、或文字层里的编号不对时用）\n\n"
                "平时直接点「自动补编号」就行：它先走文字层，没拿到号的框再自动 OCR。")
            btn_more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            menu_more = QMenu(btn_more)
            a = menu_more.addAction("补全编号（只走文字层）· 本页")
            a.setToolTip("从框内文字取 LBD 编号补名字（只跑当前页），不跑 OCR")
            a.triggered.connect(lambda _c=False: self.on_autofill(False))
            a = menu_more.addAction("补全编号（只走文字层）· 整册")
            a.setToolTip("整册逐页补：框内文字取号；取不到的按标签表顺序补（标黄）；"
                         "再取不到标红等人手填。\n"
                         "没选标签表也能跑：那就只用框内文字取号，号码不跟表核对")
            a.triggered.connect(lambda _c=False: self.on_autofill(True))
            menu_more.addSeparator()
            a = menu_more.addAction("OCR 读框内文字（强制 OCR）· 本页")
            a.setToolTip("用 OCR 读每个 LBD 框（Node）里的文字 —— 图纸里的字被转成矢量轮廓、\n"
                         "没有文字层、或文字层里的编号不对时用这个；竖排标签也能读。\n"
                         "支架框（Tracker）不读、不写名字；需要 rapidocr-onnxruntime"
                         "（没装会问你要不要装）")
            a.triggered.connect(lambda _c=False: self.on_ocr_read(False))
            a = menu_more.addAction("OCR 读框内文字（强制 OCR）· 整册")
            a.setToolTip("同上，整册逐页跑（有进度提示，可直接 Ctrl+Z 撤销）；同样只写 Node 框")
            a.triggered.connect(lambda _c=False: self.on_ocr_read(True))
            menu_more.addSeparator()
            self.chk_force = QCheckBox("重算(覆盖已有名字)")
            self.chk_force.setToolTip(
                "勾上 = 已有名字也清掉、重新识别一遍重新填（位置也会重算）。\n"
                "· 重算后没填回来的框，原来的名字会自动还回去，不会把好名字弄丢\n"
                "· 不勾 → 名字不动，只把缺的编号位置补上\n"
                "「补全编号」和「自动补编号」都认这个勾。")
            self.chk_force.setChecked(bool(self.settings.get("force")))
            wa = QWidgetAction(menu_more)
            wa.setDefaultWidget(self.chk_force)
            menu_more.addAction(wa)
            btn_more.setMenu(menu_more)
            tb.addWidget(btn_more)
            # 把"识别到的编号框"画在图上（核对用）
            self.act_labelboxes = QAction("编号框", self)
            self.act_labelboxes.setCheckable(True)
            self.show_label_boxes = bool(self.settings.get("label_boxes", True))
            self.act_labelboxes.setChecked(self.show_label_boxes)
            self.act_labelboxes.setToolTip(
                "在图上把编号的位置画出来（橙色小方框 + 红色小十字）：\n"
                "· 方框 = 编号文字的区域（文字层给的是估算框，OCR 给的是实测框）\n"
                "· 十字 = 编号中心点\n"
                "数据来自每个 Node 的 raw.label_bbox / raw.label_pos，"
                "补过编号的框才有；只画当前页。")

            def _toggle_label_boxes(checked):
                self.show_label_boxes = bool(checked)
                self.settings["label_boxes"] = bool(checked)
                save_settings(self.settings)
                try:
                    self.canvas.viewport().update()
                except Exception:
                    pass

            self.act_labelboxes.toggled.connect(_toggle_label_boxes)
            tb.addAction(self.act_labelboxes)
            tb.addWidget(QLabel("  名字 "))
            self.cmb_name = QComboBox()
            # (显示方式, 字号, 只画选中的那个)
            for text, px, selonly in (("不显示", 0, True),
                                      ("只看选中的", 11, True),
                                      ("全部·小", 9, False),
                                      ("全部·中", 11, False),
                                      ("全部·大", 14, False)):
                self.cmb_name.addItem(text, (px, selonly))
            i = int(self.settings.get("name_mode") or 1)
            self.cmb_name.setCurrentIndex(i if 0 <= i < self.cmb_name.count() else 1)
            self.cmb_name.currentIndexChanged.connect(self.on_name_px)
            tb.addWidget(self.cmb_name)
            self.act_lock = QAction("锁定窗口大小", self)
            self.act_lock.setCheckable(True)
            self.act_lock.setToolTip("勾上以后窗口就不能再拉大拉小了（再点一下解锁）")
            self.act_lock.triggered.connect(self.on_lock_window)
            tb.addAction(self.act_lock)

            tb.addSeparator()
            tb.addWidget(QLabel("  底图 "))
            self.cmb_dpi = QComboBox()
            for text, val in (("内嵌图（快）", 0), ("250 DPI", 250),
                              ("333 DPI", 333), ("400 DPI", 400)):
                self.cmb_dpi.addItem(text, val)
            i = self.cmb_dpi.findData(self.dpi)
            self.cmb_dpi.setCurrentIndex(i if i >= 0 else 0)
            self.cmb_dpi.currentIndexChanged.connect(self.on_dpi_changed)
            tb.addWidget(self.cmb_dpi)
            self.btn_pdf = QPushButton("选 PDF…")
            self.btn_pdf.clicked.connect(self.on_pick_pdf)
            tb.addWidget(self.btn_pdf)
            self.update_pdf_button()

            side = QWidget()
            sl = QVBoxLayout(side)
            self.lbl_info = QLabel("用「打开 JSON」载入识别结果")
            self.lbl_info.setWordWrap(True)
            sl.addWidget(self.lbl_info)
            nav = QHBoxLayout()
            b = QPushButton("◀ 上一页")
            b.clicked.connect(lambda: self.goto_offset(-1))
            nav.addWidget(b)
            self.cmb_page = QComboBox()
            self.cmb_page.currentIndexChanged.connect(self.on_page_combo)
            nav.addWidget(self.cmb_page, 1)
            b = QPushButton("下一页 ▶")
            b.clicked.connect(lambda: self.goto_offset(1))
            nav.addWidget(b)
            self.btn_start = QPushButton("起始页")
            self.btn_start.setToolTip(
                "把**当前页**记成这份 JSON 的起始页：以后再打开这个文件，直接停在这一页。\n"
                "再点一次＝取消（回到默认：停在第 1 个有框的页）。\n"
                "按文件记（存在本机设置里，不动 JSON）。")
            self.btn_start.clicked.connect(self.on_toggle_start_page)
            nav.addWidget(self.btn_start)
            sl.addLayout(nav)
            sl.addWidget(QLabel("———— 选中的框 ————"))
            form = QFormLayout()
            self.cmb_label = QComboBox()
            self.cmb_label.addItems(list(CLASSES))
            self.cmb_label.currentIndexChanged.connect(self.on_label_changed)
            form.addRow("类别", self.cmb_label)
            self.ed_name = QLineEdit()
            self.ed_name.editingFinished.connect(self.on_name_changed)
            form.addRow("LBD 名字", self.ed_name)
            self.lbl_bbox = QLabel("-")
            form.addRow("位置", self.lbl_bbox)
            self.chk_lock = QCheckBox("锁定这个框")
            self.chk_lock.setToolTip(
                "锁上以后这个框点不中、框选也框不到、拖不动、Delete 删不掉；\n"
                "跟工具栏的「锁定/解锁」(Ctrl+L) 是同一件事。锁定状态会存进 JSON。")
            self.chk_lock.toggled.connect(self.on_lock_check)
            form.addRow("", self.chk_lock)
            sl.addLayout(form)
            self.btn_del = QPushButton("删除这个框（Delete）")
            self.btn_del.clicked.connect(self.on_delete)
            sl.addWidget(self.btn_del)
            sl.addWidget(QLabel("———— 本页统计 ————"))
            self.lbl_stats = QLabel("-")
            self.lbl_stats.setWordWrap(True)
            self.lbl_stats.setStyleSheet("color:#222;")
            sl.addWidget(self.lbl_stats)
            self.lbl_racks = QLabel("")
            self.lbl_racks.setWordWrap(True)
            self.lbl_racks.setTextFormat(Qt.TextFormat.PlainText)
            self.lbl_racks.setStyleSheet("color:#444; font-size:11px;")
            sl.addWidget(self.lbl_racks)
            self.lbl_boxtext = QLabel("")
            self.lbl_boxtext.setWordWrap(True)
            self.lbl_boxtext.setTextFormat(Qt.TextFormat.PlainText)
            self.lbl_boxtext.setStyleSheet("color:#064; font-size:11px;")
            self.lbl_boxtext.setToolTip("选中一个框，这里列出 PDF 文字层里落在它里面"
                                        "（含略微超出边缘）的全部文字，按从上到下、从左到右排。")
            sl.addWidget(self.lbl_boxtext)
            sl.addStretch(1)
            side.setMaximumWidth(330)

            central = QWidget()
            cl = QHBoxLayout(central)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.addWidget(self.canvas, 1)
            cl.addWidget(side)
            self.setCentralWidget(central)

            self.setStatusBar(QStatusBar())
            # 右下角常驻一行：模式 / 页码 / 缩放 / 各类框数量 / 有没有改过
            self.lbl_state = QLabel()
            self.lbl_state.setMinimumWidth(420)
            self.lbl_state.setAlignment(Qt.AlignmentFlag.AlignRight
                                        | Qt.AlignmentFlag.AlignVCenter)
            self.statusBar().addPermanentWidget(self.lbl_state)
            self.scene.selectionChanged.connect(self.on_selection)
            self.set_mode("select")
            self.on_name_px()
            if self.settings.get("win_locked"):
                self.act_lock.setChecked(True)
                self.on_lock_window(True)
            self.refresh_info()
            if src:
                self.load_file(src)

        def _make_scene(self):
            from PySide6.QtWidgets import QGraphicsScene
            return QGraphicsScene(self)

        # ---------------- 载入与翻页
        def on_open(self):
            p, _ = QFileDialog.getOpenFileName(self, "选择识别结果 JSON", "",
                                               "JSON (*.json)")
            if p:
                self.load_file(p)

        def update_start_button(self):
            """「起始页」按钮的状态：当前页正好是记过的起始页就打个勾。"""
            try:
                sp = 0
                if self.dbg:
                    sp = json_start_page(self.dbg.path) or start_page_of(self.dbg.path)
                cur = self.page if self.dbg else 0
                if sp and sp == cur:
                    self.btn_start.setText("起始页 ✓")
                    self.btn_start.setToolTip("当前页就是这份 JSON 的起始页（第 %d 页）。\n"
                                              "点一下＝取消（把这个字段从 JSON 里去掉）。" % sp)
                elif sp:
                    self.btn_start.setText("起始页")
                    self.btn_start.setToolTip(
                        "这份 JSON 的起始页是第 %d 页。\n"
                        "点一下＝把当前页（第 %s 页）改成起始页。" % (sp, cur))
                else:
                    self.btn_start.setText("起始页")
                    self.btn_start.setToolTip(
                        "把当前页写成这份 JSON 的起始页（顶层字段 lbd_start_page）：\n"
                        "以后一打开就直接停在这一页，换机器、发给别人也生效。\n"
                        "再点一次＝取消。")
            except Exception:
                pass

        def on_toggle_start_page(self):
            if not self.dbg:
                return
            cur = json_start_page(self.dbg.path) or start_page_of(self.dbg.path)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                if cur == self.page:
                    write_json_start_page(self.dbg.path, 0)
                    set_start_page(self.dbg.path, 0)     # 老的本机记录也一起清掉
                    msg = "已取消起始页：这个 JSON 不再指定打开页"
                else:
                    n = write_json_start_page(self.dbg.path, self.page)
                    set_start_page(self.dbg.path, 0)
                    # 重新读一遍文件头：DebugJson 里那些按字节记的偏移不能乱
                    self.dbg = DebugJson(self.dbg.path)
                    msg = ("已把第 %d 页写进 JSON（lbd_start_page）；"
                           "以后打开这份文件直接停这儿" % n)
            except Exception as e:
                msg = "写不进去：%s" % e
            finally:
                QApplication.restoreOverrideCursor()
            self.statusBar().showMessage(msg, 8000)
            if self.dbg:
                self.goto_page(self.page)      # 重新按新文件读这一页（旧偏移作废了）
            self.update_start_button()

        def on_open_pdf(self):
            """没有识别结果时，直接打开一份 PDF 从零标注。"""
            p, _ = QFileDialog.getOpenFileName(self, "选择要直接标注的 PDF", "",
                                               "PDF (*.pdf)")
            if not p:
                return
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                doc = BlankDoc(p, dpi=float(self.dpi or 0) or 250.0)
            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(self, "读不了这份 PDF", "%s" % e)
                return
            QApplication.restoreOverrideCursor()
            if not doc.page_numbers():
                QMessageBox.warning(
                    self, "读不出页数",
                    "没能从这份 PDF 读出页码/页尺寸。\n\n"
                    "这条路径要用 poppler 的 pdfinfo.exe（和渲染底图用的 pdftoppm 在一起）。")
                return
            self.dbg = doc
            self.pdf = p
            self._ptext = None
            self.update_pdf_button()
            self.edited.clear()
            self.undo.clear()
            self.redo.clear()
            self.cmb_page.blockSignals(True)
            self.cmb_page.clear()
            for n in doc.page_numbers():
                self.cmb_page.addItem(str(n), n)
            self.cmb_page.blockSignals(False)
            self.setWindowTitle("LBD 标注工具 v%s — %s（直接标注，无识别结果）"
                                % (ANNOTATOR_VERSION, os.path.basename(p)))
            self.statusBar().showMessage("空白文档：%d 页，框从零开始画；保存会生成识别结果 JSON"
                                         % len(doc.page_numbers()), 8000)
            self.goto_page(doc.page_numbers()[0])

        def load_file(self, path):
            if os.path.isdir(path):
                self.load_folder(path)       # 传进来的是文件夹 -> 按「补标文件夹」打开
                return
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.dbg = DebugJson(path)
            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(self, "读不了", "%s" % e)
                return
            QApplication.restoreOverrideCursor()
            self._use_folder_mode(False)
            self.undo.clear()
            # 不再记住这份 JSON：路径不落盘，下次双击就是空白画布
            # （要接着上次那份，自己点「打开 JSON」再选一次）
            pgs = self.dbg.page_numbers()
            if not pgs:
                raw = getattr(self.dbg, "raw", b"") or b""
                marks = []
                for key, desc in ((b'"input_data"', "input_data（但没有 pages）"),
                                  (b'"pages"', "pages 字段"),
                                  (b'"yolo_tracker_detection_results"', "yolo_tracker_detection_results（识别结果的框）"),
                                  (b'"ocr_node_name_results"', "ocr_node_name_results（编号名字）"),
                                  (b'"shapes"', "shapes（X-AnyLabeling / labelme 标注文件）"),
                                  (b'"imagePath"', "imagePath（标注文件里的图片名）"),
                                  (b'"lbd_regions"', "lbd_regions（区域范围导出）"),
                                  (b'"config"', "config（配置文件）")):
                    if key in raw:
                        marks.append(desc)
                QMessageBox.warning(
                    self, "这份 JSON 里没有页面",
                    "没找到 input_data.pages（每页的页号和尺寸）——"
                    "标注工具靠它把标注定位到页面上。\n\n"
                    "这份文件里看到：\n  %s\n\n"
                    "要标注请打开完整的「识别结果」JSON（agent3-debug 那份，"
                    "input_data.pages 里带每页宽高）。" % ("\n  ".join(marks) or "（没有认出来的内容）"))
                return
            # PDF 对不上就纠正：文件名里没有 project_name 的，说明是别的项目的图纸
            if self.pdf and not os.path.exists(self.pdf):
                self.pdf = ""
            pname = project_name(path)
            src_name = source_pdf_name(path)          # 这份 JSON 是哪份 PDF 做出来的
            want = src_name or (pname + ".pdf" if pname else "")
            pdf_note = ""

            def _same_pdf(a, b):
                a, b = os.path.basename(str(a or "")).lower(), os.path.basename(str(b or "")).lower()
                if not a or not b:
                    return False
                sa, sb = os.path.splitext(a)[0], os.path.splitext(b)[0]
                return a == b or sa == sb or sa in sb or sb in sa

            cur = os.path.basename(self.pdf) if self.pdf else ""
            if want and self.pdf and _same_pdf(want, cur):
                pass                                    # 就是这份，没毛病
            else:
                exact = find_pdf_by_name(want, path) if want else ""
                fuzzy = "" if exact else (find_pdf_by_name(want, path, fuzzy=True)
                                          if want else "")
                if exact:
                    self.pdf = exact
                    if cur and not _same_pdf(exact, cur):
                        pdf_note = ("已按 JSON 里写的源文件名换成：\n  %s\n"
                                    % os.path.basename(exact))
                elif fuzzy:
                    # 名字像不代表是同一份：先看页数，对不上直接不用；
                    # 就算页数够，也**问一句**再用 —— 底图拿错比没底图更坑
                    sizes = pdf_page_sizes(fuzzy)
                    need = max(self.dbg.page_numbers()) if self.dbg.page_numbers() else 0
                    if sizes and need and len(sizes) < need:
                        self.pdf = ""
                        pdf_note = ("本机找到一份名字像的图纸：\n  %s\n"
                                    "但它只有 %d 页，这份 JSON 有 %d 页 —— 不是同一份，"
                                    "所以没拿它当底图。\n点「选 PDF…」选上正确的那份"
                                    "（JSON 里写的是：%s）。"
                                    % (os.path.basename(fuzzy), len(sizes), need, want))
                    else:
                        ask = ("本机找不到 JSON 里写的那份 PDF：\n  %s\n\n"
                               "找到一份名字像的：\n  %s\n（%d 页，这份 JSON 有 %d 页）\n\n"
                               "要用它当底图吗？如果不是同一份图纸，框和底图会对不上。"
                               % (want, os.path.basename(fuzzy),
                                  len(sizes) if sizes else 0, need))
                        use = False
                        if TEST_MODE:
                            print("底图询问（自检自动选否）：", ask.replace("\n", " "))
                        else:
                            use = (QMessageBox.question(
                                self, "这份 PDF 是同一份图纸吗？", ask,
                                QMessageBox.StandardButton.Yes
                                | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No)
                                == QMessageBox.StandardButton.Yes)
                        if use:
                            self.pdf = fuzzy
                            pdf_note = "已用你确认的那份 PDF 当底图：\n  %s\n" % os.path.basename(fuzzy)
                        else:
                            self.pdf = ""
                            pdf_note = ("这份 JSON 对应的是：\n  %s\n本机没找到它，"
                                        "所以这次先不给底图（免得框和底图对不上）。\n"
                                        "点「选 PDF…」选上正确的 PDF，选一次会记住。"
                                        % want)
                else:
                    if self.pdf:
                        pdf_note = ("这份 JSON 对应的是：\n  %s\n工具里上次选的是：\n  %s\n"
                                    "两者不是一份图纸，本机也没找到那份 PDF，所以这次先不给底图"
                                    "（免得框和底图对不上）。\n点「选 PDF…」选上正确的 PDF，选一次会记住。"
                                    % (want or "（JSON 里没写）", cur))
                    self.pdf = ""
            if pdf_note and not TEST_MODE:
                QMessageBox.warning(self, "底图对不上", pdf_note)
            elif pdf_note:
                print("底图警告：", pdf_note.replace("\n", " "))
            self._ptext = None
            if self.pdf:
                self.update_pdf_button()
            self._after_load(path)

        def load_folder(self, folder):
            """打开「补标文件夹」：每个 json 配一张同名图片，审完写回同一个 json。"""
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                doc = SuggestFolderDoc(folder)
            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(self, "打不开这个文件夹", "%s" % e)
                return
            QApplication.restoreOverrideCursor()
            self.dbg = doc
            self.pdf = ""                    # 底图直接用文件夹里的 png，不用 PDF
            self._ptext = None
            self.undo.clear()
            self.edited.clear()
            self._use_folder_mode(True)
            n_pg, n_box = doc.suggest_stats()
            miss = doc.missing_json()
            if not TEST_MODE:
                tip = ("共 %d 页 / %d 个建议框（类别名带问号的就是模型建议）。\n\n"
                       "· 对的框留着，错的框选中按 Delete 删掉，也可以拖框调位置\n"
                       "· 问号不用管，合并回原始标注时会自动去掉\n"
                       "· 保存写回这个文件夹里的 json，桌面上的原始标注不会被碰"
                       % (n_pg, n_box))
                if miss:
                    tip += "\n\n（有 %d 个 json 没找到同名图片，已跳过）" % len(miss)
                QMessageBox.information(self, "补标文件夹已打开", tip)
            self._after_load(doc.path)

        def _after_load(self, path):
            """打开文档后的公共收尾：页码下拉、标题、起始页。"""
            self.cmb_page.blockSignals(True)
            self.cmb_page.clear()
            for n in self.dbg.page_numbers():
                self.cmb_page.addItem(str(n), n)
            self.cmb_page.blockSignals(False)
            self.setWindowTitle("LBD 标注工具 v%s — %s"
                                % (ANNOTATOR_VERSION, os.path.basename(path)))
            if getattr(self.dbg, "is_folder", False):
                pgs = self.dbg.page_numbers()
                # 有框的页优先：整册 JSON（每页都写了空记录的那份）本来会停在第 1 页，
                # 明明有图纸却是一片空白，得自己翻半天。
                drw = [n for n in pgs
                       if (self.dbg.page_data(TRACKER_SECTION, n) or {}).get("detections")]
            else:
                pgs = self.dbg.page_numbers()
                drw = drawing_pages(self.dbg)
            # 优先用 JSON 自己带的起始页（跟着文件走），其次本机设置里记的
            want = json_start_page(path) or start_page_of(path)
            if want and want in self.dbg.pages:
                start = want                     # 这份文件记过起始页 -> 就停在那儿
            else:
                start = drw[0] if drw else pgs[0]
            self.goto_page(start)
            self.update_start_button()

        def _use_folder_mode(self, on):
            """补标文件夹模式：PDF 按钮/保存按钮的语义跟着换。"""
            self._folder_mode = bool(on)
            try:
                self.btn_pdf.setEnabled(not on)
                if on:
                    self.btn_pdf.setText("补标文件夹")
                    self.btn_pdf.setToolTip("这套底图直接来自文件夹里的 png，不需要 PDF")
                else:
                    self.update_pdf_button()
            except Exception:
                pass
            for act in (getattr(self, "act_save_as", None),
                        getattr(self, "act_save_over", None)):
                if act is None:
                    continue
                if act is getattr(self, "act_save_as", None):
                    act.setText("保存(写回文件夹)" if on else "另存为…")
                    act.setToolTip("补标文件夹是就地保存：把改过的页写回它自己的 json"
                                   if on else "另存一份 JSON")
                else:
                    act.setVisible(not on)

        def goto_page(self, number):
            if not self.dbg or number not in self.dbg.pages:
                return
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self._stash_dirty()
                # 这一页改过就还用内存里那份，别重新从文件读（否则改动会没）
                self.pm = self.edited.get(number) or PageModel(self.dbg, number)
                self._img_note = ""
                img = None
                try:
                    img, self._img_note = self.load_image(number)
                except Exception as e:
                    self._img_note = "底图没出来：%s" % e
                if img is None or img.isNull():
                    img = QImage(600, 400, QImage.Format.Format_RGB32)
                    img.fill(0x303030)
                self._pixmap = QPixmap.fromImage(img)
            finally:
                QApplication.restoreOverrideCursor()
            self.page = number
            self._rebuild_items()
            i = self.cmb_page.findData(number)
            if i >= 0:
                self.cmb_page.blockSignals(True)
                self.cmb_page.setCurrentIndex(i)
                self.cmb_page.blockSignals(False)
            self.fit()
            self.refresh_info()
            self.on_selection()
            self.prefetch_next()
            try:
                self.update_start_button()      # 翻页后刷新「起始页」按钮的勾
            except Exception:
                pass

        def prefetch_next(self):
            """后台把下一页渲染好，翻页就不用等。"""
            if self.dpi <= 0 or not self.pdf or not self.poppler:
                return
            pgs = self.dbg.page_numbers()
            i = pgs.index(self.page) + 1
            if i >= len(pgs):
                return
            page, dpi, pdf = pgs[i], self.dpi, self.pdf
            if os.path.exists(self.renderer.target(pdf, page, dpi)):
                return

            def work():
                try:
                    self.renderer.render(pdf, page, dpi)
                except Exception:
                    pass

            threading.Thread(target=work, daemon=True).start()

        def _stash_dirty(self):
            """离开一页前，把改过的页留下来，保存时一起写。"""
            if self.pm is not None and self.pm.dirty:
                self.edited[self.pm.page] = self.pm

        def _rebuild_items(self):
            self.scene.clear()
            self.items = []
            self.scene.setSceneRect(0, 0, self.pm.width, self.pm.height)
            item = self.scene.addPixmap(self._pixmap)
            # 场景坐标始终用 JSON 里的原始坐标（6000×4000）；
            # 高分辨率底图按比例缩回来放，放大看时 Qt 会用它的原始像素。
            item.setScale(self.pm.width / float(self._pixmap.width() or self.pm.width))
            item.setZValue(0)
            for s in self.pm.shapes:
                it = BoxItem(s)
                self.scene.addItem(it)
                self.items.append(it)

        def load_image(self, page):
            """返回 (QImage, 说明文字)。dpi=0 用 JSON 内嵌图，否则重渲染 PDF。"""
            info = self.dbg.pages.get(page) or {}
            if getattr(self.dbg, "is_folder", False):
                p = self.dbg.png_path(page)
                img = QImage(p)
                if img.isNull():
                    raise RuntimeError("读不了底图：%s" % p)
                if img.width() > 6000:        # 9000×6000 的原图整张进内存太重，缩一半看
                    img = img.scaled(6000, 6000, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
                    return img, ("补标底图 %s（原图 %d×%d，显示已缩到 %d 宽）"
                                 % (os.path.basename(p), info.get("width") or 0,
                                    info.get("height") or 0, img.width()))
                return img, "补标底图 %s（%d×%d）" % (os.path.basename(p),
                                                     img.width(), img.height())
            if self.dpi <= 0 and info.get("png_span"):
                img = QImage.fromData(self.dbg.png_bytes(page))
                return img, "内嵌图 %d×%d（约 167 dpi）" % (img.width(), img.height())
            if not self.poppler:
                raise RuntimeError("找不到 pdftoppm/poppler")
            if not self.pdf or not os.path.exists(self.pdf):
                raise RuntimeError("这份 JSON 没有内嵌图，需要点「选 PDF…」指定对应的 PDF")
            dpi = self.dpi if self.dpi > 0 else 250      # 没内嵌图时用 250 DPI 渲染
            t0 = time.time()
            self.statusBar().showMessage(
                "正在用 %s 重渲染第 %d 页（%d DPI）…"
                % (os.path.basename(self.pdf), page, dpi))
            QApplication.processEvents()
            path = self.renderer.render(self.pdf, page, dpi)
            img = QImage(path)
            if img.isNull():
                raise RuntimeError("渲染结果读不出来")
            k = img.width() / float(self.dbg.pages[page]["width"] or img.width())
            return img, "%d DPI 重渲染 %d×%d（%.2f 倍，%.1fs）" % (
                dpi, img.width(), img.height(), k, time.time() - t0)

        def on_dpi_changed(self, _i):
            want = int(self.cmb_dpi.currentData() or 0)
            if want >= 400 and self.dbg:
                mem = self.pm.width * 400 // 167 * self.pm.height * 400 // 167 * 4 / 1048576.0
                if QMessageBox.question(
                        self, "内存提醒",
                        "400 DPI 时每页底图约占 %.0f MB 内存，机器不够会卡甚至崩。\n"
                        "日常建议 250 或 333 DPI。要继续吗？" % mem
                ) != QMessageBox.StandardButton.Yes:
                    self.cmb_dpi.blockSignals(True)
                    i = self.cmb_dpi.findData(self.dpi)
                    self.cmb_dpi.setCurrentIndex(i if i >= 0 else 0)
                    self.cmb_dpi.blockSignals(False)
                    return
            self.dpi = want
            self.settings["dpi"] = self.dpi
            save_settings(self.settings)
            if self.dbg:
                self.goto_page(self.page)

        def on_pick_pdf(self):
            p, _ = QFileDialog.getOpenFileName(self, "选择这份识别结果对应的 PDF", "",
                                               "PDF (*.pdf)")
            if not p:
                return
            self.pdf = p
            self._ptext = None
            if self.dpi <= 0:
                # 底图还停在内嵌图的话，换 PDF 看不出任何变化，直接切到 250 DPI
                self.dpi = 250
                i = self.cmb_dpi.findData(250)
                if i >= 0:
                    self.cmb_dpi.blockSignals(True)
                    self.cmb_dpi.setCurrentIndex(i)
                    self.cmb_dpi.blockSignals(False)
            self.update_pdf_button()
            self.check_pdf_pages(p)           # 选的 PDF 页数明显对不上就提醒一句
            if self.dbg:
                self.goto_page(self.page)

        def check_pdf_pages(self, pdf):
            """选的 PDF 和这份 JSON 是不是同一份：先看名字，再看页数。

            底图不对（框和图纸错位）十次有九次是"拿了别项目的 PDF 当底图"，
            提醒比默默画错强。
            """
            if not (self.dbg and pdf):
                return
            try:
                want = source_pdf_name(self.dbg.path) or ""
                cur = os.path.basename(pdf)
                msg = ""
                if want:
                    a, b = os.path.splitext(want)[0].lower(), os.path.splitext(cur)[0].lower()
                    if not (a == b or a in b or b in a):
                        msg += ("这份 JSON 是「%s」做出来的，你现在选的是「%s」。\n" % (want, cur))
                need = max(self.dbg.page_numbers()) if self.dbg.page_numbers() else 0
                sizes = pdf_page_sizes(pdf)
                if sizes and need and len(sizes) < need:
                    msg += ("这份 PDF 只有 %d 页，但 JSON 里有 %d 页。\n" % (len(sizes), need))
                if msg:
                    QMessageBox.warning(
                        self, "这份 PDF 对吗？",
                        msg + "\n如果不是同一份图纸，底图和框会对不上；"
                              "确认要接着用就点「OK」。")
            except Exception:
                pass

        def update_pdf_button(self):
            if self.pdf:
                self.btn_pdf.setText("PDF：%s…" % os.path.basename(self.pdf)[:18])
            else:
                self.btn_pdf.setText("选 PDF…")
            self.btn_pdf.setToolTip(self.pdf or
                                    "选一份 PDF，再用 250/333/400 DPI 重渲染更清晰的底图")

        def goto_offset(self, d):
            if not self.dbg:
                return
            pgs = self.dbg.page_numbers()
            i = pgs.index(self.page) + d
            if 0 <= i < len(pgs):
                self.goto_page(pgs[i])

        def on_page_combo(self, _i):
            n = self.cmb_page.currentData()
            if n is not None and n != self.page:
                self.goto_page(n)

        def fit(self):
            if self.pm:
                self.canvas.fitInView(self.scene.sceneRect(),
                                      Qt.AspectRatioMode.KeepAspectRatio)
                self.update_state_label()

        def zoom_reset(self):
            self.canvas.resetTransform()
            self.update_state_label()

        # ---------------- 改动
        def begin_change(self):
            if self.pm:
                self._pending = [(i, dict(s)) for i, s in enumerate(self.pm.shapes)]

        def end_change(self):
            if not self.pm or self._pending is None:
                return
            now = [(i, dict(s)) for i, s in enumerate(self.pm.shapes)]
            if now != self._pending:
                # 一步撤销 = 一页或多页的快照列表（整册清理那种一次多页的操作算一步）
                self.undo.append([(self.page, self._pending)])
                del self.undo[:-200]       # 只留最近 200 步，别无限涨（一步快照很小）
                self.redo.clear()          # 有了新改动，"重做"就失效了（和常见编辑器一致）
                self.mark_dirty()
            self._pending = None

        def mark_dirty(self):
            if self.pm:
                self.pm.dirty = True
            self.refresh_info()

        def add_shape(self, label, bbox):
            if not self.pm:
                return
            s = {"label": label, "name": "", "bbox": bbox, "confidence": None,
                 "class_id": DEFAULT_CLASS_ID.get(label, 0), "source": "manual",
                 "raw": {}, "ocr_index": None}
            self.begin_change()
            self.pm.shapes.append(s)
            it = BoxItem(s)
            self.scene.addItem(it)
            self.items.append(it)
            self.scene.clearSelection()
            it.setSelected(True)
            self.end_change()
            if label == "Node":
                self.statusBar().showMessage("新加的 Node 请填 LBD 名字", 6000)

        def _ask_clean_manual(self, title, scope):
            """问"手工框之间的重复要不要一起清"。返回 True/False，点取消返回 None。"""
            box = QMessageBox(self)
            box.setWindowTitle(title)
            box.setIcon(QMessageBox.Icon.Question)
            box.setText("手工框之间的重复要不要一起清？\n范围：%s" % scope)
            box.setInformativeText(
                "· 是 —— 同类别的框互相重叠 50% 以上就只留一个\n"
                "        （优先留框里真读到过 LBD 编号的，其次置信度高的）\n"
                "· 否 —— 只清模型画的框，手工框一律不动\n\n"
                "锁定的框一律保留，不会被清掉；这一步可以用 Ctrl+Z 撤销。")
            yes = box.addButton("是（连手工框一起清）", QMessageBox.ButtonRole.YesRole)
            no = box.addButton("否（只清模型框）", QMessageBox.ButtonRole.NoRole)
            box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(no)
            box.exec()
            hit = box.clickedButton()
            if hit is yes:
                return True
            if hit is no:
                return False
            return None

        def clean_pages(self, pages, clean_manual=False, quiet=True, on_step=None):
            """在指定页上清多余框（本页和整册都走这里）。

            锁定的框一律保留（clean_shapes 里保证）。整册算"一步"，
            Ctrl+Z 一次就能把所有页一起撤回。返回 (统计, 每页明细)。
            """
            undone, lines = [], []
            tot = {"removed": 0, "by_manual": 0, "by_number": 0, "by_overlap": 0,
                   "manual_dup": 0, "by_shape": 0, "kept": 0, "kept_manual": 0,
                   "pages": 0, "first": None, "cur": 0, "changed": []}
            for k, pg in enumerate(pages):
                if on_step is not None and on_step(k, len(pages), pg) is False:
                    break                      # 用户点了取消
                pm = self._pm_of(pg)
                if pm is None:
                    continue
                kept, st = clean_shapes(pm.shapes, clean_manual=clean_manual)
                if not st["removed"]:
                    continue
                undone.append((pg, [(i, dict(s)) for i, s in enumerate(pm.shapes)]))
                pm.shapes = kept
                pm.dirty = True
                self.edited[pg] = pm
                for key in ("removed", "by_manual", "by_number", "by_overlap",
                            "manual_dup", "by_shape", "kept", "kept_manual"):
                    tot[key] += st.get(key, 0)
                tot["pages"] += 1
                tot["changed"].append(pg)
                if tot["first"] is None:
                    tot["first"] = pg
                if pg == self.page:
                    tot["cur"] += st["removed"]
                lines.append("第 %s 页：删 %d" % (pg, st["removed"]))
            if undone:
                self.undo.append(undone)
                del self.undo[:-200]
                self.redo.clear()
                if any(pg == self.page for pg, _s in undone):
                    self._rebuild_items()
                self.mark_dirty()
                self.on_selection()
            if tot["removed"]:
                self.statusBar().showMessage(
                    "清理多余框：%d 页共删了 %d 个" % (tot["pages"], tot["removed"]), 8000)
            else:
                self.statusBar().showMessage("清理多余框：没有多余框可清", 5000)
            if not quiet:
                self._show_clean_result(tot, lines, clean_manual, len(pages))
            return tot, lines

        def _show_clean_result(self, tot, lines, clean_manual, n_pages):
            if not tot["removed"]:
                QMessageBox.information(self, "清理多余框",
                                        "这 %d 页里没有多余框可清。" % n_pages)
                return
            body = "\n".join(lines[:40])
            if len(lines) > 40:
                body += "\n…（共 %d 页有清理）" % len(lines)
            box = QMessageBox(self)
            box.setWindowTitle("清理多余框")
            box.setIcon(QMessageBox.Icon.Information)
            box.setText(
                "共删掉 %d 个多余框（%d 页）；本页删了 %d 个。" % (
                    tot["removed"], tot["pages"], tot.get("cur", 0)))
            box.setInformativeText(
                "已经有手工框了 %d 个\n"
                "同一个编号重复 %d 个\n"
                "模型框互相重叠 %d 个\n"
                "手工框互相重叠 %d 个%s\n"
                "形状离谱（Tracker 又宽又扁）%d 个\n\n"
                "锁定的框一个没动；Ctrl+Z 可以一次撤销这一步。\n"
                "改动要保存才落盘 —— 别忘了存。"
                % (tot["by_manual"], tot["by_number"], tot["by_overlap"],
                   tot["manual_dup"],
                   "" if clean_manual else "（这次没清手工框）", tot["by_shape"]))
            box.setDetailedText(body)
            b_save = box.addButton("现在保存", QMessageBox.ButtonRole.AcceptRole)
            b_jump = None
            if tot.get("first") is not None and tot["first"] != self.page:
                b_jump = box.addButton("跳到第 %s 页看看" % tot["first"],
                                       QMessageBox.ButtonRole.ActionRole)
            box.addButton("关闭", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            hit = box.clickedButton()
            if hit is b_save:
                self.on_save_over()
            elif b_jump is not None and hit is b_jump:
                self.goto_page(tot["first"])

        def scan_clean(self, pages, on_step=None):
            """只扫描不修改：返回 (只清模型框的统计, 连手工框一起清的统计)。

            每项含 removed / pages / first / cur（当前页能删多少）。
            """
            a = {"removed": 0, "pages": 0, "first": None, "cur": 0}
            b = {"removed": 0, "pages": 0, "first": None, "cur": 0}
            for k, pg in enumerate(pages):
                if on_step is not None and on_step(k, len(pages), pg) is False:
                    break
                pm = self._pm_of(pg)
                if pm is None:
                    continue
                _k1, s1 = clean_shapes(pm.shapes, clean_manual=False)
                _k2, s2 = clean_shapes(pm.shapes, clean_manual=True)
                for d, s in ((a, s1), (b, s2)):
                    if s["removed"]:
                        d["removed"] += s["removed"]
                        d["pages"] += 1
                        if d["first"] is None:
                            d["first"] = pg
                        if pg == self.page:
                            d["cur"] += s["removed"]
            return a, b

        def on_clean_shapes(self, quiet=False, clean_manual=None):
            """清掉当前页多余的框（只删模型画的，人工框不动）。"""
            if not self.pm:
                return None
            if clean_manual is None:
                clean_manual = self._ask_clean_manual("清理多余框", "第 %d 页" % self.page)
                if clean_manual is None:
                    return None
            return self.clean_pages([self.page], clean_manual, quiet=quiet)

        def _progress(self, total, title, verb):
            """建一个带取消按钮的进度框，返回 (prog, step 回调, 取消状态 dict)。

            重要：不能在 prog.close() 之后读 wasCanceled() —— close() 自己就会把它置真，
            那样每回都会被当成"用户取消"（v0.7 里整册清理扫完却什么都不删，就是这个坑）。
            取消状态只在 setValue 之前、循环中间读。
            """
            from PySide6.QtWidgets import QProgressDialog
            prog = QProgressDialog("%s…" % title, "取消", 0, total, self)
            prog.setWindowTitle(title)
            prog.setMinimumDuration(0)
            prog.setWindowModality(Qt.WindowModality.WindowModal)
            state = {"cancel": False}

            def step(k, all_n, pg):
                if prog.wasCanceled():
                    state["cancel"] = True
                    return False
                prog.setValue(k)
                prog.setLabelText("%s第 %d 页（%d/%d）…" % (verb, pg, k + 1, all_n))
                QApplication.processEvents()
                return not prog.wasCanceled()

            return prog, step, state

        def scan_clean_progress(self, pages):
            """带进度条扫一遍整册，返回 (只清模型框的统计, 连手工框的统计, 是否被取消)。"""
            prog, step, state = self._progress(len(pages), "清理多余框（整册）", "扫描")
            try:
                sa, sb = self.scan_clean(pages, on_step=step)
            finally:
                prog.close()
            return sa, sb, state["cancel"]

        def clean_pages_progress(self, pages, clean_manual):
            """带进度条清整册，返回 (统计, 每页明细, 是否被取消)。"""
            prog, step, state = self._progress(len(pages), "清理多余框（整册）", "清理")
            try:
                tot, lines = self.clean_pages(pages, clean_manual,
                                              quiet=True, on_step=step)
            finally:
                prog.close()
            return tot, lines, state["cancel"]

        def on_clean_shapes_all(self, quiet=False, clean_manual=None):
            """整册清理：先扫描汇报能删多少，确认后逐页清，最后给"保存/跳转"。

            clean_manual 给了就跳过所有弹窗（自检用），quiet=True 只跳过最后的结果框。
            """
            if not (self.pm and self.dbg):
                return None
            pages = self.dbg.page_numbers()
            if clean_manual is None:
                sa, sb, canceled = self.scan_clean_progress(pages)
                if canceled:
                    self.statusBar().showMessage("已取消扫描，什么都没改", 5000)
                    return None
                if not sa["removed"] and not sb["removed"]:
                    QMessageBox.information(
                        self, "清理多余框（整册）",
                        "整册 %d 页扫完了，按现在的规则没有多余框可清。\n\n"
                        "（只按「模型框之间的重复」算：手工框要么用「连手工框一起清」，"
                        "要么本来就不重复；锁定的框一律不算。）" % len(pages))
                    return None
                box = QMessageBox(self)
                box.setWindowTitle("清理多余框（整册）")
                box.setIcon(QMessageBox.Icon.Question)
                box.setText("整册 %d 页扫完了，要按哪种规则清？" % len(pages))
                box.setInformativeText(
                    "· 只清模型框：能删 %d 个（分布在 %d 页）\n"
                    "· 连手工框一起清：能删 %d 个（分布在 %d 页）\n\n"
                    "本页（第 %s 页）分别能删 %d 个 / %d 个。\n"
                    "锁定的框一个都不动。清完可以用 Ctrl+Z 一次全撤销。"
                    % (sa["removed"], sa["pages"], sb["removed"], sb["pages"],
                       self.page, sa["cur"], sb["cur"]))
                b_m = box.addButton("只清模型框", QMessageBox.ButtonRole.YesRole)
                b_a = box.addButton("连手工框一起清", QMessageBox.ButtonRole.ActionRole)
                box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
                box.setDefaultButton(b_m)
                box.exec()
                hit = box.clickedButton()
                if hit is b_m:
                    clean_manual = False
                elif hit is b_a:
                    clean_manual = True
                else:
                    self.statusBar().showMessage("已取消，什么都没改", 4000)
                    return None

            tot, lines, canceled = self.clean_pages_progress(pages, clean_manual)
            if canceled:
                self.statusBar().showMessage("整册清理被取消（已清的部分还可以 Ctrl+Z 撤销）", 6000)
            if not quiet:
                self._show_clean_result(tot, lines, clean_manual, len(pages))
            return tot, lines

        def on_delete(self):
            if not self.pm:
                return
            sel = [i for i in self.scene.selectedItems() if isinstance(i, BoxItem)]
            if not sel:
                return
            self.begin_change()
            for it in sel:
                if it.is_locked():          # 锁上的框删不掉（正常也选不中，这里再兜一道）
                    continue
                if it.shape_data in self.pm.shapes:
                    self.pm.shapes.remove(it.shape_data)
                if it in self.items:
                    self.items.remove(it)
                self.scene.removeItem(it)
            self.end_change()
            self.on_selection()

        def current_item(self):
            sel = [i for i in self.scene.selectedItems() if isinstance(i, BoxItem)]
            return sel[0] if sel else None

        def on_selection(self):
            it = self.current_item()
            self.cmb_label.blockSignals(True)
            self.ed_name.blockSignals(True)
            if it is None:
                self.ed_name.setText("")
                self.lbl_bbox.setText("-")
                self.ed_name.setEnabled(False)
                self.cmb_label.setEnabled(False)
                self.btn_del.setEnabled(False)
            else:
                self.ed_name.setEnabled(True)
                self.cmb_label.setEnabled(True)
                self.btn_del.setEnabled(not it.is_locked())
                self.ed_name.setText(it.shape_data.get("name") or "")
                i = self.cmb_label.findText(it.shape_data["label"])
                if i >= 0:
                    self.cmb_label.setCurrentIndex(i)
                b = it.scene_box()
                self.lbl_bbox.setText("%.0f, %.0f → %.0f, %.0f" % (b[0], b[1], b[2], b[3]))
            self.cmb_label.blockSignals(False)
            self.ed_name.blockSignals(False)
            self.sync_lock_ui()
            self.update_state_label()

        def on_label_changed(self, _i):
            it = self.current_item()
            if it is None:
                return
            it.shape_data["label"] = self.cmb_label.currentText()
            it.shape_data["class_id"] = DEFAULT_CLASS_ID.get(it.shape_data["label"], 0)
            if it.shape_data["label"] != "Node":
                it.shape_data["name"] = ""
                self.ed_name.setText("")
            it.apply_pen()
            it.update()
            self.mark_dirty()

        def on_name_changed(self):
            it = self.current_item()
            if it is None:
                return
            txt = self.ed_name.text().strip()
            if txt == (it.shape_data.get("name") or ""):
                return
            it.shape_data["name"] = txt
            it.update()
            self.mark_dirty()

        def set_mode(self, mode):
            for m in ("select",) + CLASSES:
                getattr(self, "act_" + m).setChecked(m == mode)
            self.canvas.set_mode(mode)
            self.refresh_info()

        # ---------------- 缩放 / 状态栏
        def zoom_by(self, f):
            self.canvas.zoom_step(f)
            self.update_state_label()

        def update_state_label(self):
            """状态栏右下角常驻信息（模式 / 页码 / 缩放 / 数量 / 改动）。"""
            try:
                if not (self.pm and self.dbg):
                    self.lbl_state.setText("未载入图纸")
                    return
                c = self.pm.counts()
                locked = sum(1 for s in self.pm.shapes if s.get("locked"))
                self.lbl_state.setText(
                    "%s ｜ 第 %d/%d 页 ｜ 缩放 %.0f%% ｜ Node %d · Tracker %d · Box %d%s%s"
                    % (MODE_TEXT.get(self.canvas.mode, self.canvas.mode),
                       self.page, len(self.dbg.page_numbers()),
                       self.canvas.transform().m11() * 100.0,
                       c["Node"], c["Tracker"], c.get("Box", 0),
                       (" ｜ 锁定 %d" % locked) if locked else "",
                       " ｜ ● 未保存" if self.pm.dirty else ""))
            except Exception:
                pass

        # ---------------- 本页统计（右侧栏下半部分）
        def page_text_items(self, page=None):
            """这一页的 PDF 文字块（懒加载 + 按页缓存；没选 PDF 就返回空）。"""
            if not (self.pdf and os.path.exists(self.pdf)):
                return []
            try:
                if self._ptext is None:
                    self._ptext = PdfText(self.pdf)
                return self._ptext.items(int(page if page is not None else self.page))
            except Exception:                      # noqa: BLE001
                return []

        def box_texts_sorted(self, shape, items, width, height, tol_ratio=0.006):
            """框里（含略微超出边缘）的**全部文字**，按 从上到下、从左到右 排好。"""
            if not (shape and items):
                return []
            b = shape.get("bbox")
            if not b:
                return []
            tol = max(6.0, float(tol_ratio) * max(width or 1, height or 1))
            out = []
            for it in items:
                txt = (it[0] or "").strip()
                if not txt:
                    continue
                px, py = it[1] * (width or 1), (1.0 - it[2]) * (height or 1)
                if _box_dist(b, px, py) <= tol:
                    out.append((round(py, 1), round(px, 1), txt))
            out.sort()
            return [t for _y, _x, t in out]

        def page_rack_stats(self):
            """本页按 LBD 汇总：每个 Node（阵列块）里套着几个支架、共几串。

            判定方式：支架框的中心落在哪个 Node 框里，就算那个 LBD 的。
            串数取「支架按长度分档」写进 raw.strings 的值；没分过档的就是 0。
            返回 (每行明细, 合计)。
            """
            rows, tot = [], {"racks": 0, "strings": 0, "graded": 0}
            if not self.pm:
                return rows, tot
            nodes = [s for s in self.pm.shapes if s.get("label") == "Node"]
            racks = [s for s in self.pm.shapes if s.get("label") == "Tracker"]
            tot["racks"] = len(racks)
            for r in racks:
                st = (r.get("raw") or {}).get("strings")
                if isinstance(st, int) and st > 0:
                    tot["strings"] += st
                    tot["graded"] += 1
            for nd in nodes:
                b = nd.get("bbox")
                if not b:
                    continue
                inside = []
                for r in racks:
                    rb = r.get("bbox")
                    if not rb:
                        continue
                    cx, cy = (rb[0] + rb[2]) / 2.0, (rb[1] + rb[3]) / 2.0
                    if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                        inside.append(r)
                if not inside:
                    continue
                st = sum((r.get("raw") or {}).get("strings") or 0 for r in inside)
                rows.append(((nd.get("name") or "（没填名字）"), len(inside), st))
            rows.sort(key=lambda x: str(x[0]))
            return rows, tot

        def update_sidebar_stats(self):
            """右侧栏下半部分：本页各类框数量 + 按 LBD 的支架/串数汇总。"""
            try:
                if not (self.pm and self.dbg):
                    self.lbl_stats.setText("-")
                    self.lbl_racks.setText("")
                    return
                c = self.pm.counts()
                locked = sum(1 for s in self.pm.shapes if s.get("locked"))
                self.lbl_stats.setText(
                    "本页：Tracker %d 个，Node %d 个%s"
                    % (c["Tracker"], c["Node"], ("，锁定 %d" % locked) if locked else ""))
                rows, tot = self.page_rack_stats()
                if not tot["graded"]:
                    self.lbl_racks.setText(
                        "还没分档：点「支架按长度分档」之后，\n这里会按 LBD 列出各有多少串。")
                    return
                lines = ["按 LBD 汇总（本页）："]
                for nm, n, st in rows[:30]:
                    lines.append("  %s：%d 个支架%s"
                                 % (nm, n, "，共 %d 串" % st if st else ""))
                if len(rows) > 30:
                    lines.append("  …还有 %d 个 LBD" % (len(rows) - 30))
                lines.append("合计：%d 个支架，共 %d 串%s"
                             % (tot["racks"], tot["strings"],
                                "" if tot["graded"] == tot["racks"]
                                else "（%d 个没分档）" % (tot["racks"] - tot["graded"])))
                self.lbl_racks.setText("\n".join(lines))
            except Exception:                      # noqa: BLE001
                pass
            # 选中的框里到底印了哪些字（全部文字，不只编号）
            try:
                it = self.current_item()
                if it is None:
                    self.lbl_boxtext.setText("")
                    return
                items = self.page_text_items()
                if not items:
                    self.lbl_boxtext.setText("框内文字：（读不到 PDF 文字层 —— 没选 PDF，"
                                             "或者这份图纸是扫描图/文字已转曲）")
                    return
                txts = self.box_texts_sorted(it.shape_data, items,
                                             self.pm.width, self.pm.height)
                self.lbl_boxtext.setText(
                    "框内文字（%d 条）：\n%s" % (len(txts), "\n".join(txts[:20]))
                    if txts else "框内文字：（这个框里没读到文字）")
            except Exception:                      # noqa: BLE001
                pass

        # ---------------- 锁定框
        def sync_lock_ui(self):
            """按当前选中状态刷新「锁定/解锁」按钮和侧栏那个勾。"""
            it = self.current_item()
            lock = bool(it is not None and it.is_locked())
            for w in (getattr(self, "act_lockbox", None), getattr(self, "chk_lock", None)):
                if w is None:
                    continue
                w.blockSignals(True)
                w.setChecked(lock)
                w.setEnabled(it is not None)
                w.blockSignals(False)

        def on_toggle_show_conf(self, on):
            """建议框上的置信度数字开关（记住设置）。"""
            BoxItem.show_conf = bool(on)
            self.settings["show_conf"] = bool(on)
            save_settings(self.settings)
            self._rebuild_items()

        def on_toggle_lock(self, _checked=False):
            """锁定/解锁选中的框：只要不是"全都锁着"，就一律锁上。"""
            sel = [i for i in self.scene.selectedItems() if isinstance(i, BoxItem)]
            if not sel:
                self.statusBar().showMessage("先选中要锁定的框（可以框选多个）再按 Ctrl+L", 5000)
                self.sync_lock_ui()
                return
            target = not all(i.is_locked() for i in sel)
            self.begin_change()
            for it in sel:
                it.set_locked(target)
            self.end_change()
            self.scene.clearSelection()
            self.on_selection()
            self.statusBar().showMessage(
                "已%s %d 个框%s"
                % ("锁定" if target else "解锁", len(sel),
                   "（锁上的框点不中、拖不动，清理多余框和补编号也不会碰它）"
                   if target else ""), 6000)

        def on_lock_check(self, checked):
            it = self.current_item()
            if it is None or it.is_locked() == bool(checked):
                return
            self.begin_change()
            it.set_locked(checked)
            self.end_change()
            self.scene.clearSelection()
            self.on_selection()
            self.statusBar().showMessage(
                "已锁定这个框（Ctrl+L 也能锁）" if checked else "已解锁这个框", 4000)

        def on_unlock_page(self):
            if not self.pm:
                return
            locked = [s for s in self.pm.shapes if s.get("locked")]
            if not locked:
                self.statusBar().showMessage("本页没有锁定的框", 3000)
                return
            self.begin_change()
            for s in locked:
                s["locked"] = False
            self._rebuild_items()
            self.end_change()
            self.on_selection()
            self.statusBar().showMessage("本页 %d 个框已解锁" % len(locked), 5000)

        def unlock_item(self, it):
            """只解锁一个框（锁上的框点不中，用 Ctrl+点 走这条路）。"""
            self.begin_change()
            it.set_locked(False)
            self.end_change()
            self.scene.clearSelection()
            it.setSelected(True)
            self.on_selection()
            self.statusBar().showMessage("已解锁这个框（可以选了）", 4000)

        # ---------------- 一键锁整页的 Node
        def page_nodes(self):
            if not self.pm:
                return []
            return [s for s in self.pm.shapes if s.get("label") == "Node"]

        def sync_lock_nodes_action(self):
            """本页 Node 全锁着时，按钮变成"解锁本页 Node"并保持按下。"""
            act = getattr(self, "act_lock_nodes", None)
            if act is None:
                return
            nodes = self.page_nodes()
            all_lock = bool(nodes) and all(s.get("locked") for s in nodes)
            act.blockSignals(True)
            act.setChecked(all_lock)
            act.setText("解锁本页 Node" if all_lock else "锁定本页 Node")
            act.setEnabled(bool(nodes))
            act.blockSignals(False)

        def on_lock_page_nodes(self, _checked=False):
            """一键锁定/解锁本页所有 Node（Tracker、Box 不动）。"""
            nodes = self.page_nodes()
            if not nodes:
                self.statusBar().showMessage("本页没有 Node 框", 3000)
                self.sync_lock_nodes_action()
                return
            target = not all(s.get("locked") for s in nodes)
            changed = sum(1 for s in nodes if bool(s.get("locked")) != target)
            self.begin_change()
            for s in nodes:
                s["locked"] = target
            self._rebuild_items()
            self.end_change()
            self.on_selection()
            self.sync_lock_nodes_action()
            self.statusBar().showMessage(
                ("本页 %d 个 Node 已锁定（点不中、拖不动、删不掉；"
                 "Ctrl+Shift+L 再点一次＝全部解锁）" % len(nodes)) if target
                else ("本页 %d 个 Node 已解锁（实际改了 %d 个；Tracker/Box 一直没动）"
                      % (len(nodes), changed)), 8000)

        def refresh_info(self):
            if not (self.pm and self.dbg):
                self.statusBar().showMessage(
                    "点「打开 JSON」载入识别结果（默认不改原文件）")
                self.update_state_label()
                return
            c = self.pm.counts()
            if getattr(self.dbg, "is_folder", False):
                src = "底图 文件夹里的 png"
            else:
                src = ("底图 %d DPI" % self.dpi) if self.dpi > 0 else "底图 内嵌图"
            self.setWindowTitle("LBD 标注工具 v%s [%s] — %s ｜ %s"
                                % (ANNOTATOR_VERSION, BUILD_STAMP,
                                   os.path.basename(self.dbg.path), src))
            extra = ""
            if getattr(self.dbg, "is_folder", False):
                extra = ("\n图纸：%s\n页名：%s\n（保存＝写回这一页自己的 json）"
                         % (self.dbg.group_of(self.page), self.dbg.title_of(self.page)))
            n_sug = n_hi = n_mid = n_low = 0
            for s in self.pm.shapes:
                raw = s.get("raw") if isinstance(s.get("raw"), dict) else {}
                if (s.get("source") == "suggest"
                        or str(raw.get("xl_label") or "").endswith("?")):
                    n_sug += 1
                    try:
                        cf = float(s.get("confidence") or 0.0)
                    except Exception:
                        cf = 0.0
                    if cf >= 0.7:
                        n_hi += 1
                    elif cf >= 0.5:
                        n_mid += 1
                    else:
                        n_low += 1
            if n_sug:
                extra += ("\n建议框 %d 个（≥0.7：%d　0.5~0.7：%d　<0.5 要重点核：%d）"
                          % (n_sug, n_hi, n_mid, n_low))
                n_orig = len(self.pm.shapes) - n_sug
                if n_orig:
                    extra += ("\n原有标注 %d 个（实线，不用管，只是给你参照）" % n_orig)
            self.lbl_info.setText(
                "第 %d 页 / 共 %d 页\n坐标尺寸 %d × %d\n底图：%s\n"
                "标签表：%s\nNode %d　Tracker %d　Box %d%s%s"
                % (self.page, len(self.dbg.page_numbers()), self.pm.width,
                   self.pm.height, self._img_note or "（载入中）",
                   (os.path.basename(self.xlsx) if self.xlsx else "（没选，补编号要用）"),
                   c["Node"], c["Tracker"], c.get("Box", 0),
                   extra,
                   "\n\n● 已修改，记得保存" if self.pm.dirty else ""))
            self.statusBar().showMessage(
                "模式：%s　拖框内=移动，拖白点=改大小，方向键=微调(Shift 加速)，"
                "Delete=删除，Ctrl+Z=撤销，中键拖动=平移，滚轮=缩放，A/D=翻页"
                % MODE_TEXT.get(self.canvas.mode, self.canvas.mode))
            self.update_state_label()
            self.sync_lock_nodes_action()
            self.update_sidebar_stats()

        def keyPressEvent(self, ev):
            k = ev.key()
            ctrl = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
            if ctrl and k == Qt.Key.Key_C:
                if not ev.isAutoRepeat():
                    self.on_copy()
                return
            if ctrl and k == Qt.Key.Key_V:
                # 过滤键盘自动重复：不然按住 Ctrl+V 一秒钟就会贴出几十份框
                if not ev.isAutoRepeat():
                    self.on_paste()
                return
            if k == Qt.Key.Key_Escape:
                self.on_escape()
                return
            if k == Qt.Key.Key_Delete:
                self.on_delete()
                return
            if k in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
                it = self.current_item()
                if it is not None:
                    step = 10 if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
                    dx = {Qt.Key.Key_Left: -step, Qt.Key.Key_Right: step}.get(k, 0)
                    dy = {Qt.Key.Key_Up: -step, Qt.Key.Key_Down: step}.get(k, 0)
                    self.begin_change()
                    b = it.scene_box()
                    it.setPos(b[0] + dx, b[1] + dy)
                    it.sync_shape()
                    self.end_change()
                    self.on_selection()
                    return
            if k == Qt.Key.Key_A:
                self.goto_offset(-1)
                return
            if k == Qt.Key.Key_D:
                self.goto_offset(1)
                return
            # 1/2/3/4 切模式：只在画布上生效（写在 keyPressEvent 里，
            # 用 QAction 的快捷键会被"名字"输入框抢走 —— 名字里全是数字）
            if k in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4):
                self.set_mode({Qt.Key.Key_1: "select", Qt.Key.Key_2: "Node",
                               Qt.Key.Key_3: "Tracker", Qt.Key.Key_4: "Box"}[k])
                return
            super().keyPressEvent(ev)

        def on_escape(self):
            """ESC：画到一半就丢掉当前这一笔；否则从画框模式退回选择模式。"""
            if self.canvas.cancel_draw():
                self.statusBar().showMessage("已取消这一笔", 3000)
                return
            if self.canvas.mode != "select":
                self.set_mode("select")
                self.statusBar().showMessage("已退出画框模式", 3000)

        def on_reload_page(self, quiet=False):
            """把当前页恢复成「上次打开/保存时」的样子 —— 画乱了/粘多了的出口。"""
            if not self.pm:
                return
            if not quiet and QMessageBox.question(
                    self, "重载本页",
                    "把第 %d 页恢复成上次打开 / 保存时的样子？\n\n"
                    "这一页上后来画的、改的、粘的都会丢掉，别的页不受影响。"
                    % self.page) != QMessageBox.StandardButton.Yes:
                return
            self.begin_change()
            self.pm.load()
            self.pm.dirty = True
            self._rebuild_items()
            self.end_change()
            self.refresh_info()
            self.on_selection()
            self.statusBar().showMessage(
                "第 %d 页已恢复（Ctrl+Z 可以撤销这次恢复）" % self.page, 6000)

        def on_copy(self):
            sel = [i for i in self.scene.selectedItems() if isinstance(i, BoxItem)]
            if not sel:
                self.statusBar().showMessage("先选中要复制的框（可以框选多个）", 4000)
                return
            self.clip = [copy.deepcopy(it.shape_data) for it in sel]
            for s in self.clip:
                # 复制出来的是"新框"：不能再认原来那条编号记录，否则回写时两个框抢一条记录
                s["ocr_index"] = None
            self._paste_n = 0
            self.statusBar().showMessage(
                "已复制 %d 个框；切到目标页按 Ctrl+V 原位贴一份" % len(self.clip), 6000)

        def on_paste(self):
            if not self.pm:
                return
            if not self.clip:
                self.statusBar().showMessage("剪贴板里还没有框：先选中再按 Ctrl+C", 4000)
                return
            self._paste_n += 1
            clip = self.clip
            # 原位粘贴：新框直接盖在被复制的那一份上面（同一个位置），贴出来是选中状态，
            # 直接拖到要去的地方就行 —— 不用再自己把偏移回来的框拖回去。
            self.begin_change()
            self.scene.clearSelection()
            made = 0
            for s in clip:
                ns = copy.deepcopy(s)
                ns["ocr_index"] = None
                ns["locked"] = False      # 贴出来的必须能动，哪怕被复制的原框是锁着的
                self.pm.shapes.append(ns)
                it = BoxItem(ns)
                self.scene.addItem(it)
                self.items.append(it)
                it.setSelected(True)
                made += 1
            self.end_change()
            self.on_selection()
            self.statusBar().showMessage(
                "已原位粘贴 %d 个框（盖在原框上面，现在是选中的，直接拖走就行）" % made, 6000)

        def on_undo(self):
            if not self.undo:
                self.statusBar().showMessage("没有可撤销的操作（Ctrl+Y 可以重做）", 3000)
                return
            entry = _as_pages(self.undo.pop())
            self.redo.append(self._snap_pages(entry))
            self._restore_many(entry)
            self.statusBar().showMessage("已撤销；想还原按 Ctrl+Y", 4000)

        def on_redo(self):
            if not self.redo:
                self.statusBar().showMessage("没有可重做的操作", 3000)
                return
            entry = _as_pages(self.redo.pop())
            self.undo.append(self._snap_pages(entry))
            self._restore_many(entry)
            self.statusBar().showMessage("已重做", 3000)

        def _pm_of(self, page):
            """拿到某一页的页面模型：当前页 / 改过还没存的页 / 从 JSON 临时读一页。"""
            try:
                if self.pm is not None and page == self.page:
                    return self.pm
                if page in self.edited:
                    return self.edited[page]
                return PageModel(self.dbg, page)
            except Exception:
                return None

        def _snap_pages(self, entry):
            """把 entry 里涉及的页的"现在"存成快照（撤销/重做互换时用）。"""
            out = []
            for page, _snap in entry:
                pm = self._pm_of(page)
                if pm is not None:
                    out.append((page, [(i, dict(s)) for i, s in enumerate(pm.shapes)]))
            return out

        def _restore_many(self, entry):
            """把若干页恢复成快照的样子（撤销/重做共用，整册操作也是一步一件）。"""
            pages = [p for p, _s in entry]
            # 先把不在当前页的那些页恢复好（写进 edited，goto 时会用内存里这份）
            for page, snap in [(p, s) for p, s in entry if p != self.page]:
                pm = self._pm_of(page)
                if pm is None:
                    continue
                pm.shapes = [dict(s) for _i, s in snap]
                pm.dirty = True
                self.edited[page] = pm
            if pages and self.page not in pages:
                # 撤销的不是当前页：跳回去，让人看见撤销了什么（和以前的习惯一致）
                self.goto_page(pages[0])
            for _page, snap in [(p, s) for p, s in entry if p == self.page]:
                self.pm.shapes = [dict(s) for _i, s in snap]
                self._rebuild_items()
                self.pm.dirty = True
            self.refresh_info()
            self.on_selection()

        # ---------------- 标签表 / 自动补编号
        def sync_shapes(self):
            """把界面上框的位置写回数据。

            拖动框之后必须调一次 —— 这个 PySide6 版本的 QGraphicsItem::itemChange
            收不到 "位置变了" 事件，靠它同步的话拖动永远不会写回 bbox，
            保存出来还是老坐标（就是"移动支架位置保存不生效"那个 bug）。
            """
            if not self.pm:
                return
            for it in self.items:
                it.sync_shape()
            it = self.current_item()
            if it is not None:
                b = it.scene_box()
                self.lbl_bbox.setText("%.0f, %.0f → %.0f, %.0f"
                                      % (b[0], b[1], b[2], b[3]))

        def on_name_px(self, _i=None):
            px, selonly = self.cmb_name.currentData() or (11, True)
            BoxItem.name_px = float(px)
            BoxItem.name_sel_only = bool(selonly)
            for it in self.items:
                it.update()
            s = load_settings()
            s["name_mode"] = self.cmb_name.currentIndex()
            save_settings(s)

        def on_lock_window(self, on):
            try:
                if on:
                    # 锁的是"画布区"：画布固定住，整个窗口仍然可以拉大拉小
                    self.canvas.setFixedSize(self.canvas.size())
                else:
                    self.canvas.setMinimumSize(320, 240)
                    self.canvas.setMaximumSize(16777215, 16777215)
                    self.setMaximumSize(16777215, 16777215)
                    self.setMinimumSize(1180, 740)
            except Exception:
                pass
            s = load_settings()
            s["win_locked"] = bool(on)
            save_settings(s)
            self.statusBar().showMessage(
                "窗口大小已锁定（再点一下「锁定窗口大小」解锁）" if on
                else "窗口大小已解锁（现在可以自由拉动）", 4000)

        def closeEvent(self, ev):
            try:
                s = load_settings()
                s["win_w"], s["win_h"] = int(self.width()), int(self.height())
                s["name_mode"] = self.cmb_name.currentIndex()
                save_settings(s)
            except Exception:
                pass
            super().closeEvent(ev)

        def on_pick_xlsx(self):
            base = (self.xlsx if (self.xlsx and os.path.exists(self.xlsx))
                    else os.path.expanduser("~"))
            p, _ = QFileDialog.getOpenFileName(self, "选择 LBD 标签表", base,
                                               "Excel (*.xlsx)")
            if not p:
                return
            self.xlsx = p
            self._sheet_cache = None
            s = load_settings()
            s["xlsx"] = p
            save_settings(s)
            self.refresh_info()
            self.statusBar().showMessage("标签表：%s" % p, 6000)

        def _sheet_rows(self):
            """(分表名列表, 取某分表 LBD 行的函数)；读不到返回 (None, None)。"""
            return self._sheets_and_rows()

        def _page_order(self):
            """要处理的页顺序（升序）：识别结果 JSON = 有识别记录的图纸页；
            直接标 PDF（BlankDoc）= 本会话里已经画了框的页。"""
            pages = []
            if hasattr(self.dbg, "_elements"):
                try:
                    pages = list(drawing_pages(self.dbg))
                except Exception:
                    pages = []
            for pg in self.dbg.page_numbers():
                pm = self.pm if pg == self.page else self.edited.get(pg)
                if pm is not None and any(s.get("label") == "Node" for s in pm.shapes):
                    pages.append(pg)
            return sorted(set(pages))

        def _sheets_and_rows(self):
            if not self.xlsx or not os.path.exists(self.xlsx):
                return None, None
            if self._sheet_cache is None or self._sheet_cache[0] != self.xlsx:
                self._sheet_cache = (self.xlsx, xlsx_sheet_names(self.xlsx), {})
            _p, sheets, cache = self._sheet_cache

            def rows_of(name):
                if name not in cache:
                    try:
                        cache[name] = xlsx_lbd_rows(self.xlsx, name)
                    except Exception:
                        cache[name] = []
                return cache[name]

            return sheets, rows_of

        def on_autofill(self, whole=True):
            """按「已框好的 LBD 区域 + 框内文字」补 LBD 名字，整册跑时逐页报进度。"""
            try:
                self._autofill_impl(whole)
            except Exception:
                import traceback
                QMessageBox.critical(self, "补编号出错（把这段发我）",
                                     traceback.format_exc())

        def _autofill_impl(self, whole=True):
            from PySide6.QtWidgets import QProgressDialog
            if not self.pm or not self.dbg:
                self.statusBar().showMessage("先打开一份 JSON（或 PDF）再补编号", 5000)
                return
            # 本页先看一眼有没有可补的框：没有就直说，别让人以为"点了没反应"
            if not whole:
                n_node = sum(1 for s in self.pm.shapes if s.get("label") == "Node")
                if n_node == 0:
                    QMessageBox.information(
                        self, "本页没有 Node 框",
                        "第 %d 页上 Node 框 0 个（这页不是图纸页，或者框还没画）。\n\n"
                        "补编号只对有 Node 框（LBD 区域）的图纸页有用；"
                        "也可以直接用「补全编号(整册)」。" % self.page)
                    return
                self.statusBar().showMessage("正在补第 %d 页（Node %d 个）…" % (self.page, n_node))
                QApplication.processEvents()
            sheets, rows_of = self._sheet_rows()
            text_only = False
            if not sheets:
                # 没选标签表也能干：只用框内文字取号（号码不跟表核对）
                if QMessageBox.question(
                        self, "没选标签表",
                        "没有选 LBD 标签表(xlsx)。\n\n"
                        "· 是 —— 只用「框内文字」取号：框里印了编号就用它，"
                        "框里没印号的留空标红，不按标签表顺序推。\n"
                        "· 否 —— 先点工具栏「选标签表…」选一个"
                        "（那样还能用表里的号按顺序补、并和表核对）。",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes) != QMessageBox.StandardButton.Yes:
                    return
                text_only = True
            if not self.pdf or not os.path.exists(self.pdf):
                QMessageBox.information(
                    self, "缺 PDF",
                    "框内取文字要读 PDF 文字层，请先「选 PDF…」指定这份 JSON 对应的 PDF。")
                return
            # 直接标 PDF（没 JSON）时 dbg 是 BlankDoc：它没有"识别记录"，
            # 只能按"本会话里有框的页"来排；识别结果 JSON 就按 debug_page_map 那套。
            blank = not hasattr(self.dbg, "_elements")
            order = self._page_order()
            if self.page not in order:
                order = sorted(set(order) | {self.page})
            todo = order if whole else [self.page]
            force = False
            try:
                force = bool(self.chk_force.isChecked())
                s = load_settings()
                s["force"] = force
                save_settings(s)
            except Exception:
                force = False
            try:
                ptext = PdfText(self.pdf)
            except Exception as e:
                QMessageBox.warning(self, "读不了 PDF", "打开 PDF 失败：%s" % e)
                return
            prog = QProgressDialog("正在按框内文字补编号…", "取消", 0, len(todo), self)
            prog.setWindowTitle("补全 LBD 编号")
            prog.setMinimumDuration(0)
            prog.setWindowModality(Qt.WindowModality.WindowModal)
            lines, n_ok, n_auto, n_miss, n_skip, n_locked = [], 0, 0, 0, 0, 0
            check = []          # 要人核的：位置可能错的（紫）+ 按顺序推的（黄）
            try:
                for k, pg in enumerate(todo):
                    prog.setValue(k)
                    prog.setLabelText("第 %d 页（%d/%d）…" % (pg, k + 1, len(todo)))
                    QApplication.processEvents()
                    if prog.wasCanceled():
                        break
                    pm = self.pm if pg == self.page else (self.edited.get(pg)
                                                          or PageModel(self.dbg, pg))
                    if force:
                        # 重算：把 Node 上已有的名字清掉（识别给的号不对时用这个）
                        for sh in pm.shapes:
                            if sh.get("label") == "Node":
                                sh["name"] = ""
                                sh["_auto"] = False
                                sh["_miss"] = False
                                sh["_check"] = False
                    items = ptext.items(pg)
                    hit, _cnt = page_sheet_by_text(items, sheets)
                    idx = order.index(pg) if pg in order else -1
                    by_order = sheets[idx] if 0 <= idx < len(sheets) else ""
                    # 直接标 PDF 时没有"图纸顺序"可依，就用页面上印的分表名；
                    # 识别结果 JSON 还是按"第几页 = 第几个分表"（和主程序一致）。
                    # 没选标签表时也只能靠页面上印的名字（否则名字里会缺 INV 前缀）
                    sheet = (hit or by_order) if (blank or text_only) else by_order
                    num_set = (set() if text_only
                               else {n for n, _l in (rows_of(sheet) if sheet else [])})
                    st = autofill_shapes(pm.shapes, items, sheet, num_set,
                                         pm.width, pm.height)
                    pm.dirty = True
                    self.edited[pg] = pm
                    # 没选标签表时：没取到号的按「位置顺序 + 编号连续」推一把（标黄）
                    n_tui = (infer_lbd_names(pm.shapes, only_label="Node")
                             if text_only else 0)
                    if pg == self.page:
                        self._rebuild_items()
                    n_ok += st["filled"]
                    n_auto += st["auto"] + n_tui
                    n_miss += st["missed"]
                    n_skip += st["kept"]
                    n_locked += st.get("locked", 0)
                    note = ""
                    if hit and sheet and _clean_key(hit) != _clean_key(sheet):
                        note = "；⚠页面文字里印的是 %s（按顺序推的是 %s）" % (hit, sheet)
                    details = ""
                    if n_tui:
                        details += "，按位置+编号连续性推 %d（黄色）" % n_tui
                    if st["auto"]:
                        details += "，其中按标签表顺序推 %d（黄色）" % st["auto"]
                    if st["missed"]:
                        details += "，没取到 %d（红色）" % st["missed"]
                    lines.append("第 %s 页 %s：补上 %d 个%s%s"
                                 % (pg, sheet or "（推不出分表）", st["filled"],
                                    details, note))
                    for nm, x, y in (st.get("sus_list") or []):
                        check.append("第 %s 页 %s —— 同一排的号不连续，位置可能错"
                                     "（图上大约 x=%d y=%d）" % (pg, nm or "（空）", x, y))
                    for nm in (st.get("auto_list") or []):
                        check.append("第 %s 页 %s —— 图上没找到编号，按标签表顺序推的"
                                     % (pg, nm or "（空）"))
            finally:
                prog.close()
                self.refresh_info()
                self.on_selection()
            head = ("框内文字取到 %d 个；按标签表顺序补 %d 个（黄色，请核一下）；"
                    "没取到 %d 个（红色，手填）；已有名字跳过 %d 个%s。\n"
                    "要你核的一共 %d 个（下面列出来；紫=号跳号、黄=按顺序推的）\n\n"
                    % (n_ok, n_auto, n_miss, n_skip,
                       ("；锁定的 %d 个没动" % n_locked) if n_locked else "",
                       len(check)))
            if text_only:
                head = ("【这次没选标签表】名字全部来自框内文字，"
                        "号码没有和标签表核对过 —— 请抽几页确认一下。\n\n") + head
            if not whole:
                n_node = sum(1 for s in self.pm.shapes if s.get("label") == "Node")
                head = ("第 %d 页（Node 框 %d 个）\n" % (self.page, n_node)) + head
                if n_node and not (n_ok or n_auto or n_miss) and n_skip == n_node:
                    head = ("第 %d 页：%d 个框全都有名字了，我默认不动它们。\n"
                            "如果这些名字是错的，勾上工具栏的「重算(覆盖已有名字)」再点一次。\n\n"
                            % (self.page, n_node)) + head
            body = "\n".join(lines[:60])
            if len(lines) > 60:
                body += "\n…（共 %d 页，完整清单见状态栏）" % len(lines)
            if check:
                body += "\n\n【需要核对的】\n" + "\n".join(check[:60])
                if len(check) > 60:
                    body += "\n…（还有 %d 个，见 --autofill 的日志）" % (len(check) - 60)
            try:
                self.statusBar().showMessage(
                    "补编号完成：框内 %d / 推 %d / 缺 %d" % (n_ok, n_auto, n_miss), 0)
            except Exception:
                pass
            QMessageBox.information(self, "补全 LBD 编号", head + body)

        def on_export_dataset(self):
            """导出 YOLO 训练集：images/ + labels/ + classes.txt + data.yaml。"""
            from PySide6.QtWidgets import QProgressDialog
            if not (self.dbg and self.pdf and os.path.exists(self.pdf)):
                QMessageBox.information(self, "缺 PDF",
                                        "导出数据集要用 PDF 渲染图片，请先「选 PDF…」。")
                return
            base = os.path.join(os.path.dirname(self.dbg.path), "yolo_dataset")
            out = QFileDialog.getExistingDirectory(self, "选数据集输出目录", base)
            if not out:
                return
            img_dir = os.path.join(out, "images")
            lab_dir = os.path.join(out, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lab_dir, exist_ok=True)
            names = list(CLASSES)
            with open(os.path.join(out, "classes.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(names) + "\n")
            with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
                f.write("path: %s\ntrain: images\nval: images\nnames:\n"
                        % out.replace("\\", "/"))
                for i, n in enumerate(names):
                    f.write("  %d: %s\n" % (i, n))
            pages = drawing_pages(self.dbg) or [self.page]
            dpi = self.dpi if self.dpi > 0 else 250
            prog = QProgressDialog("正在导出 YOLO 数据集…", "取消", 0, len(pages), self)
            prog.setWindowModality(Qt.WindowModality.WindowModal)
            prog.setMinimumDuration(0)
            n_img = n_box = 0
            try:
                for k, pg in enumerate(pages):
                    prog.setValue(k)
                    prog.setLabelText("第 %d 页（%d/%d）…" % (pg, k + 1, len(pages)))
                    QApplication.processEvents()
                    if prog.wasCanceled():
                        break
                    src = self.renderer.render(self.pdf, pg, dpi)
                    dst = os.path.join(img_dir, "p%04d.png" % pg)
                    shutil.copyfile(src, dst)
                    pm = self.pm if pg == self.page else (self.edited.get(pg)
                                                          or PageModel(self.dbg, pg))
                    W = float(pm.width) or 1.0
                    H = float(pm.height) or 1.0
                    lines = []
                    for s in pm.shapes:
                        b = s["bbox"]
                        cls = names.index(s["label"]) if s["label"] in names else 0
                        lines.append("%d %.6f %.6f %.6f %.6f"
                                     % (cls, (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H,
                                        (b[2] - b[0]) / W, (b[3] - b[1]) / H))
                        n_box += 1
                    with open(os.path.join(lab_dir, "p%04d.txt" % pg), "w",
                              encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                    n_img += 1
            finally:
                prog.close()
            QMessageBox.information(
                self, "数据集导出完成",
                "目录：%s\n\n图片 %d 张、标注框 %d 个，类别顺序：%s\n\n"
                "训练（在装了 ultralytics 的 Python 环境里）：\n"
                "yolo train data=%s/data.yaml model=yolov8n.pt imgsz=2560\n"
                % (out, n_img, n_box, " / ".join(names),
                   out.replace("\\", "/")))

        def on_ai_detect(self, whole=False):
            """用训练好的 YOLO 模型识别图纸：选模型 → 阈值 → 本页/整册 → 自动把框画上去。"""
            # 点了先给一个反馈：这样"没反应"到底是"没点进来"还是"面板没弹出来"一目了然
            try:
                self.statusBar().showMessage("正在打开「用模型识别」面板…", 4000)
                QApplication.processEvents()
            except Exception:
                pass
            # 出任何错都弹出来（窗口版看不到控制台，不然就是"点了没反应"）
            try:
                self._ai_detect_impl(whole)
            except Exception:
                import traceback
                QMessageBox.critical(self, "识别面板出错（把这段发我）", traceback.format_exc())

        # ---------------- OCR：读 LBD 框里的文字（图纸里那些字是矢量轮廓时的兜底）
        def clear_tracker_lbd_names(self, pages):
            """把旧版 OCR 写到支架框（Tracker）上的 LBD 编号清掉。

            只动"名字里有 LBD 字样"的支架框：锁定的、以及分档写的「3串」那种不动。
            整册算"一步"，Ctrl+Z 一次能全撤回来。返回清掉的个数。
            """
            undone, n = [], 0
            for pg in pages:
                pm = self._pm_of(pg)
                if pm is None:
                    continue
                hit = [i for i, s in enumerate(pm.shapes)
                       if s.get("label") != "Node" and not s.get("locked")
                       and re.search(r"LBD", s.get("name") or "", re.I)]
                if not hit:
                    continue
                undone.append((pg, [(i, dict(s)) for i, s in enumerate(pm.shapes)]))
                for i in hit:
                    pm.shapes[i]["name"] = ""
                    pm.shapes[i]["_auto"] = False
                    pm.shapes[i]["_miss"] = False
                    pm.shapes[i]["_check"] = False
                    n += 1
                pm.dirty = True
                self.edited[pg] = pm
            if undone:
                self.undo.append(undone)
                del self.undo[:-200]
                self.redo.clear()
                if any(pg == self.page for pg, _s in undone):
                    self._rebuild_items()
                self.mark_dirty()
                self.on_selection()
            return n

        def on_ocr_read(self, whole=False):
            """用 OCR 读每个 LBD 框（Node）里的文字 —— 竖排也能读；支架框不读。"""
            try:
                self._ocr_read_impl(whole)
            except Exception:
                import traceback
                QMessageBox.critical(self, "OCR 出错（把这段发我）", traceback.format_exc())

        def on_auto_name(self, whole=True):
            """一键补编号：先读 PDF 文字层（快、精确），没拿到号的框再自动上 OCR。"""
            try:
                self._ocr_read_impl(whole, text_first=True)
            except Exception:
                import traceback
                QMessageBox.critical(self, "自动补编号出错（把这段发我）",
                                     traceback.format_exc())

        def _ocr_read_impl(self, whole=False, text_first=False):
            import json as _json
            import subprocess
            import tempfile
            if not (self.pm and self.dbg):
                self.statusBar().showMessage("先打开一份 JSON 再补编号", 5000)
                return
            if not (self.pdf and os.path.exists(self.pdf)):
                QMessageBox.information(self, "缺 PDF",
                                        "补编号要用 PDF（文字层 + 渲染底图），请先「选 PDF…」。")
                return
            script = os.path.join(tempfile.gettempdir(), "lbd_ocr.py")
            # 解释器路径先取好（下面跑 OCR 要用）；"有没有装 rapidocr"那步才是懒加载的 ——
            # 这样只有文字层能搞定的图纸，压根不会去查/装 OCR 组件。
            pyp = str(self.settings.get("python") or "")
            ocr_ready = {"v": None}
            # 「重算(覆盖已有名字)」勾选框：勾上 = 已有名字也清掉重新填
            try:
                force_names = bool(self.chk_force.isChecked())
            except Exception:
                force_names = False

            def ensure_ocr():
                """真要 OCR 了才查 Python / rapidocr —— 只有文字层的图纸不用装任何东西。"""
                if ocr_ready["v"] is not None:
                    return ocr_ready["v"]
                if not (pyp and os.path.exists(pyp)):
                    QMessageBox.information(self, "缺 Python",
                                            "剩下这些框要 OCR，先在「训练环境…」里选好"
                                            " Python 解释器（3.12 那个）。")
                    ocr_ready["v"] = False
                    return False
                chk = subprocess.run([pyp, "-c", "import rapidocr_onnxruntime"],
                                     capture_output=True, text=True, creationflags=_NO_WINDOW)
                if chk.returncode != 0:
                    if QMessageBox.question(
                            self, "要装 OCR 组件",
                            "剩下这些框要 OCR，但这个 Python 里还没有 rapidocr-onnxruntime"
                            "（约 20MB，离线跑 OCR 用）。\n\n现在装吗？装完会自动继续。",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes) != QMessageBox.StandardButton.Yes:
                        ocr_ready["v"] = False
                        return False
                    self.statusBar().showMessage("正在装 rapidocr-onnxruntime …", 0)
                    QApplication.processEvents()
                    p = subprocess.run([pyp, "-m", "pip", "install", "rapidocr-onnxruntime"],
                                       capture_output=True, text=True, encoding="utf-8",
                                       errors="replace", creationflags=_NO_WINDOW)
                    self.statusBar().showMessage("", 0)
                    if p.returncode != 0:
                        QMessageBox.warning(self, "装不上",
                                            ((p.stdout or "")[-600:] + (p.stderr or "")[-600:]))
                        ocr_ready["v"] = False
                        return False
                with open(script, "w", encoding="utf-8") as f:
                    f.write(OCR_BOX_SCRIPT)
                ocr_ready["v"] = True
                return True
            # OCR 用当前渲染的底图就行（250 DPI 实测字高 ~30px，读得稳）
            dpi = self.dpi if self.dpi > 0 else 250
            dbg_dir = os.path.join(work_dir(), "ocr_debug")
            order = self._page_order() or [self.page]
            todo = order if whole else [self.page]
            # 只读 LBD 框（Node）里的字，支架框（Tracker）一个都不写。
            # v0.20 及以前的 OCR 是 Node+Tracker 一起读的，支架上因此留下了编号
            #（用户反馈"支架也被填上名字了"就是这条路径）—— 先问一句要不要清掉。
            stale = 0
            for _pg in todo:
                _pm0 = self._pm_of(_pg)
                if _pm0 is None:
                    continue
                stale += sum(1 for s in _pm0.shapes
                             if s.get("label") != "Node" and not s.get("locked")
                             and re.search(r"LBD", s.get("name") or "", re.I))
            if stale and QMessageBox.question(
                    self, "支架上还留着旧版写的编号",
                    "扫到 %d 个支架框（Tracker）上写着 LBD 编号 —— 那是 v0.21 之前的 OCR"
                    "连支架一起读留下的。\n\n"
                    "· 是 —— 现在把这些名字清掉（Ctrl+Z 能撤销）\n"
                    "· 否 —— 留着不动。这次 OCR 只会写 LBD 框（Node），支架不会再被写"
                    % stale,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes) == QMessageBox.StandardButton.Yes:
                n_clr = self.clear_tracker_lbd_names(todo)
                self.statusBar().showMessage("已清掉支架上 %d 个旧编号" % n_clr, 6000)
            stat = {"boxes": 0, "filled": 0, "raw": 0, "missed": 0, "pages": 0,
                    "text": 0, "text_pages": 0, "no_python": 0}
            heights = []
            undone, ptext = [], None
            for n, pg in enumerate(todo):
                self.statusBar().showMessage(
                    "%s第 %s 页（%d/%d）…" % ("补编号 " if text_first else "OCR ",
                                             pg, n + 1, len(todo)), 0)
                QApplication.processEvents()
                pm = self.pm if pg == self.page else (self.edited.get(pg)
                                                      or PageModel(self.dbg, pg))
                snap = [(i, dict(s)) for i, s in enumerate(pm.shapes)]
                touched = False
                cleared = {}            # 这次被清掉名字的框（重算）-> 最后没填回来就还原

                def restore_cleared():
                    """重算没填回来的框，把原来的名字还回去 —— 重算绝不能把好名字弄丢。

                    必须放在 OCR 之后：还原早了那些框就有名字了，OCR 就不会去补它们。
                    """
                    n_r = 0
                    for si, old in cleared.items():
                        s2 = pm.shapes[si]
                        if not (s2.get("name") or "").strip():
                            s2["name"] = old
                            s2["_auto"] = False
                            s2["_miss"] = False
                            n_r += 1
                    if n_r:
                        stat["restored"] = stat.get("restored", 0) + n_r
                    return n_r

                def commit_page():
                    """这一页只要动过（补了名字/位置、或重算还原过），就登记成"改过" ——
                    保存（另存为/覆盖原文件）写的是 self.edited 里的页，不登记就等于白干。
                    这里以前漏了：文字层补好了、但 OCR 那步因为没装/没配 Python 直接跳过时，
                    整页的结果不会进 self.edited，覆盖原文件以后就"看不出改过"。
                    """
                    if touched:
                        undone.append((pg, snap))
                        pm.dirty = True
                        self.edited[pg] = pm
                        if pg == self.page:
                            self._rebuild_items()
                # ① 有文字层的先把文字层用上：1~2 秒一页，而且字是原文、不会认错
                if text_first:
                    if ptext is None:                   # 这个 PDF 读完一次就复用
                        try:
                            ptext = PdfText(self.pdf)
                        except Exception:
                            ptext = False
                    items = []
                    if ptext:
                        try:
                            items = ptext.items(pg)
                        except Exception:
                            items = []
                    if items:
                        # 勾了「重算」-> 已有名字也清掉重填；没勾 -> 只把"不是规范编号"
                        # 的名字清掉（规范名字留着，缺位置的话下面会补上）
                        for si, s in enumerate(pm.shapes):
                            if s.get("label") != "Node":
                                continue
                            nm = (s.get("name") or "").strip()
                            if not (force_names or not _full_lbd_name(nm)):
                                continue
                            if nm:
                                touched = True
                                cleared[si] = nm
                            s["name"] = ""
                            s["_auto"] = False
                            s["_check"] = False
                            s["_miss"] = False
                        st0 = autofill_shapes(pm.shapes, items,
                                              page_sheet_by_text(items, [])[0] or "",
                                              set(), pm.width, pm.height)
                        if st0.get("filled") or st0.get("pos_added"):
                            touched = True
                            stat["text"] += st0["filled"]
                            if st0.get("filled"):
                                stat["text_pages"] += 1
                            stat["pos_added"] = stat.get("pos_added", 0) + st0.get("pos_added", 0)
                # ② 文字层没给到号的框（自动补编号时）才送去 OCR；单跑 OCR 时全部重读
                boxes = [{"ix": i, "label": s.get("label"), "bbox": s["bbox"]}
                         for i, s in enumerate(pm.shapes)
                         if s.get("label") == "Node"
                         and not (text_first and _full_lbd_name(s.get("name")))]
                if not boxes:
                    restore_cleared()
                    commit_page()
                    continue
                if not ensure_ocr():
                    restore_cleared()
                    stat["no_python"] += len(boxes)
                    commit_page()
                    continue
                img = self.renderer.render(self.pdf, pg, dpi)
                cfgp = os.path.join(tempfile.gettempdir(), "lbd_ocr_cfg.json")
                with open(cfgp, "w", encoding="utf-8") as f:
                    _json.dump({"img": img, "page_w": pm.width, "page_h": pm.height,
                                "page": pg, "debug_dir": dbg_dir, "boxes": boxes},
                               f, ensure_ascii=False)
                pr = subprocess.run([pyp, script, cfgp], capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=7200,
                                    creationflags=_NO_WINDOW)
                data = []
                for line in (pr.stdout or "").splitlines():
                    if line.startswith("@@"):
                        data = _json.loads(line[2:])
                if not data:
                    restore_cleared()
                    self.statusBar().showMessage("第 %s 页没拿到 OCR 结果：%s"
                                                 % (pg, (pr.stderr or "")[-200:]), 8000)
                    commit_page()
                    continue
                got = {int(r["ix"]): r for r in data}
                for r in data:
                    if r.get("h"):
                        heights.append(float(r["h"]))
                for i, s in enumerate(pm.shapes):
                    r = got.get(i)
                    if not r:            # 锁定的框也照写编号（锁只防误拖/误删）
                        continue
                    touched = True
                    txt = (r.get("text") or "").strip()
                    stat["boxes"] += 1
                    name, kind = ocr_text_to_name(txt)
                    if kind == "miss":
                        s["_miss"] = True
                        stat["missed"] += 1
                        continue
                    if kind == "noise":
                        # 读出来的是别的字：不写进名字（免得名字和框里的东西没关系），
                        # 只留个底，标红等人手填/等推理补号
                        s["name"] = ""
                        s["_check"] = False
                        s["_miss"] = True
                        raw = s.get("raw") if isinstance(s.get("raw"), dict) else {}
                        raw["ocr_text"] = txt[:60]
                        s["raw"] = raw
                        stat["missed"] += 1
                        stat["noise"] = stat.get("noise", 0) + 1
                        continue
                    s["_miss"] = False
                    if kind == "ok":
                        s["name"] = name                     # 标签本身就是 INV..-LBD-..
                        s["_check"] = False
                        stat["filled"] += 1
                    else:
                        s["name"] = name                     # 像编号但不标准 -> 标紫要人核
                        s["_check"] = True
                        stat["raw"] += 1
                    # 把"编号印在图上的哪儿"一起写下来（OCR 反算回页面坐标，实测和
                    # 文字层独立算出来的位置差 8~9 px / 页面 8500 px）
                    bb = r.get("pos")
                    if bb:
                        raw = s.get("raw") if isinstance(s.get("raw"), dict) else {}
                        raw["label_pos"] = [int(round(float(bb[0]))),
                                            int(round(float(bb[1])))]
                        bb2 = r.get("bbox")
                        if bb2:
                            raw["label_bbox"] = [int(round(float(x))) for x in bb2]
                        raw["label_src"] = "ocr"
                        s["raw"] = raw
                pm.dirty = True
                self.edited[pg] = pm
                stat["pages"] += 1
                if touched:
                    undone.append((pg, snap))
                # 没读到的：按「位置顺序 + 编号连续」推一把（推出来的标黄，要人核）
                stat["inferred"] = stat.get("inferred", 0) + infer_lbd_names(pm.shapes,
                                                                            only_label="Node")
                # 到这里文字层 + OCR 都试过了：还没名字的重算框，还原原来的名字
                restore_cleared()
                if pg == self.page:
                    self._rebuild_items()
            # 整册算"一步"：Ctrl+Z 一次能把这次补的名字全撤回来
            if undone:
                self.undo.append(undone)
                del self.undo[:-200]
                self.redo.clear()
            self.mark_dirty()
            self.on_selection()
            if text_first:
                self.statusBar().showMessage(
                    "自动补编号完成：文字层给了 %d 个（%d 页），OCR 再补 %d 个，剩 %d 个要人填"
                    % (stat["text"], stat["text_pages"], stat["filled"],
                       max(0, stat["missed"] - stat.get("inferred", 0))), 0)
            else:
                self.statusBar().showMessage(
                    "OCR 完成：%d 页 / %d 个 LBD 框，认出编号 %d 个，读到别的文字 %d 个，没读到 %d 个"
                    % (stat["pages"], stat["boxes"], stat["filled"], stat["raw"],
                       stat["missed"]), 0)
            hmed = sorted(heights)[len(heights) // 2] if heights else 0
            hint = ""
            if hmed:
                hint = ("\n标签在原图里约 %.0f px 高 —— %s"
                        % (hmed, "够用" if hmed >= 25 else
                           "偏小（认不出来多半是这个原因），可以把底图 DPI 调高再跑"))
            if text_first:
                text_msg = (
                    "自动补编号跑完 %d 页：\n"
                    "  · PDF 文字层直接给到编号：%d 个（%d 页）—— 这一路是原文，不会认错字\n"
                    "  · 文字层没给到、改用 OCR：%d 个框，其中认出标准编号 %d 个（绿）\n"
                    "  · 像编号但不标准：%d 个（紫，写成名字了，请核一下）\n"
                    "  · 框里读到的是别的字（旁注之类）：%d 个（没写进名字，标红等人填）\n"
                    "  · 没读到、但按位置+编号连续性推出来的：%d 个（黄，请核一下）\n"
                    "  · 没读到也推不出来的：%d 个（红，等人手填）%s\n\n"
                    % (len(todo), stat["text"], stat["text_pages"], stat["boxes"],
                       stat["filled"], stat["raw"], stat.get("noise", 0),
                       stat.get("inferred", 0),
                       max(0, stat["missed"] - stat.get("inferred", 0)), hint))
                if stat["boxes"]:
                    text_msg += ("剩下的框是按框本身裁图、每 5 个框拼一张识别的；"
                                 "裁好的小图在：%s\n" % dbg_dir)
                else:
                    text_msg += ("这批页一个框都没用上 OCR（文字层就够了），"
                                 "所以没要 Python / rapidocr 组件。\n")
                if stat.get("no_python"):
                    text_msg += ("⚠ 有 %d 个框本来要 OCR，但没选 Python / 没装 OCR 组件，"
                                 "这次跳过了。\n" % stat["no_python"])
                if force_names:
                    text_msg += "（勾了「重算」：已有名字已清掉重填）"
                    if stat.get("restored"):
                        text_msg += "，其中 %d 个没填回来、已还原原来的名字" % stat["restored"]
                    text_msg += "\n"
                QMessageBox.information(
                    self, "自动补编号",
                    text_msg + "支架框（Tracker）一个没读、没写名字；锁定的框也没动。"
                    "可以直接 Ctrl+Z 撤销。")
                return
            QMessageBox.information(
                self, "OCR 读框内文字",
                "跑完 %d 页、%d 个 LBD 框（Node）：\n"
                "  · 认出标准编号：%d 个（绿）\n"
                "  · 像编号但不标准：%d 个（紫，写成名字了，请核一下）\n"
                "  · 框里读到的是别的字（旁注之类）：%d 个（没写进名字，标红等人填）\n"
                "  · 没读到、但按位置+编号连续性推出来的：%d 个（黄，请核一下）\n"
                "  · 没读到也推不出来的：%d 个（红，等人手填）%s\n\n"
                "按框本身裁图、每 5 个框拼一张识别；没读到标准编号的框会自动单独重试一次。\n"
                "裁好的小图存在：%s\n"
                "支架框（Tracker）一个没读、没写名字；锁定的框也没动。可以直接 Ctrl+Z 撤销。"
                % (stat["pages"], stat["boxes"], stat["filled"], stat["raw"],
                   stat.get("noise", 0),
                   stat.get("inferred", 0),
                   max(0, stat["missed"] - stat.get("inferred", 0)), hint, dbg_dir))

        def _ai_detect_impl(self, whole=False):
            from PySide6.QtWidgets import (QDialog, QPlainTextEdit, QVBoxLayout, QHBoxLayout,
                                           QPushButton, QLabel, QComboBox, QLineEdit,
                                           QCheckBox)
            from PySide6.QtCore import QObject as _QO, Signal as _SIG
            import subprocess
            import tempfile
            import threading
            if not (self.dbg and self.pdf and os.path.exists(self.pdf)):
                QMessageBox.information(self, "缺 PDF", "识别要用 PDF 渲染底图，请先「选 PDF…」。")
                return
            pyp = str(self.settings.get("python") or "")
            if not (pyp and os.path.exists(pyp)):
                QMessageBox.information(self, "缺 Python",
                                        "先在「训练环境…」里选好 Python 解释器（3.12 那个）。")
                return
            dlg = QDialog(self)
            dlg.setWindowTitle("用模型识别（YOLO）")
            dlg.resize(860, 560)
            v = QVBoxLayout(dlg)
            h1 = QHBoxLayout()
            h1.addWidget(QLabel("模型："))
            ed_m = QLineEdit(str(self.settings.get("model") or ""))
            h1.addWidget(ed_m, 1)
            b_m = QPushButton("选…")
            h1.addWidget(b_m)
            v.addLayout(h1)
            h2 = QHBoxLayout()
            h2.addWidget(QLabel("conf"))
            ed_c = QLineEdit(str(self.settings.get("model_conf") or "0.35"))
            ed_c.setMaximumWidth(60)
            ed_c.setToolTip("识别置信度阈值。实测：0.25 时框最全但误检多，0.35 精确率约 +3~4 个点、\n"
                            "漏检几乎不变，是推荐值；怕漏就调回 0.25。")
            h2.addWidget(ed_c)
            h2.addWidget(QLabel("imgsz"))
            ed_i = QLineEdit("2560")          # 和训练用的尺寸一致，别改小
            ed_i.setMaximumWidth(70)
            h2.addWidget(ed_i)
            h2.addWidget(QLabel("NMS"))
            ed_n = QLineEdit(str(self.settings.get("model_iou") or "0.4"))
            ed_n.setMaximumWidth(60)
            ed_n.setToolTip("NMS 的 IoU 阈值（去重强度）。默认 0.4 —— 实测同一板条被检出两个重叠框的\n"
                            "情况能一次清掉，精确率 +6~7 个点，召回只掉 1 个点；\n"
                            "0.7（旧默认）几乎不去重。想更狠可以 0.3，怕误删相邻板条就用 0.5。")
            h2.addWidget(ed_n)
            # 模型旁边有 meta.txt 就按它填，省得记错（选模型时还会再刷一次）
            _m0 = model_meta(ed_m.text().strip())
            if _m0.get("imgsz"):
                ed_i.setText(_m0["imgsz"])
                ed_i.setToolTip("这个值是从模型旁边的 meta.txt 读出来的，别乱改")
            chk_clean = QCheckBox("识别后清理多余框")
            chk_clean.setChecked(True)
            chk_clean.setToolTip("删掉：已经有手工框的地方、同一个编号重复的、互相重叠 90% 以上的、"
                                 "形状离谱的 Tracker。只删模型画的框，手工框不动。")
            h2.addWidget(chk_clean)
            h2.addWidget(QLabel("范围"))
            cmb_scope = QComboBox()
            cmb_scope.addItems(["本页", "整册", "指定页码"])
            cmb_scope.setCurrentIndex(1 if whole else 0)
            h2.addWidget(cmb_scope)
            ed_pages = QLineEdit()
            ed_pages.setPlaceholderText("例：3-8,12（选「指定页码」时用）")
            ed_pages.setMaximumWidth(200)
            ed_pages.setEnabled(False)
            ed_pages.setToolTip("只识别这几页；页码是 JSON 里的页号。"
                                "支持 3-8 区间、逗号分隔，例如 3-8,12")
            h2.addWidget(ed_pages)
            cmb_scope.currentIndexChanged.connect(
                lambda i: ed_pages.setEnabled(i == 2))
            h2.addStretch(1)
            v.addLayout(h2)
            from PySide6.QtWidgets import QProgressBar
            hp = QHBoxLayout()
            lbl_prog = QLabel("识别进度：还没开始")
            hp.addWidget(lbl_prog)
            prog = QProgressBar()
            prog.setRange(0, 100)
            prog.setValue(0)
            hp.addWidget(prog, 1)
            v.addLayout(hp)
            log = QPlainTextEdit()
            log.setReadOnly(True)
            v.addWidget(log, 1)
            hb = QHBoxLayout()
            b_run = QPushButton("开始识别")
            b_cl = QPushButton("关闭")
            hb.addWidget(b_run)
            hb.addStretch(1)
            hb.addWidget(b_cl)
            v.addLayout(hb)
            dpi = self.dpi if self.dpi > 0 else 250
            # 让外部 python 干活的脚本（只干一件事：给一张图，吐一行 JSON 框）
            script = os.path.join(tempfile.gettempdir(), "lbd_predict.py")
            try:
                with open(script, "w", encoding="utf-8") as f:
                    f.write(
                        "import sys, json\n"
                        "from ultralytics import YOLO\n"
                        "m = YOLO(sys.argv[1])\n"
                        "iou = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7\n"
                        "r = m.predict(source=sys.argv[2], imgsz=int(sys.argv[4]),\n"
                        "              conf=float(sys.argv[3]), iou=iou, max_det=1000,\n"
                        "              verbose=False)[0]\n"
                        "out = []\n"
                        "names = getattr(r, 'names', None) or {}\n"
                        "for c, xy, cf in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(),\n"
                        "                     r.boxes.conf.tolist()):\n"
                        "    out.append([str(names.get(int(c), c)), float(xy[0]), float(xy[1]),\n"
                        "                float(xy[2]), float(xy[3]), float(cf)])\n"
                        "print('@@' + json.dumps(out))\n")
            except Exception as e:
                QMessageBox.warning(self, "建脚本失败", "%s" % e)
                return

            class _Sig(_QO):
                res = _SIG(int, str, str)      # 页号, 图片路径, JSON 框
                msg = _SIG(str)
                prog = _SIG(int, int)          # 已完成页数, 总页数

            em = _Sig()
            em.msg.connect(lambda s: (log.appendPlainText(s),
                                      log.clear() if log.blockCount() > 3000 else None))
            def _on_prog(done, total):
                prog.setRange(0, max(1, total))
                prog.setValue(done)
                lbl_prog.setText("识别进度：%d/%d 页（%.0f%%）"
                                 % (done, total, 100.0 * done / max(1, total)))
            em.prog.connect(_on_prog)
            labels = ["Node", "Tracker", "Box"]

            def _apply(pg, img_path, js):
                def _map_label(nm):
                    """按模型的类别名映射到工具里的标签（模型可能是 {0:Tracker,1:Node}）"""
                    s = str(nm).strip().lower()
                    if s.isdigit():                      # 老格式：只有索引
                        i = int(s)
                        return labels[i] if 0 <= i < len(labels) else "Tracker"
                    if "node" in s:
                        return "Node"
                    if "box" in s:
                        return "Box"
                    return "Tracker"                     # tracker / typical / rack 都归 Tracker

                try:
                    boxes = json.loads(js) if js else []
                except Exception:
                    boxes = []
                # 类别名：优先用模型旁边的 classes.txt
                nonlocal labels
                if labels == ["Node", "Tracker", "Box"]:
                    try:
                        ct = os.path.join(os.path.dirname(ed_m.text().strip()), "classes.txt")
                        if os.path.exists(ct):
                            with open(ct, encoding="utf-8") as f:
                                labels = [x.strip() for x in f.read().splitlines() if x.strip()]
                    except Exception:
                        pass
                pm = self.pm if pg == self.page else (self.edited.get(pg)
                                                      or PageModel(self.dbg, pg))
                try:
                    from PySide6.QtGui import QImage as _QI
                    _q = _QI(img_path)
                    iw, ih = _q.width(), _q.height()
                except Exception:
                    iw, ih = pm.width, pm.height
                iw = iw or pm.width
                ih = ih or pm.height
                sx = float(pm.width) / float(iw)
                # 纵向必须单独算：以前图省事让 sy = sx，只要渲染图和页面的宽高比
                # 差一点点（换渲染器、吃了 CropBox、DPI 取整），纵向就会累积成
                # "整体往下/往上偏"，越靠下越明显。宽高比一致时两者本来就相等。
                sy = float(pm.height) / float(ih)
                if abs(sx - sy) > 0.002:
                    em.msg.emit("    ⚠ 底图和页面比例不一致：宽 ×%.4f / 高 ×%.4f"
                                "（底图 %d×%d，页面 %d×%d）—— 框会纵向偏移，"
                                "多半是渲染器和做标注时用的不是同一个"
                                % (sx, sy, iw, ih, pm.width, pm.height))
                if not boxes:
                    em.msg.emit("第 %s 页：没检出框" % pg)
                    return
                for c, x1, y1, x2, y2, cf in boxes:
                    lab = _map_label(c)
                    pm.shapes.append({"label": lab, "name": "", "confidence": cf,
                                      "class_id": DEFAULT_CLASS_ID.get(lab, 0),
                                      "source": "model", "raw": {"conf": cf},
                                      "ocr_index": None,
                                      "bbox": [x1 * sx, y1 * sy, x2 * sx, y2 * sy]})
                n_clean = 0
                if chk_clean.isChecked():
                    kept, st = clean_shapes(pm.shapes)
                    n_clean = st["removed"]
                    if n_clean:
                        pm.shapes = kept
                pm.dirty = True
                self.edited[pg] = pm
                if pg == self.page:
                    self._rebuild_items()
                em.msg.emit("第 %s 页：加进来 %d 个框%s"
                            % (pg, len(boxes), "，清理掉 %d 个多余的" % n_clean if n_clean else ""))

            em.res.connect(_apply)

            def work():
                idx = cmb_scope.currentIndex()
                if idx == 0:
                    pages = [self.page]
                elif idx == 1:
                    # 整册 = 文档里的**所有页**（识别模型本来就是给"还没框的图"找框的，
                    # 不能只挑"已经画过 Node 的页"，否则一张没标过的图会显示 0 页）
                    pages = sorted(self.dbg.page_numbers())
                else:
                    pages = parse_page_spec(ed_pages.text(), self.dbg.page_numbers())
                    if not pages:
                        em.msg.emit("「指定页码」里没填有效的页号。例：3-8,12"
                                    "（页码用 JSON 里的页号，整册是 %s~%s）"
                                    % (self.dbg.page_numbers()[0],
                                       self.dbg.page_numbers()[-1]))
                        return
                model = ed_m.text().strip()
                conf = ed_c.text().strip() or "0.25"
                imgsz = ed_i.text().strip() or "1920"
                iou = ed_n.text().strip() or "0.4"
                em.msg.emit("开始：%d 页，模型 %s" % (len(pages), os.path.basename(model)))
                em.prog.emit(0, len(pages))
                for n, pg in enumerate(pages):
                    em.msg.emit("[%d/%d] 第 %s 页 …" % (n + 1, len(pages), pg))
                    try:
                        img = self.renderer.render(self.pdf, pg, dpi)
                        # 先平滑缩到训练时的尺寸再喂模型：直接喂大图会让细线在
                        # ultralytics 内部的 cv2 降采样里丢掉（见 model_input_image）
                        img_in = model_input_image(img)
                        pr = subprocess.run([pyp, script, model, img_in, conf, imgsz, iou],
                                            capture_output=True, text=True, encoding="utf-8",
                                            errors="replace", timeout=3600,
                                            creationflags=_NO_WINDOW)
                        js = ""
                        for line in (pr.stdout or "").splitlines():
                            if line.startswith("@@"):
                                js = line[2:]
                        if not js:
                            em.msg.emit("    没拿到结果：" + ((pr.stderr or "").strip()[-300:]))
                        em.res.emit(pg, img_in, js)
                    except Exception as e:
                        em.msg.emit("    第 %s 页出错：%s" % (pg, e))
                    em.prog.emit(n + 1, len(pages))
                em.msg.emit("识别结束")

            def do_run():
                if not (ed_m.text().strip() and os.path.exists(ed_m.text().strip())):
                    log.appendPlainText("先选模型文件（best.pt）")
                    return
                self.settings["model"] = ed_m.text().strip()
                self.settings["model_conf"] = ed_c.text().strip() or "0.35"
                self.settings["model_iou"] = ed_n.text().strip() or "0.4"
                save_settings(self.settings)
                threading.Thread(target=work, daemon=True).start()

            def do_pick_model():
                f, _x = QFileDialog.getOpenFileName(dlg, "选模型", "",
                                                    "模型 (*.pt *.onnx)")
                if f:
                    ed_m.setText(f)
                    _m = model_meta(f)
                    if _m.get("imgsz"):
                        ed_i.setText(_m["imgsz"])
                        log.appendPlainText("按模型旁边的 meta.txt 把 imgsz 设成 %s"
                                            % _m["imgsz"])
                    else:
                        log.appendPlainText("提示：这个模型旁边没有 meta.txt，"
                                            "imgsz 要自己填对（和训练时一致）")

            b_m.clicked.connect(do_pick_model)
            b_run.clicked.connect(do_run)
            b_cl.clicked.connect(dlg.accept)
            dlg.exec()

        def on_train_panel(self):
            """训练环境面板：探测 Python / 检测 ultralytics / 一键装（日志实时显示）。"""
            from PySide6.QtWidgets import (QDialog, QPlainTextEdit, QVBoxLayout, QHBoxLayout,
                                           QPushButton, QLabel, QComboBox, QLineEdit)
            import subprocess
            import threading
            dlg = QDialog(self)
            dlg.setWindowTitle("训练环境（先装环境，下一步接一键训练）")
            dlg.resize(780, 500)
            v = QVBoxLayout(dlg)
            h = QHBoxLayout()
            h.addWidget(QLabel("Python 解释器："))
            cmb = QComboBox()
            for p in find_pythons():
                cmb.addItem(p)
            if cmb.count() == 0:
                cmb.addItem("（没找到 Python，点右边「浏览…」）")
            if self.settings.get("python"):
                cmb.addItem(str(self.settings["python"]))
                cmb.setCurrentText(str(self.settings["python"]))
            h.addWidget(cmb, 1)
            b_br = QPushButton("浏览…")
            h.addWidget(b_br)
            v.addLayout(h)
            h1m = QHBoxLayout()
            h1m.addWidget(QLabel("模型："))
            ed_model = QComboBox()
            ed_model.setEditable(True)
            ed_model.addItems(["yolov8n.pt", "yolov8s.pt", "yolo26n.pt", "yolo26s.pt"])
            ed_model.setCurrentText(str(self.settings.get("train_model") or "yolov8n.pt"))
            h1m.addWidget(ed_model, 1)
            b_mo = QPushButton("选…")
            h1m.addWidget(b_mo)
            v.addLayout(h1m)
            from PySide6.QtWidgets import QCheckBox
            chk_resume = QCheckBox("续训(resume)：用 last.pt 接着原来那次跑")
            chk_resume.setToolTip("勾上后 data.yaml / epochs / imgsz / batch / device 都会被忽略"
                                  "（resume 会沿用原来那次的设置和剩余轮数）")
            v.addWidget(chk_resume)
            lbl = QLabel("Python 3.14 装不上 torch —— 请选 3.11 / 3.12。"
                         "这台机器是 GT 1030(2GB)+老驱动，只能跑 CPU 版。")
            lbl.setWordWrap(True)
            v.addWidget(lbl)
            log = QPlainTextEdit()
            log.setReadOnly(True)
            v.addWidget(log, 1)
            hv = QHBoxLayout()
            hv.addWidget(QLabel("data.yaml："))
            ed_data = QLineEdit()
            hv.addWidget(ed_data, 1)
            b_d = QPushButton("选…")
            hv.addWidget(b_d)
            v.addLayout(hv)
            hv2 = QHBoxLayout()
            for _t, _w, _d in (("epochs", 60, "20"), ("imgsz", 70, "1280"),
                               ("batch", 50, "4"), ("device", 70, "cpu")):
                hv2.addWidget(QLabel(_t))
                _e = QLineEdit(_d)
                _e.setMaximumWidth(_w)
                hv2.addWidget(_e)
                if _t == "epochs":
                    ed_ep = _e
                elif _t == "imgsz":
                    ed_im = _e
                elif _t == "batch":
                    ed_ba = _e
                else:
                    ed_dv = _e
            hv2.addStretch(1)
            v.addLayout(hv2)
            hb = QHBoxLayout()
            b1 = QPushButton("检测环境")
            b2 = QPushButton("一键装环境(CPU)")
            b4 = QPushButton("开始训练")
            b3 = QPushButton("关闭")
            hb.addWidget(b1)
            hb.addWidget(b2)
            hb.addWidget(b4)
            hb.addStretch(1)
            hb.addWidget(b3)
            v.addLayout(hb)

            def cur_py():
                t = cmb.currentText().strip()
                return t if (t and os.path.exists(t)) else None

            def run(args, tag):
                log.appendPlainText("$ " + " ".join(args))

                # ★ 子线程绝对不能直接动 Qt 控件（ultralytics 每秒吐上百行，直接 append
                #   会把 Qt 刷崩 = 闪退）。改用信号：子线程 emit，GUI 线程 append。
                from PySide6.QtCore import QObject as _QO, Signal as _SIG

                class _Emit(_QO):
                    sig = _SIG(str)

                em = _Emit()

                def _add(s):
                    log.appendPlainText(s)
                    if log.blockCount() > 3000:      # 别让日志无限长
                        log.clear()

                em.sig.connect(_add)

                def work():
                    try:
                        pr = subprocess.Popen(args, stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, text=True,
                                              encoding="utf-8", errors="replace",
                                              creationflags=_NO_WINDOW)
                        for line in pr.stdout:
                            em.sig.emit(line.rstrip())
                        pr.wait()
                        em.sig.emit("[%s] 结束，退出码 %s" % (tag, pr.returncode))
                    except Exception as e:
                        em.sig.emit("[%s] 出错：%s" % (tag, e))
                threading.Thread(target=work, daemon=True).start()

            def do_check():
                p = cur_py()
                if not p:
                    lbl.setText("先选一个有效的 python.exe（「浏览…」）")
                    return
                self.settings["python"] = p
                save_settings(self.settings)
                lbl.setText("检测中…（看下面日志）")
                run([p, "-c", "import sys;print('PY',sys.version.split()[0]);"
                              "import ultralytics;print('UL',ultralytics.__version__)"], "检测")

            def do_install():
                p = cur_py()
                if not p:
                    lbl.setText("先选一个有效的 python.exe（「浏览…」）")
                    return
                self.settings["python"] = p
                save_settings(self.settings)
                lbl.setText("正在装 ultralytics（CPU 版 torch，几分钟）")
                run([p, "-m", "pip", "install", "-U", "ultralytics"], "安装")

            def do_browse():
                p, _f = QFileDialog.getOpenFileName(dlg, "选 python.exe", "C:\\",
                                                    "python.exe (python.exe)")
                if p:
                    cmb.addItem(p)
                    cmb.setCurrentText(p)

            b1.clicked.connect(do_check)
            b2.clicked.connect(do_install)
            b3.clicked.connect(dlg.accept)
            b_br.clicked.connect(do_browse)
            def do_pick_data():
                f, _x = QFileDialog.getOpenFileName(dlg, "选 data.yaml（导出数据集时生成的）",
                                                    "", "YAML (*.yaml *.yml)")
                if f:
                    ed_data.setText(f)

            def do_train():
                p = cur_py()
                if not p:
                    lbl.setText("先选一个有效的 python.exe（「浏览…」）")
                    return
                d = ed_data.text().strip()
                if not (d and os.path.exists(d)):
                    lbl.setText("先选 data.yaml —— 就是「导出 YOLO 数据集…」生成的那个")
                    return
                mdl = ed_model.currentText().strip() or "yolov8n.pt"
                mdl_local = os.path.exists(mdl)
                if (os.path.sep in mdl or "/" in mdl) and not mdl_local:
                    lbl.setText("模型文件不存在：%s" % mdl)
                    return
                self.settings["train_model"] = mdl
                save_settings(self.settings)
                if chk_resume.isChecked():
                    # 续训只能用那次训练的 last.pt —— 用别的（比如默认的 yolov8n.pt）
                    # ultralytics 拿不到原来那次的设置，会按默认值跑 coco8（4 张示例图），
                    # 结果就是"训练完了但学的是猫狗"。
                    if not mdl_local:
                        lbl.setText("续训要选那次训练留下的 last.pt（runs/detect/<名字>/"
                                    "weights/last.pt），不能用 %s" % mdl)
                        return
                    if os.path.basename(mdl).lower() != "last.pt":
                        QMessageBox.warning(
                            self, "续训要选 last.pt",
                            "续训（resume）只有那次训练留下的 weights/last.pt 才能接着跑，"
                            "你现在选的是：\n%s\n\n"
                            "用别的模型（比如默认的 yolov8n.pt、或者 best.pt）时，"
                            "ultralytics 读不到「原来那次」的设置，会拿它自带的 coco8"
                            "（4 张示例图：人/狗/马…）按默认参数跑一遍 —— "
                            "看着像训练成功了，其实和你的数据无关。\n\n"
                            "要接着自己的数据练，请：\n"
                            "· 续训：选 runs/detect/<那次的名字>/weights/last.pt\n"
                            "· 或取消勾选「续训」，用下面的 data.yaml + 模型正常开一轮"
                            % mdl)
                        return
                    code = ("from ultralytics import YOLO;"
                            "YOLO(r'%s').train(resume=True)" % mdl)
                    lbl.setText("续训：用 last.pt 接着原来那次跑"
                                "（data/epochs 这些参数会被忽略）")
                else:
                    names, root = peek_data_yaml(d)
                    ep, im = ed_ep.text().strip() or "20", ed_im.text().strip() or "1280"
                    ba, dv = ed_ba.text().strip() or "4", ed_dv.text().strip() or "cpu"
                    if QMessageBox.question(
                            self, "开始训练前确认",
                            "模型：%s%s\n数据：%s\n类别：%s\n"
                            "轮数 %s　imgsz %s　batch %s　device %s\n\n开始训练吗？"
                            % (mdl, "" if mdl_local else "（本地没有，会自动下载）", d,
                               ("%d 类：%s" % (len(names), "、".join(names[:8])))
                               if names else "（读不出类别名，自己确认一下这份 data.yaml）",
                               ep, im, ba, dv),
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes) != QMessageBox.StandardButton.Yes:
                        lbl.setText("已取消")
                        return
                    code = ("from ultralytics import YOLO;"
                            "YOLO(r'%s').train(data=r'%s', imgsz=%s, epochs=%s, "
                            "batch=%s, device='%s')"
                            % (mdl, d, im, ep, ba, dv))
                lbl.setText("训练已启动（日志在下面滚；权重在 runs/detect/train/weights/best.pt）")
                run([p, "-c", code], "训练")

            b4.clicked.connect(do_train)
            b_d.clicked.connect(do_pick_data)
            def do_pick_model_file():
                f, _x = QFileDialog.getOpenFileName(
                    dlg, "选模型（默认 yolov8n.pt；也可以选自己训好的 best.pt）",
                    "", "模型 (*.pt)")
                if f:
                    ed_model.setCurrentText(f)

            b_mo.clicked.connect(do_pick_model_file)
            dlg.exec()

        def on_check_update(self):
            """联网查 Release：有新版本就给下载/打开网页的入口。

            整个检查（连读本机凭据）都在后台线程里做，界面只弹一个能取消的忙等框：
            以前是先设全局等待光标、再在后台请求，网络一慢光标就转半天，看着像卡死；
            而且读凭据那一步还在主线程上，凭据库慢的时候界面真的会僵住。
            只读公开 API，不写任何文件（下载是你点了才做）。
            """
            from PySide6.QtWidgets import QProgressDialog
            from PySide6.QtCore import QTimer
            repo = str(self.settings.get("update_repo") or UPDATE_REPO)
            page = "https://github.com/%s/releases/latest" % repo
            prog = QProgressDialog("正在检查更新（%s）…\n本工具 v%s"
                                   % (repo, ANNOTATOR_VERSION), "取消", 0, 0, self)
            prog.setWindowTitle("检查更新（本工具 v%s）" % ANNOTATOR_VERSION)
            prog.setMinimumDuration(0)
            prog.setWindowModality(Qt.WindowModality.WindowModal)
            self.statusBar().showMessage("正在检查更新（%s）…" % repo, 0)
            st = {"cancel": False, "done": False}
            t0 = time.time()

            # 每秒把"已等几秒"写进忙等框：卡住时至少看得出在等什么
            tick = QTimer(self)
            tick.setInterval(1000)
            tick.timeout.connect(
                lambda: prog.setLabelText(
                    "正在检查更新（%s）…\n本工具 v%s    已等 %.0f 秒%s"
                    % (repo, ANNOTATOR_VERSION, time.time() - t0,
                       ("（第 %d 轮重试）" % state["round"]) if state["round"] > 1 else "")
                    + "\n" + (state["step"] or "正在选通道…")))
            tick.start()

            def stop_busy():
                tick.stop()
                prog.close()

            def on_cancel():
                # 取消要真的把框关掉：以前只设了个标记，框会一直挂在屏幕上
                if st["done"] or st["cancel"]:
                    return
                st["cancel"] = True
                stop_busy()
                self.statusBar().showMessage("已取消检查更新", 3000)

            prog.canceled.connect(on_cancel)

            def on_timeout():
                # 硬超时：不管卡在 DNS、代理还是凭据，最多等 15 秒就给结论
                if st["done"] or st["cancel"]:
                    return
                st["cancel"] = True
                stop_busy()
                b = QMessageBox(self)
                b.setWindowTitle("检查更新超时")
                b.setIcon(QMessageBox.Icon.Warning)
                b.setText("等了 %d 秒还没拿到 GitHub 的回复，先不查了。"
                          % int(time.time() - t0))
                b.setInformativeText(
                    "多半是防火墙/杀毒软件拦了本工具的外连（打包成 exe 很常见）。\n"
                    "工具已经试过自己发请求和系统 curl.exe 两条路，都被拦。\n"
                    "点下面的按钮用浏览器打开下载页，那里能直接下新版本。")
                b.setDetailedText("下载页：%s" % page)
                b_open = b.addButton("用浏览器打开下载页",
                                     QMessageBox.ButtonRole.AcceptRole)
                b_diag = b.addButton("运行诊断（把结果发我）",
                                     QMessageBox.ButtonRole.ActionRole)
                b.addButton("关闭", QMessageBox.ButtonRole.RejectRole)
                b.exec()
                if b.clickedButton() is b_diag:
                    self.on_update_diagnose()
                    return
                if b.clickedButton() is b_open:
                    self._open_url(page)

            hard = QTimer(self)
            hard.setSingleShot(True)
            hard.timeout.connect(on_timeout)
            # curl 8s + urllib 8s = 最多 ~16s，硬超时给到 25s（以前 15s 会把 curl 那条路
            # 直接掐掉，表现就是"一直卡着检查、最后什么也没更新"）
            # 正常路径（git + HEAD）2~4 秒就回来了；最坏是"git 15s + HEAD 13s +
            # API 4 次 ×6s"≈52s，所以硬超时给 60s（框上有"已等 N 秒"和取消按钮）
            hard.start(60000)

            from PySide6.QtCore import QObject as _QO2, Signal as _SIG2

            class _Sig(_QO2):
                done = _SIG2(object)
                round = _SIG2(int)              # 第几轮重试（网络不通时让人看到在重试）
                step = _SIG2(str)               # 正在试哪条通道（git / HEAD / api）

            sig = _Sig()
            state = {"round": 1}
            state["step"] = ""

            def _on_round(rd):
                state["round"] = rd

            sig.round.connect(_on_round)
            sig.step.connect(lambda s: state.__setitem__("step", s))

            def worker():
                try:
                    tok = update_token(self.settings)      # CredRead 也可能慢，别放主线程
                    px = update_proxy(self.settings)
                    info, err = fetch_latest_release(
                        repo, tok, timeout=6, rounds=2,
                        on_round=lambda rd: sig.round.emit(rd), proxy=px,
                        on_step=lambda s: sig.step.emit(s))
                except Exception as e:                     # noqa: BLE001
                    info, err = None, "%s" % e
                sig.done.emit((info, err))

            def back(res):
                if st["done"] or st["cancel"]:
                    return                                 # 已经取消/超时了，结果丢掉
                st["done"] = True
                hard.stop()
                stop_busy()
                self.statusBar().showMessage("", 0)
                info, err = res if isinstance(res, tuple) else (None, "未知错误")
                self._show_update_result(info, err, repo, time.time() - t0)

            sig.done.connect(back)
            threading.Thread(target=worker, daemon=True).start()

        def _show_update_result(self, info, err, repo, elapsed=0.0):
            page = "https://github.com/%s/releases/latest" % repo
            if err or not info:
                b = QMessageBox(self)
                b.setWindowTitle("检查更新失败")
                b.setIcon(QMessageBox.Icon.Warning)
                b.setText("等了 %.0f 秒没拿到结果。" % elapsed)
                b.setInformativeText(
                    "试过三条路：① git ls-remote（和自己 clone 同一条通道）、"
                    "② 直接 HEAD 安装包地址（github.com）、③ api.github.com。\n"
                    "三条都不通，说明这台机器到 github.com 的网络被拦或不稳。\n"
                    "· 能正常 clone 这个仓库的机器上会用本机 git 凭据，一般不用管；\n"
                    "· 也可以在 annotator_settings.json 里加 \"update_token\"（只读 token）"
                    "或 \"update_proxy\"（如 http://127.0.0.1:7890）；\n"
                    "· 最稳的是点下面的按钮用浏览器打开下载页手动下（浏览器多半有代理/VPN）。")
                b.setDetailedText("错误：%s\n\n下载页：%s" % (err, page))
                b_open = b.addButton("用浏览器打开下载页",
                                     QMessageBox.ButtonRole.AcceptRole)
                b_diag = b.addButton("运行诊断（把结果发我）",
                                     QMessageBox.ButtonRole.ActionRole)
                b.addButton("关闭", QMessageBox.ButtonRole.RejectRole)
                b.exec()
                if b.clickedButton() is b_diag:
                    self.on_update_diagnose()
                    return
                if b.clickedButton() is b_open:
                    self._open_url(page)
                return
            tag = (info.get("tag_name") or "").strip()
            cur, new = parse_version(ANNOTATOR_VERSION), parse_version(tag)
            exe = [info.get("_lbd_asset")] if info.get("_lbd_asset") else []
            date = (info.get("published_at") or "")[:10]
            if new and cur and new > cur:
                lines = ["当前版本：v%s      最新版本：%s（%s）" % (ANNOTATOR_VERSION, tag, date)]
                if exe:
                    lines.append("安装包：%s（%.1f MB）"
                                 % (exe[0].get("name"),
                                    (exe[0].get("size") or 0) / 1048576.0))
                note = (info.get("body") or "").strip()
                if note:
                    lines.append("")
                    lines.append(note[:900] + ("…" if len(note) > 900 else ""))
                box = QMessageBox(self)
                box.setWindowTitle("发现新版本 %s" % tag)
                box.setIcon(QMessageBox.Icon.Information)
                head = ("发现新版本：%s（%s）\n当前版本：v%s\n"
                        % (tag, date, ANNOTATOR_VERSION))
                head += "（检查用时 %.1f 秒）\n" % elapsed
                if exe:
                    head += "安装包：%s（%.1f MB）\n" % (
                        exe[0].get("name"), (exe[0].get("size") or 0) / 1048576.0)
                else:
                    if info.get("_from") == "git":
                        head += ("（GitHub 的 API 这次没通，是用 git 查到的最新 tag ——"
                                 "拿不到安装包信息）\n")
                    else:
                        head += "⚠ 这个版本还没挂可下载的 exe（可能上传还没完成）\n"
                    head += "点「打开下载页」在浏览器里手动下。\n"
                head += "\n点「下载到程序目录」会存一份新 exe 到程序旁边（不自动覆盖）；\n" \
                        "点「打开下载页」就是浏览器里手动下。"
                box.setText(head)
                note = (info.get("body") or "").strip()
                if note:
                    box.setDetailedText(note)
                b_dl = box.addButton("下载到程序目录", QMessageBox.ButtonRole.AcceptRole)
                b_web = box.addButton("打开下载页", QMessageBox.ButtonRole.ActionRole)
                box.addButton("以后再说", QMessageBox.ButtonRole.RejectRole)
                box.exec()
                hit = box.clickedButton()
                if hit is b_web or (hit is None and not exe):
                    self._open_url(page)
                elif hit is b_dl:
                    self._download_update(exe[0] if exe else None, tag, page)
                return
            if new and cur and new < cur:
                QMessageBox.information(
                    self, "检查更新",
                    "当前 v%s 比 Release 里最新的 %s 还新（本地开发版）。\n\n"
                    "（检查用时 %.1f 秒）" % (ANNOTATOR_VERSION, tag, elapsed))
                return
            QMessageBox.information(
                self, "已是最新版本",
                "当前 v%s 就是最新版本（远端 %s，%s）。\n\n（检查用时 %.1f 秒）"
                % (ANNOTATOR_VERSION, tag, date, elapsed))

        def on_update_diagnose(self):
            """一键跑「更新诊断」：DNS / curl / git / 安装包地址 / api 各测一次，
            结果显示出来（可一键复制）并存到程序目录的 update_check_report.txt。"""
            from PySide6.QtWidgets import QProgressDialog
            import threading
            from PySide6.QtCore import QObject as _QOd, Signal as _SIGd
            prog = QProgressDialog("正在跑更新诊断…（每一步都会写在这里）",
                                   "取消", 0, 7, self)
            prog.setWindowTitle("更新诊断（本工具 v%s）" % ANNOTATOR_VERSION)
            prog.setMinimumDuration(0)
            prog.setWindowModality(Qt.WindowModality.WindowModal)

            class _Sigd(_QOd):
                done = _SIGd(object)
                step = _SIGd(str, int)

            sg = _Sigd()
            _n = {"i": 0}

            def on_step(name):
                _n["i"] += 1
                sg.step.emit(str(name), _n["i"])

            def _show_step(name, i):
                prog.setLabelText("正在跑更新诊断…\n第 %d 步：%s\n（卡住的话就是这一步）"
                                  % (i, name))
                prog.setValue(i - 1)

            sg.step.connect(_show_step)

            def worker():
                try:
                    rep = update_diagnose(settings=self.settings, on_step=on_step)
                except Exception as e:                 # noqa: BLE001
                    rep = "诊断本身出错：%r" % e
                sg.done.emit(rep)

            def back(rep):
                prog.close()
                path = os.path.join(app_dir(), "update_check_report.txt")
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(rep)
                except Exception:
                    path = "（写不进程序目录）"
                box = QMessageBox(self)
                # 直接拿记事本打开报告文件（比弹一个大文本框友好，也不会让人以为卡住），
                # 打不开才退回弹窗
                opened = False
                try:
                    os.startfile(path)                 # noqa: S606
                    opened = True
                    self.statusBar().showMessage("诊断报告已打开（文件：%s）" % path, 12000)
                except Exception:
                    opened = False
                if not opened:
                    box = QMessageBox(self)
                    box.setWindowTitle("更新诊断（把这段发我）")
                    box.setIcon(QMessageBox.Icon.Information)
                    box.setText(rep)
                    box.setDetailedText(rep + "\n\n（也写到了：%s）" % path)
                    b_copy = box.addButton("复制到剪贴板", QMessageBox.ButtonRole.ActionRole)
                    box.addButton("关闭", QMessageBox.ButtonRole.RejectRole)
                    box.exec()
                    if box.clickedButton() is b_copy:
                        QApplication.clipboard().setText(rep)
                        self.statusBar().showMessage("诊断报告已复制到剪贴板", 5000)

            sg.done.connect(back)
            prog.show()
            threading.Thread(target=worker, daemon=True).start()

        def _open_url(self, url):
            try:
                import webbrowser
                webbrowser.open(url)
            except Exception:                       # noqa: BLE001
                QMessageBox.information(self, "下载地址", url)

        def _download_update(self, asset, tag, page):
            """把新版本 exe 下载到程序目录（不自动覆盖正在运行的自己）。"""
            import urllib.request
            import urllib.parse
            from PySide6.QtWidgets import QProgressDialog

            class StripAuthRedirect(urllib.request.HTTPRedirectHandler):
                """跨域名跳转（GitHub → S3 签名地址）时摘掉 Authorization，否则下载被拒。"""

                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    new = super().redirect_request(req, fp, code, msg, headers, newurl)
                    try:
                        if new is not None and (
                                urllib.parse.urlsplit(newurl).netloc
                                != urllib.parse.urlsplit(req.full_url).netloc):
                            new.headers.pop("Authorization", None)
                            new.unredirected_hdrs.pop("Authorization", None)
                    except Exception:               # noqa: BLE001
                        pass
                    return new
            url = (asset or {}).get("browser_download_url") or ""
            if not url:
                self._open_url(page)
                return
            _proxy = update_proxy(self.settings)
            total = int((asset or {}).get("size") or 0)
            name = "LBD标注工具_%s.exe" % (tag or "new")
            dest = os.path.join(app_dir(), name)
            if not os.access(app_dir(), os.W_OK):
                dest = os.path.join(os.path.expanduser("~"), "Downloads", name)
            prog = QProgressDialog("正在下载 %s …" % name, "取消", 0, 100, self)
            prog.setWindowTitle("下载更新")
            prog.setMinimumDuration(0)
            prog.setWindowModality(Qt.WindowModality.WindowModal)
            cancel = threading.Event()
            prog.canceled.connect(cancel.set)

            from PySide6.QtCore import QObject as _QO3, Signal as _SIG3

            class _Sig(_QO3):
                step = _SIG3(int, int)      # 已下载, 总量
                done = _SIG3(str, str)      # 保存路径, 错误

            sig = _Sig()

            def progress(got, all_bytes):
                if all_bytes > 0:
                    prog.setValue(min(100, int(got * 100 / all_bytes)))
                else:
                    prog.setLabelText("已下载 %.1f MB…" % (got / 1048576.0))

            def finished(path, err):
                prog.close()
                if err:
                    QMessageBox.warning(self, "下载失败", "%s\n\n也可以手动下载：\n%s" % (err, page))
                    return
                QMessageBox.information(
                    self, "下载完成",
                    "新版本已下载到：\n%s\n\n"
                    "关掉本工具后，把它改名成 LBD标注工具.exe（覆盖旧的）就是新版本了。\n"
                    "（正在运行的 exe 没法自己覆盖自己，所以留这一步手动操作）" % path)
                try:
                    subprocess.Popen('explorer /select,"%s"' % path)
                except Exception:                   # noqa: BLE001
                    pass

            sig.step.connect(progress)
            sig.done.connect(finished)

            def worker():
                tmp = dest + ".part"
                got = 0
                try:
                    # 私有仓库的附件：走 assets API + Accept: octet-stream（带 token），
                    # GitHub 会 302 到签名地址；公开仓库走普通下载地址也一样能用。
                    tok = update_token(self.settings)
                    api_url = (asset or {}).get("url") or ""
                    ua = "LBD-Annotator/%s" % ANNOTATOR_VERSION
                    use_api = bool(tok and api_url)
                    try:
                        if use_api:
                            req = urllib.request.Request(
                                api_url, headers={"User-Agent": ua,
                                                  "Accept": "application/octet-stream",
                                                  "Authorization": "Bearer %s" % tok})
                        else:
                            req = urllib.request.Request(url, headers={"User-Agent": ua})
                        _handlers = [StripAuthRedirect]
                        if _proxy:
                            _handlers.append(urllib.request.ProxyHandler(
                                {"http": _proxy, "https": _proxy}))
                        opener = urllib.request.build_opener(*_handlers)
                        with opener.open(req, timeout=30) as r:
                            if not total:
                                total_hint = int(r.headers.get("Content-Length") or 0)
                            else:
                                total_hint = total
                            with open(tmp, "wb") as f:
                                while True:
                                    if cancel.is_set():
                                        raise RuntimeError("已取消")
                                    chunk = r.read(262144)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                                    got += len(chunk)
                                    if got % 1048576 < 262144:
                                        sig.step.emit(got, total_hint)
                    except Exception as e1:          # noqa: BLE001
                        # 打包成 exe 后自己的外连常被防火墙/杀软拦掉，退回系统 curl.exe
                        sig.step.emit(got, total or 0)
                        import shutil as _sh
                        import subprocess as _sp
                        exe = _sh.which("curl") or _sh.which("curl.exe")
                        if not exe:
                            raise RuntimeError("%s（系统里也没有 curl.exe 可退）" % e1)
                        cfg = ('header = "User-Agent: %s"\n'
                               'header = "Accept: application/octet-stream"\n' % ua)
                        if use_api:
                            cfg = ('header = "User-Agent: %s"\n'
                                   'header = "Accept: application/octet-stream"\n'
                                   'header = "Authorization: Bearer %s"\n' % (ua, tok))
                        cfg += ('location\nfail\n'
                                'output = "%s"\n'
                                'url = "%s"\n' % (tmp, api_url if use_api else url))
                        _curl_args = [exe, "-sS", "--config", "-"]
                        if _proxy:
                            _curl_args += ["--proxy", _proxy]
                        pr = _sp.Popen(_curl_args,
                                       stdin=_sp.PIPE, stdout=_sp.PIPE,
                                       stderr=_sp.STDOUT, creationflags=_NO_WINDOW)
                        pr.stdin.write(cfg.encode("utf-8"))
                        pr.stdin.close()
                        while pr.poll() is None:       # 边下边报进度
                            if cancel.is_set():
                                pr.kill()
                                raise RuntimeError("已取消")
                            try:
                                got = os.path.getsize(tmp)
                            except OSError:
                                pass
                            sig.step.emit(got, total or 0)
                            time.sleep(0.5)
                        if pr.returncode != 0:
                            msg = (pr.stdout.read() or b"").decode("utf-8", "replace").strip()
                            raise RuntimeError("curl 下载失败（%s）：%s" % (pr.returncode, msg[:200]))
                    os.replace(tmp, dest)
                    sig.done.emit(dest, "")
                except Exception as e:              # noqa: BLE001
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except Exception:               # noqa: BLE001
                        pass
                    sig.done.emit("", "%s" % e)

            threading.Thread(target=worker, daemon=True).start()

        def _page_scale_info(self, pg, pm=None):
            """这一页的 (1 IN = ? FT, 哪来的, 每英寸多少像素)。整册分档要按页折算长度用。

            比例尺和"每英寸像素"两个都从**文字层**里读：
              · 比例尺 = 标题栏标尺表 / "1\" = 100'-0\"" 这种文字；
              · 每英寸像素 = 图纸幅面（"22\" x 34\" SHEETS"）反推，比依赖 DPI 设置可靠。
            读不到的（没选 PDF、图纸没有文字层）返回 (None, 原因, dpi)。
            """
            cache = getattr(self, "_scale_cache", None)
            if cache is None:
                cache = self._scale_cache = {}
            if pg in cache:
                return cache[pg]
            dpi = float(self.dpi) if self.dpi and self.dpi > 0 else 250.0
            items = []
            if self.pdf and os.path.exists(self.pdf):
                try:
                    if getattr(self, "_ptext_scale", None) is None:
                        self._ptext_scale = PdfText(self.pdf)
                    items = self._ptext_scale.items(pg)
                except Exception:
                    items = []
            ppi = dpi
            try:
                sheet_w = read_sheet_width_in(items)
                if sheet_w and pm is not None and getattr(pm, "width", 0):
                    ppi = float(pm.width) / float(sheet_w)
            except Exception:
                pass
            sc, why = read_page_scale(items)
            out = (sc, why, ppi)
            cache[pg] = out
            return out

        def on_rack_grade(self):
            """Tracker（支架）按长边长度分档：短的 2 串、长的 3 串（可改）→ 写进 raw.strings。

            分档用的是**整册所有页画过的支架**（不是只看当前页），
            这样每一页的"长/短"用的是同一套门槛，跨页也不会忽长忽短。

            长度先按**每一页自己的比例尺**折算成英尺再比。这本图纸每页比例尺不一样
            （第 1 页 1IN=100FT、第 2 页 1IN=120FT、第 30 页 1IN=80FT…），直接拿像素
            比长度会把同一种支架分成两类 —— 用户反馈的"明明差不多长却分成两种"里，
            有一部分就是这个原因造成的。
            """
            from PySide6.QtWidgets import QInputDialog
            if not self.dbg:
                return
            pages = self._page_order() or [self.page]
            pms = {}
            for pg in pages:
                pms[pg] = (self.pm if pg == self.page
                           else (self.edited.get(pg) or PageModel(self.dbg, pg)))
            racks = [(pg, s) for pg, pm in pms.items() for s in pm.shapes
                     if s.get("label") == "Tracker"]
            if not racks:
                QMessageBox.information(self, "没有支架框", "整册里都没有 Tracker（支架）框。")
                return
            # ---- 每页比例尺：1 IN = ? FT（从标题栏文字层读；读不到就用整册最常见的那个）
            from PySide6.QtWidgets import QProgressDialog
            prog = QProgressDialog("正在读每页比例尺（从图纸标题栏的文字层）…", "取消",
                                   0, len(pms), self)
            prog.setWindowTitle("支架按长度分档")
            prog.setMinimumDuration(0)
            prog.setWindowModality(Qt.WindowModality.WindowModal)
            scale_of, ppi_of, why_of = {}, {}, {}
            for kp, (pg, pm) in enumerate(pms.items()):
                prog.setValue(kp)
                prog.setLabelText("第 %s 页 比例尺（%d/%d）…" % (pg, kp + 1, len(pms)))
                QApplication.processEvents()
                if prog.wasCanceled():
                    prog.close()
                    return
                sc, why, ppi = self._page_scale_info(pg, pm)
                scale_of[pg], why_of[pg], ppi_of[pg] = sc, why, ppi
            prog.close()
            known = [v for v in scale_of.values() if v]
            dom = None
            if known:
                cnt = collections.Counter(round(v, 3) for v in known)
                dom = cnt.most_common(1)[0][0]
            no_scale = sorted(pg for pg in pms if not scale_of[pg])
            use_ft = bool(known)

            def length_of(pg, s):
                """这个支架有多长。有比例尺 -> 英尺；没有 -> 退回像素。"""
                L = max(s["bbox"][2] - s["bbox"][0], s["bbox"][3] - s["bbox"][1])
                if not use_ft:
                    return float(L), None
                sc = scale_of[pg] or dom
                ppi = ppi_of[pg] or 250.0
                return (L / ppi) * sc, sc

            longs = sorted(length_of(pg, s)[0] for pg, s in racks)
            # 按"长度差 ≤10%"聚类：差在 10% 以内的算同一类（不再等分位硬切）
            groups = []
            for L in longs:
                if groups:
                    m = sum(groups[-1]) / len(groups[-1])
                    if abs(L - m) <= RACK_LEN_TOL * m:
                        groups[-1].append(L)
                        continue
                groups.append([L])
            k0 = len(groups)
            unit = "英尺" if use_ft else "像素"
            if use_ft:
                sc_txt = "；".join(
                    "%s FT/IN×%d 页" % (s, c)
                    for s, c in collections.Counter(
                        round(scale_of[pg], 3) for pg in pms if scale_of[pg]
                    ).most_common())
                head_note = ("整册 %d 个支架框，长度已按每页比例尺折算成英尺"
                             "（本册比例尺：%s）\n" % (len(racks), sc_txt))
                if no_scale:
                    head_note += ("⚠ 这 %d 页没读到比例尺，按最常见的 %g FT/IN 算：%s\n"
                                  % (len(no_scale), dom, ",".join(str(x) for x in no_scale[:12])))
            else:
                head_note = ("整册 %d 个支架框（**没读到比例尺**，只能按像素长度分档 ——\n"
                             "  每页比例尺不一样时这样不准；先在「选 PDF…」选上 PDF，\n"
                             "  比例尺要从图纸标题栏的文字层里读）\n" % len(racks))
            txt, ok = QInputDialog.getText(
                self, "支架串数分档",
                head_note +
                "按「长度差 ≤%d%% 算同一类」分成 %d 类。\n"
                "按「短 → 长」填每类的串数（逗号分开）。\n"
                "档数跟上面的类数一致最准（长度差不多的支架保证在同一档）；\n"
                "填的档数不一样时，会按长度重新聚成你填的档数，不会切在长度接近的地方："
                % (int(RACK_LEN_TOL * 100), k0),
                text=",".join(str(2 + i) for i in range(k0)))
            if not ok or not txt.strip():
                return
            levels = sorted(int(x) for x in re.split(r"[^0-9]+", txt) if x.strip())
            if not levels:
                QMessageBox.warning(self, "填错了", "按 2,3 这种写法填。")
                return
            k = len(levels)
            if k == k0:
                means = [sum(g) / len(g) for g in groups]
                cuts = [(means[i] + means[i + 1]) / 2.0 for i in range(k - 1)]
            else:
                # 你填的档数和"长度差 ≤10% 算一类"自动分出来的类数不一致。
                # 绝不能退回等分位硬切 —— 那会把长度几乎一样的支架分到两类里
                #（用户反馈"明明一样长却分成两种"就是这个）。先问清楚：
                if QMessageBox.question(
                        self, "档数对不上",
                        "按「长度差 ≤%d%% 算同一类」自动分出来是 %d 类，"
                        "你填了 %d 个串数。\n\n"
                        "· 是 —— 按长度重新聚成 %d 档（切在长度间隙最大的地方，"
                        "不会把长度接近的切开）\n"
                        "· 否 —— 取消，什么都不改；推荐再点一次「支架按长度分档」，"
                        "按默认的 %d 个串数填，那样分档严格等于长度聚类的结果"
                        % (int(RACK_LEN_TOL * 100), k0, k, k, k0),
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                    return
                cuts = _kmeans_cuts(longs, k)
            counts = {lv: 0 for lv in levels}
            spans = {lv: [None, None] for lv in levels}
            undone = []          # 整册分档算"一步"：Ctrl+Z 一次全撤回
            for pg, pm in pms.items():
                hit = False
                snap = [(i, dict(s)) for i, s in enumerate(pm.shapes)]
                for s in pm.shapes:
                    if s.get("label") != "Tracker" or s.get("locked"):
                        continue          # 锁上的支架不动
                    L, _sc = length_of(pg, s)
                    gi = 0
                    for c in cuts:
                        if L > c:
                            gi += 1
                    lv = levels[min(gi, k - 1)]
                    raw = s.get("raw") if isinstance(s.get("raw"), dict) else {}
                    raw["strings"] = lv
                    raw["length_ft"] = None if not use_ft else round(L, 1)
                    if use_ft:
                        raw["page_scale_ft_per_in"] = _sc
                    s["raw"] = raw
                    s["name"] = "%d串" % lv
                    counts[lv] = counts.get(lv, 0) + 1
                    sp = spans[lv]
                    sp[0] = L if sp[0] is None else min(sp[0], L)
                    sp[1] = L if sp[1] is None else max(sp[1], L)
                    hit = True
                if hit:
                    undone.append((pg, snap))
                    pm.dirty = True
                    self.edited[pg] = pm
            if undone:
                self.undo.append(undone)
                del self.undo[:-200]
                self.redo.clear()
            self.mark_dirty()
            self.on_selection()
            for it in self.items:
                it.update()
            QMessageBox.information(
                self, "支架分档完成",
                "整册 %d 个支架框（%d 页），按长度分成 %d 档"
                "（自动按「长度差 ≤%d%% 算一类」分出来 %d 类%s）：\n%s\n\n"
                "长度按每页比例尺折算成%s再比；已写进 JSON"
                "（raw.strings、raw.length_ft、raw.page_scale_ft_per_in），框上也标了串数。\n"
                "（主程序要拿这个值来定类型，还需要我加 3 行读取代码）"
                % (len(racks), len(pages), k, int(RACK_LEN_TOL * 100), k0,
                   "" if k == k0 else "，你填的档数不一样，已按长度重新聚档",
                   "\n".join(
                       "%d串：%d 个%s" % (
                           lv, counts[lv],
                           "（长度 %.0f~%.0f %s）" % (spans[lv][0], spans[lv][1], unit)
                           if counts[lv] else "")
                       for lv in levels), unit))

        def on_export_check(self):
            """导出核对表 CSV：每页每个 Node 框一行（现有名字 / 框内候选 / 建议）。"""
            import csv
            from PySide6.QtWidgets import QProgressDialog
            if not self.pm or not self.dbg:
                self.statusBar().showMessage("先打开一份 JSON 再导出核对表", 5000)
                return
            sheets, rows_of = self._sheet_rows()
            if not sheets:
                QMessageBox.information(self, "缺标签表",
                                        "先点工具栏「选标签表…」选上 LBD 标签表(xlsx)。")
                return
            if not self.pdf or not os.path.exists(self.pdf):
                QMessageBox.information(self, "缺 PDF",
                                        "读框内文字要 PDF，请先「选 PDF…」指定这份 JSON 的 PDF。")
                return
            base = os.path.splitext(os.path.basename(self.dbg.path))[0] + "_核对表.csv"
            path, _ = QFileDialog.getSaveFileName(self, "导出核对表", base, "CSV (*.csv)")
            if not path:
                return
            blank = not hasattr(self.dbg, "_elements")
            order = self._page_order()
            if self.page not in order:
                order = sorted(set(order) | {self.page})
            try:
                ptext = PdfText(self.pdf)
            except Exception as e:
                QMessageBox.warning(self, "读不了 PDF", "打开 PDF 失败：%s" % e)
                return
            prog = QProgressDialog("正在生成核对表…", "取消", 0, len(order), self)
            prog.setWindowTitle("导出核对表")
            prog.setMinimumDuration(0)
            prog.setWindowModality(Qt.WindowModality.WindowModal)
            rows = []
            try:
                for k, pg in enumerate(order):
                    prog.setValue(k)
                    prog.setLabelText("第 %d 页（%d/%d）…" % (pg, k + 1, len(order)))
                    QApplication.processEvents()
                    if prog.wasCanceled():
                        break
                    pm = self.pm if pg == self.page else (self.edited.get(pg)
                                                          or PageModel(self.dbg, pg))
                    items = ptext.items(pg)
                    hit, _cnt = page_sheet_by_text(items, sheets)
                    idx = order.index(pg)
                    by_order = sheets[idx] if idx < len(sheets) else ""
                    sheet = (hit or by_order) if blank else by_order
                    num_set = {n for n, _l in (rows_of(sheet) if sheet else [])}
                    rr, _dry = check_rows(pm.shapes, items, sheet, num_set,
                                          pm.width, pm.height)
                    # 每个框里到底印了哪些字（全部文字，不只编号）—— 给人在 Excel 里核
                    box_txt = {i: self.box_texts_sorted(s, items, pm.width, pm.height)
                               for i, s in enumerate(pm.shapes)
                               if s.get("label") == "Node"}
                    for r in rr:
                        r["page"] = pg
                        r["sheet"] = sheet
                        r["boxtext"] = " | ".join(box_txt.get(r.get("ix"), []))
                        rows.append(r)
            finally:
                prog.close()
            cols = ["page", "sheet", "ix", "cx", "cy", "now", "cand", "sug", "src",
                    "boxtext"]
            head = {"page": "页号", "sheet": "分表", "ix": "框序号", "cx": "中心X",
                    "cy": "中心Y", "now": "现有名字", "cand": "框内候选(号@距离px)",
                    "sug": "建议名字", "src": "建议来源", "boxtext": "框内全部文字"}
            try:
                with open(path, "w", encoding="utf-8-sig", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    w.writerow(head)
                    for r in rows:
                        w.writerow({c: r.get(c, "") for c in cols})
            except Exception as e:
                QMessageBox.warning(self, "写不了文件", "%s" % e)
                return
            n_diff = sum(1 for r in rows if (r.get("now") or "") != (r.get("sug") or ""))
            n_cand = sum(1 for r in rows if r.get("cand"))
            QMessageBox.information(
                self, "核对表已导出",
                "写好了：\n%s\n\n共 %d 行（%d 个框）；框内找到候选的 %d 个；"
                "现有名字和建议不一致的 %d 个。\n\n"
                "在 Excel 里看：cand 列是这个框里找到的文字（号@距离），"
                "sug 是按现在规则算的建议名，src=框内 就是真从框里取到的。"
                % (path, len(rows), len(rows), n_cand, n_diff))

        # ---------------- 保存
        def do_save(self, dest, quiet=False):
            if not self.dbg:
                return 0
            self._stash_dirty()
            mod = {p: pm.result() for p, pm in self.edited.items()}
            pages = sorted(mod)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.dbg.save(dest, mod)
                if not getattr(self.dbg, "is_folder", False):
                    self.dbg = DebugJson(dest)
            finally:
                QApplication.restoreOverrideCursor()
            self.edited.clear()
            self.undo.clear()
            self.cmb_page.blockSignals(True)
            self.cmb_page.clear()
            for n in self.dbg.page_numbers():
                self.cmb_page.addItem(str(n), n)
            self.cmb_page.blockSignals(False)
            self.setWindowTitle("LBD 标注工具 v%s — %s"
                                % (ANNOTATOR_VERSION, os.path.basename(dest)))
            keep = self.page if self.page in self.dbg.pages else self.dbg.page_numbers()[0]
            self.pm = None
            self.goto_page(keep)
            if not quiet:
                if getattr(self.dbg, "is_folder", False):
                    QMessageBox.information(
                        self, "已写回补标文件夹",
                        "写回 %d 页：%s\n\n（每页写进它自己的 json，图片没动；"
                        "桌面上的原始标注不受影响）\n页号：%s"
                        % (len(pages), self.dbg.path,
                           "、".join(str(p) for p in pages) if pages else "（没有改动）"))
                    return len(mod)
                QMessageBox.information(
                    self, "已保存",
                    "写好了：\n%s\n\n写入的页：%s\n（没改过的页和段落是逐字节照搬的）"
                    % (dest, "、".join(str(p) for p in pages) if pages else "（没有改动）"))
            return len(mod)

        def on_open_folder(self):
            """打开「补标文件夹」（一个 json 配一张同名图片）。"""
            base = os.path.expanduser("~")
            try:
                if self.dbg and getattr(self.dbg, "path", ""):
                    base = (self.dbg.path if getattr(self.dbg, "is_folder", False)
                            else os.path.dirname(self.dbg.path))
            except Exception:
                pass
            d = QFileDialog.getExistingDirectory(
                self, "选择补标文件夹（每个 json 旁边有同名图片）", base)
            if d:
                self.load_folder(d)

        def on_folder_save(self):
            """补标文件夹就地保存：改过的页写回各自的 json。"""
            if not self.dbg or not getattr(self.dbg, "is_folder", False):
                return
            self._stash_dirty()
            pages = sorted(p for p, pm in self.edited.items() if getattr(pm, "dirty", False))
            if not pages:
                QMessageBox.information(self, "没有改动", "这次没有改过任何一页，不用保存。")
                return
            if QMessageBox.question(
                    self, "写回补标文件夹",
                    "把改过的 %d 页写回：\n%s\n\n"
                    "（只覆盖这个文件夹里的 json，图片和你桌面上的原始标注都不动）\n"
                    "继续吗？" % (len(pages), self.dbg.path)
            ) != QMessageBox.StandardButton.Yes:
                return
            n = self.do_save(self.dbg.path, quiet=True)
            self.statusBar().showMessage("已写回 %d 页：%s" % (n, self.dbg.path), 8000)

        def on_save_as(self):
            if not self.dbg:
                return
            if getattr(self.dbg, "is_folder", False):
                self.on_folder_save()
                return
            base = os.path.splitext(self.dbg.path)[0] + "_annotated.json"
            p, _ = QFileDialog.getSaveFileName(self, "另存标注结果", base,
                                               "JSON (*.json)")
            if p:
                self.do_save(p)

        def on_save_over(self):
            if not self.dbg:
                return
            if getattr(self.dbg, "is_folder", False):
                self.on_folder_save()
                return
            src = self.dbg.path
            if QMessageBox.question(
                    self, "覆盖原文件",
                    "直接覆盖：\n%s\n\n（不留备份文件，覆盖后原内容就没了）\n继续吗？" % src
            ) != QMessageBox.StandardButton.Yes:
                return
            self.do_save(src)
            self.statusBar().showMessage("已覆盖原文件：%s" % src, 8000)

    app = QApplication.instance() or QApplication(sys.argv)
    win = Win(path)
    win.show()
    if memtest:
        print("READY dpi=%s 页=%s 框=%d 底图=%s"
              % (win.dpi, win.page, len(win.items), win._img_note), flush=True)
        time.sleep(float(memtest))
        return 0
    if roundtrip:
        # 复现「一页画完 -> 翻下一页 -> 再翻回来」：看框会不会串页、底图会不会花
        pgs = win.dbg.page_numbers()
        p0 = pgs[0]
        print("起始页 %d，共 %d 页，当前 %d 个框" % (p0, len(pgs), len(win.items)), flush=True)
        win.goto_page(p0)
        n0 = len(win.items)
        win.add_shape("Node", [100, 100, 400, 900])
        b0 = list(win.items[-1].shape_data["bbox"])
        img0 = win._pixmap.copy(0, 0, 100, 100).toImage().bits().tobytes()
        print("第 %d 页画好一个 Node，框数 %d" % (p0, len(win.items)), flush=True)
        win.goto_offset(1)
        p1 = win.page
        n1 = len(win.items)
        print("翻到第 %d 页，框数 %d" % (p1, n1), flush=True)
        win.add_shape("Tracker", [200, 200, 600, 300])
        img1 = win._pixmap.copy(0, 0, 100, 100).toImage().bits().tobytes()
        print("第 %d 页画好一个 Tracker，框数 %d" % (p1, len(win.items)), flush=True)
        win.goto_page(p0)
        print("第 %d 页 %d -> %d 个框；第 %d 页 %d -> %d 个框"
              % (p0, n0, len(win.items), p1, n1, len(win.items)))
        print("   回到第 %d 页后框数 = 原来+1: %s" % (p0, len(win.items) == n0 + 1))
        back = [x for x in win.items if x.shape_data["bbox"] == b0]
        print("   画的框还在原位: %s" % bool(back))
        img0b = win._pixmap.copy(0, 0, 100, 100).toImage().bits().tobytes()
        print("   第 %d 页底图与最初一致: %s" % (p0, img0b == img0))
        print("   两页底图本来就不同: %s" % (img0 != img1))
        win.goto_page(p1)
        print("   第 %d 页那个框还在: %s" % (p1, len(win.items) == n1 + 1))
        # 全量对比：这一页每个框的坐标，翻页往返前后必须完全一致
        def snap_all():
            return sorted(tuple(round(v, 1) for v in it.shape_data["bbox"])
                          for it in win.items)

        win.goto_page(p1)
        s_before = snap_all()
        win.goto_page(p0)
        win.goto_page(p1)
        s_after = snap_all()
        print("第 %d 页全部 %d 个框逐个对比: %s"
              % (p1, len(s_before), "完全一致" if s_before == s_after else "有变化！"))
        if s_before != s_after:
            for a, b in list(zip(s_before, s_after))[:8]:
                if a != b:
                    print("     变了: %s -> %s" % (a, b))
        return 0
    if smoke:
        if not path:
            print("smoke：没给 JSON 路径")
            return 2
        # 开发用：LBD_JSON / LBD_PDF 可以指定"用哪个文件跑截图检查"
        _j = os.environ.get("LBD_JSON")
        if _j:
            if os.environ.get("LBD_REAL_SETTINGS"):
                # 开发用：这次自检用**真实**的设置目录（默认自检走临时目录，
                # 免得把用户设置改了）—— 用来验证「起始页」这类按文件记的设置
                globals()["TEST_MODE"] = False
                print("（这次用真实设置目录：%s）" % work_dir())
            win.pdf = os.environ.get("LBD_PDF") or win.pdf
            path = _j
        win.load_file(path)
        print("载入 OK：共 %d 页，第 %d 页 %d 个框"
              % (len(win.dbg.page_numbers()), win.page, len(win.items)))
        # 开发用：LBD_SHOT=<png 路径> 时，把窗口截一张图、并把工具栏按钮列出来，
        # 方便改完界面肉眼核对（自检本来就跑在 offscreen 上，不需要显示器）
        _shot = os.environ.get("LBD_SHOT")
        if _shot:
            from PySide6.QtWidgets import QToolBar, QToolButton

            def _p(s):
                try:
                    print(s)
                except UnicodeEncodeError:      # 控制台按 GBK 编码时别把自检弄崩
                    print(s.encode("ascii", "replace").decode("ascii"))

            for t in win.findChildren(QToolBar):
                _p("工具栏[%s]：%s" % (t.windowTitle(),
                                     " | ".join(a.text() for a in t.actions() if a.text())))
            for b in win.findChildren(QToolButton):
                if b.menu() is not None:
                    _p("下拉[%s]：%s" % (b.text(),
                                       " | ".join(a.text() or "（勾选框）"
                                                  for a in b.menu().actions()
                                                  if not a.isSeparator())))
            win.show()
            QApplication.processEvents()
            _zoom = os.environ.get("LBD_ZOOM")       # 开发用：截图前先放大几档
            if _zoom:
                try:
                    win.zoom_by(float(_zoom))
                    QApplication.processEvents()
                except Exception:
                    pass
            win.grab().save(_shot)
            _p("截图：%s（%dx%d）" % (_shot, win.width(), win.height()))
        print("内存：载入后 %.0f MB（%s）"
              % (rss_mb(), win._img_note or "底图来源见下"))
        first = win.page
        win.add_shape("Node", [120, 120, 420, 900])
        it = win.current_item()
        it.shape_data["name"] = "SMOKE-LBD-001"
        it.update()
        win.on_selection()
        print("画框 OK：本页 %d 个框，选中类别 %s"
              % (len(win.items), it.shape_data["label"]))
        win.goto_offset(1)
        print("翻页 OK：第 %d 页 %d 个框" % (win.page, len(win.items)))
        win.goto_page(first)
        for i in list(win.items):
            if i.shape_data.get("name") == "SMOKE-LBD-001":
                win.scene.clearSelection()
                i.setSelected(True)
        win.on_delete()
        print("删除 OK：本页剩 %d 个框" % len(win.items))
        from PySide6.QtCore import QPointF
        from PySide6.QtTest import QTest
        from PySide6.QtCore import Qt as _Qt
        print("控制点计算检查:", Canvas.apply_handle([0, 0, 100, 100], 7, 50, 50),
              Canvas.apply_handle([0, 0, 100, 100], 0, -20, -20))
        # 自检本来是给自带样例用的：别的文件/补标文件夹上有的页可能是空的，
        # 这里先保证当前页至少有一个框，免得后面 win.items[0] 直接 IndexError
        if not win.items:
            win.add_shape("Tracker", [200, 200, 600, 300])
            print("   （本页原本没有框，先补一个再继续自检）")
        win.scene.clearSelection()
        it0 = win.items[0]
        it0.setSelected(True)
        win.on_selection()
        before = it0.scene_box()
        hx, hy = Canvas.handle_points(before)[7]
        p0 = win.canvas.mapFromScene(QPointF(hx, hy))
        p1 = win.canvas.mapFromScene(QPointF(hx + 60, hy + 40))
        QTest.mousePress(win.canvas.viewport(), _Qt.MouseButton.LeftButton,
                         _Qt.KeyboardModifier.NoModifier, p0)
        QTest.mouseMove(win.canvas.viewport(), p1)
        QTest.mouseRelease(win.canvas.viewport(), _Qt.MouseButton.LeftButton,
                           _Qt.KeyboardModifier.NoModifier, p1)
        after = it0.scene_box()
        print("拉边改大小：%s -> %s"
              % (["%.0f" % v for v in before], ["%.0f" % v for v in after]))
        grew = after[2] > before[2] and after[3] > before[3]
        print("拉右下角变大:", grew)
        # 点在框内部：会走 Qt 的命中测试（就是刚才报 shape() 错的那条路径）
        win.scene.clearSelection()
        it1 = win.items[1]
        b = it1.scene_box()
        mid = win.canvas.mapFromScene(QPointF((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0))
        QTest.mousePress(win.canvas.viewport(), _Qt.MouseButton.LeftButton,
                         _Qt.KeyboardModifier.NoModifier, mid)
        QTest.mouseRelease(win.canvas.viewport(), _Qt.MouseButton.LeftButton,
                           _Qt.KeyboardModifier.NoModifier, mid)
        print("点框内部 OK：选中 %d 个框" % len(win.canvas.scene().selectedItems()))
        dest = os.path.join(app_dir(), "smoke_out.json")
        n = win.do_save(dest, quiet=True)
        print("另存 OK：%s（写了 %d 页）" % (dest, n))
        # 删框同步验证：删掉 3 个 Tracker + 1 个 Node，另存后重新读一遍，看是不是真少了
        win.goto_page(first)
        base = PageModel(SuggestFolderDoc(path) if os.path.isdir(path) else DebugJson(path),
                         first).counts()
        before = len(win.items)
        victims = [it for it in win.items if it.shape_data["label"] == "Tracker"][:3]
        victims += [it for it in win.items if it.shape_data["label"] == "Node"][:1]
        win.scene.clearSelection()
        for v in victims:
            v.setSelected(True)
        print("   诊断：选中 %d 个；删除前 dirty=%s shapes=%d edited=%d pending=%s"
              % (len(win.scene.selectedItems()), win.pm.dirty, len(win.pm.shapes),
                 len(win.edited), win._pending is not None))
        it_dbg = win.scene.selectedItems()[0]
        print("   诊断：item.shape_data 是 pm.shapes 里的对象？ %s；in 判定？ %s；类型 %s"
              % (any(it_dbg.shape_data is s for s in win.pm.shapes),
                 it_dbg.shape_data in win.pm.shapes,
                 type(it_dbg.shape_data).__name__))
        win.on_delete()
        print("   诊断：删除后 dirty=%s shapes=%d" % (win.pm.dirty, len(win.pm.shapes)))
        after = len(win.items)
        dest2 = os.path.join(app_dir(), "smoke_del.json")
        win.do_save(dest2, quiet=True)
        print("   诊断：保存后 edited=%d" % len(win.edited))
        c2 = PageModel(SuggestFolderDoc(path) if os.path.isdir(path) else DebugJson(dest2),
                       first).counts()
        print("删框同步：界面 %d -> %d，另存后重新读 Node %d / Tracker %d"
              % (before, after, c2["Node"], c2["Tracker"]))
        ok_del = (c2["Node"] == base["Node"] - 1 and c2["Tracker"] == base["Tracker"] - 3)
        print("   期望 Node %d / Tracker %d -> %s"
              % (base["Node"] - 1, base["Tracker"] - 3, "通过" if ok_del else "不一致！"))
        # ESC：画到一半取消；再按回到选择模式
        from PySide6.QtCore import QRectF
        win.set_mode("Tracker")
        win.canvas._origin = QPointF(500, 500)
        win.canvas._rubber = win.canvas.scene().addRect(QRectF(500, 500, 100, 100))
        nb = len(win.items)
        win.on_escape()
        e1 = (win.canvas._rubber is None and len(win.items) == nb)
        win.on_escape()
        e2 = (win.canvas.mode == "select")
        print("ESC：画到一半取消 %s；再按回到选择模式 %s" % (e1, e2))
        # 复制粘贴：选中两个框 -> 复制 -> 粘贴两次，看数量与错开量
        win.scene.clearSelection()
        for it in win.items[:2]:
            it.setSelected(True)
        win.on_copy()
        n_before = len(win.items)
        win.on_paste()
        n_mid = len(win.items)
        win.on_paste()
        n_after = len(win.items)
        pasted = win.scene.selectedItems()
        print("复制粘贴：选中 2 个 -> 复制 -> 粘贴两次 %d -> %d -> %d（每次 +2）"
              % (n_before, n_mid, n_after))
        print("   粘贴后选中 %d 个，新增框的 ocr_index 都是 None: %s"
              % (len(pasted), all(p.shape_data.get("ocr_index") is None for p in pasted)))
        orig_boxes = {tuple(round(v, 1) for v in it.shape_data["bbox"])
                      for it in win.items[:n_before]}
        new_boxes = [tuple(round(v, 1) for v in p.shape_data["bbox"]) for p in pasted]
        n_same = sum(1 for b in new_boxes if b in orig_boxes)
        print("   粘贴的框和原框位置完全重合的个数: %d/%d（应全部重合：就是原位叠一份）  %s"
              % (n_same, len(new_boxes),
                 "通过" if new_boxes and n_same == len(new_boxes) else "不一致！"))
        # 跨页粘贴：复制 -> 翻到下一页 -> 粘贴
        win.scene.clearSelection()
        if not win.items:
            win.add_shape("Tracker", [200, 200, 600, 300])
        win.items[0].setSelected(True)
        win.on_copy()
        win.goto_offset(1)
        n_page2 = len(win.items)
        win.on_paste()
        print("跨页粘贴：第 %d 页 %d -> %d 个框"
              % (win.page, n_page2, len(win.items)))
        # 撤销 / 重做
        n0 = len(win.items)
        win.on_undo()
        n_undo = len(win.items)
        win.on_redo()
        n_redo = len(win.items)
        win.on_undo()
        n_undo2 = len(win.items)
        print("撤销/重做：%d ->(撤销) %d ->(重做) %d ->(再撤销) %d  重做栈 %d"
              % (n0, n_undo, n_redo, n_undo2, len(win.redo)))
        print("   期望 粘贴前/撤销后一致、重做后回到粘贴后: %s"
              % ("通过" if (n_redo == n0 and n_undo == n_page2 and n_undo2 == n_page2) else "不一致！"))
        # 翻页往返：框的坐标和底图像素都不能变
        win.goto_page(first)
        it_chk = win.items[0]
        shape_ref = it_chk.shape_data
        box_before = list(shape_ref["bbox"])
        pm_before = win.pm
        pix_before = win._pixmap.copy(0, 0, 120, 120).toImage().bits().tobytes()
        win.goto_offset(1)
        win.goto_page(first)
        same_model = (win.pm is pm_before)
        in_list = any(s is shape_ref for s in win.pm.shapes)
        box_after = list(shape_ref["bbox"])
        pix_after = win._pixmap.copy(0, 0, 120, 120).toImage().bits().tobytes()
        print("翻页往返：同一个页面模型 %s；框对象还在 %s" % (same_model, in_list))
        print("   坐标 %s -> %s  %s" % (box_before, box_after,
                                        "一致" if box_after == box_before else "变了！"))
        print("   底图像素 %s" % ("一致" if pix_after == pix_before else "变了！"))

        # ---------------- 锁定框：不能选/不能拖/删不掉/能撤销/能存盘/清理不删
        win.goto_page(first)
        it_lock = win.items[0]
        win.scene.clearSelection()
        it_lock.setSelected(True)
        win.on_toggle_lock()
        flag = BoxItem.GraphicsItemFlag
        ok_sel = not (it_lock.flags() & flag.ItemIsSelectable)
        ok_mov = not (it_lock.flags() & flag.ItemIsMovable)
        n_before_lock = len(win.items)
        it_lock.setSelected(True)                 # 锁上的框不该还能被选中
        win.on_delete()                           # 选不中 -> 也删不掉
        ok_del = (len(win.items) == n_before_lock
                  and it_lock.shape_data in win.pm.shapes)
        win.on_undo()                             # Ctrl+Z 撤销"锁定"这一步
        ok_undo = bool(win.items) and not win.items[0].is_locked()
        print("锁定：不可选 %s；不可拖 %s；选不中也删不掉 %s；撤销能解锁 %s"
              % (ok_sel, ok_mov, ok_del, ok_undo))
        win.items[0].set_locked(True)
        dest_lock = os.path.join(app_dir(), "smoke_lock.json")
        win.do_save(dest_lock, quiet=True)
        back = [s for s in PageModel(SuggestFolderDoc(path) if os.path.isdir(path)
                                     else DebugJson(dest_lock), first).shapes
                if s.get("locked")]
        print("   锁定状态存进 JSON 再读回来: %d 个（应为 1）  %s"
              % (len(back), "通过" if len(back) == 1 else "不一致！"))
        fake = [{"label": "Tracker", "name": "", "bbox": [0, 0, 20, 300],
                 "confidence": 0.9, "source": "model", "raw": {}, "locked": True},
                {"label": "Tracker", "name": "", "bbox": [0, 0, 20, 300],
                 "confidence": 0.9, "source": "model", "raw": {}}]
        kept2, _st2 = clean_shapes(fake)
        print("   清理多余框会保留锁定的那个: %s"
              % ("通过" if len(kept2) == 1 and kept2[0].get("locked") else "不一致！"))

        # 一键锁定整页 Node
        win.goto_page(first)
        win.on_lock_page_nodes()
        nd = [s for s in win.pm.shapes if s.get("label") == "Node"]
        tk = [s for s in win.pm.shapes if s.get("label") != "Node"]
        ok_lock_all = bool(nd) and all(s.get("locked") for s in nd)
        ok_only_node = not any(s.get("locked") for s in tk)
        win.on_lock_page_nodes()                 # 再点一次 = 全解锁
        ok_unlock_all = not any(s.get("locked") for s in win.pm.shapes)
        print("一键锁定本页 Node：%d 个 Node 全锁上 %s；别的类别没动 %s；再点一次全解锁 %s"
              % (len(nd), ok_lock_all, ok_only_node, ok_unlock_all))

        # 整册清理多余框 + 一步撤销（用页面模型直接量，不靠翻页重新读盘）
        pgs_all = win.dbg.page_numbers()
        # 自检本来跑的是自带样例；拿别的文件跑时有的页可能是空的（比如封面页），
        # 不能假设"第 1 页一定有框"，否则自检会在这儿崩（用户看到的就是这个 traceback）
        p_first = next((p for p in pgs_all
                        if (win.edited.get(p) or PageModel(win.dbg, p)).shapes), pgs_all[0])
        win.goto_page(p_first)
        donor = dict(win.pm.shapes[0]) if win.pm.shapes else {
            "label": "Node", "name": "", "bbox": [100.0, 100.0, 400.0, 900.0],
            "confidence": None, "class_id": DEFAULT_CLASS_ID["Node"],
            "source": "manual", "raw": {}, "ocr_index": None}
        donor.update({"source": "model", "raw": {}, "locked": False, "name": ""})
        win.pm.shapes.append(dict(donor))     # 造两个和它完全一样的"多余框"
        win.pm.shapes.append(dict(donor))
        win.pm.dirty = True
        win.edited[p_first] = win.pm
        n_before_clean = len(win.pm.shapes)
        tot, lines = win.clean_pages(pgs_all, clean_manual=False, quiet=True)
        n_after_clean = len(win.edited[p_first].shapes)
        win.on_undo()                         # 一次撤销要把整册一起撤回
        n_after_undo = len((win.edited.get(p_first) or win.pm).shapes)
        print("整册清理：第 %d 页 %d -> %d 个框（整册共删 %d，动了 %d 页）；"
              "Ctrl+Z 一次撤回 -> %d  %s"
              % (p_first, n_before_clean, n_after_clean, tot["removed"],
                 tot["pages"], n_after_undo,
                 "通过" if (n_after_clean < n_before_clean
                            and n_after_undo == n_before_clean) else "不一致！"))
        # 存盘再重读：清理必须真的落到文件里（"清理不生效"最容易卡在这一步）。
        # 注意要放在撤销检查之后 —— 保存会清空撤销栈。
        win.clean_pages(pgs_all, clean_manual=False, quiet=True)
        dest_all = os.path.join(app_dir(), "smoke_clean_all.json")
        win.do_save(dest_all, quiet=True)
        n_saved = len(PageModel(SuggestFolderDoc(path) if os.path.isdir(path)
                                else DebugJson(dest_all), p_first).shapes)
        print("   整册清理后存盘重读：第 %d 页 %d 个框（清理前 %d）  %s"
              % (p_first, n_saved, n_before_clean,
                 "通过" if n_saved < n_before_clean else "没写进去！"))

        # 在线更新：版本比较 + 仓库地址（不联网，纯逻辑）
        ok_ver = (parse_version("v0.6") > parse_version("v0.5")
                  and parse_version("v0.10") > parse_version("v0.9")
                  and parse_version("v1.0") > parse_version("v0.10")
                  and parse_version(ANNOTATOR_VERSION) == parse_version("v" + ANNOTATOR_VERSION))
        ok_repo = UPDATE_REPO.endswith("/LBD-Annotator")
        print("在线更新：版本比较 %s；检查的仓库 %s %s"
              % (ok_ver, UPDATE_REPO, "通过" if ok_repo else "写错了！"))

        # 识别页码解析 + 右侧栏统计（本页 Tracker 数 / 按 LBD 的串数汇总）
        ok_spec = (parse_page_spec("3-5,7", [1, 2, 3, 4, 5, 6, 7]) == [3, 4, 5, 7]
                   and parse_page_spec("x", [1, 2]) == []
                   and parse_page_spec("2-3", [1, 5]) == [])
        print("识别页码解析（3-5,7 / 无效输入）: %s"
              % ("通过" if ok_spec else "不一致！"))
        win.goto_page(pgs_all[0])
        _nodes = [s for s in win.pm.shapes if s["label"] == "Node"]
        _racks = [s for s in win.pm.shapes if s["label"] == "Tracker"]
        if _nodes and _racks:
            _nd = _nodes[0]
            _nd["name"] = "TEST-LBD-1"
            _racks[0]["bbox"] = [(_nd["bbox"][0] + _nd["bbox"][2]) / 2 - 5,
                                 (_nd["bbox"][1] + _nd["bbox"][3]) / 2 - 5,
                                 (_nd["bbox"][0] + _nd["bbox"][2]) / 2 + 5,
                                 (_nd["bbox"][1] + _nd["bbox"][3]) / 2 + 5]
            _racks[0].setdefault("raw", {})["strings"] = 3
            _rows, _tot = win.page_rack_stats()
            _hit = [r for r in _rows if r[0] == "TEST-LBD-1"]
            print("侧栏统计：本页 Tracker %d 个；TEST-LBD-1 -> %s（应为 1 个支架 3 串）  %s"
                  % (len(_racks), _hit[0][1:] if _hit else "没算到",
                     "通过" if (_hit and _hit[0][1] == 1 and _hit[0][2] == 3) else "不一致！"))
        else:
            print("侧栏统计：本页没有 Node/Tracker，跳过")

        # 按钮级自检：真的走一遍"点按钮"的路径
        # （QProgressDialog 漏 import 那种错，只有点了按钮才会炸，光测底层函数测不出来）
        try:
            n_before_btn = len(win.pm.shapes)
            tot_btn, _ls = win.on_clean_shapes_all(quiet=True, clean_manual=False)
            ok_btn = bool(tot_btn) and "removed" in tot_btn
            print("点按钮路径：清理多余框(整册) 扫了 %d 页，删 %d 个；本页 %d -> %d  %s"
                  % (len(win.dbg.page_numbers()), tot_btn["removed"] if ok_btn else -1,
                     n_before_btn, len(win.pm.shapes), "通过" if ok_btn else "出错！"))
            tot_one, _l1 = win.on_clean_shapes(quiet=True, clean_manual=False)
            ok_one = bool(tot_one) and "removed" in tot_one
            print("点按钮路径：清理多余框(本页) %s" % ("通过" if ok_one else "出错！"))
            # 进度条那条路：没人点取消时，取消标记必须是 False
            # （v0.7 就是因为 close() 之后再读 wasCanceled()，扫完被判成"已取消"，什么都不删）
            sa2, sb2, c2 = win.scan_clean_progress(win.dbg.page_numbers())
            print("进度条扫描：取消标记 %s（应为 False）；只清模型可删 %d / 连手工可删 %d  %s"
                  % (c2, sa2["removed"], sb2["removed"],
                     "通过" if not c2 else "又被当成取消了！"))
            tot_p, _lp, c3 = win.clean_pages_progress(win.dbg.page_numbers(), False)
            print("进度条清理：取消标记 %s（应为 False），删了 %d 个  %s"
                  % (c3, tot_p["removed"], "通过" if not c3 else "又被当成取消了！"))
        except Exception as e:                       # noqa: BLE001
            print("点按钮路径：出错！%s: %s" % (type(e).__name__, e))
        return 0
    return app.exec()


def fix_std_streams():
    """打包成不带控制台的 exe 后 sys.stdout/stderr 是 None，print 会直接抛异常。"""
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            try:
                setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))
            except Exception:
                pass


def last_json():
    """上次打开过的那份识别结果（存在设置文件里，双击 exe 接着用）。"""
    p = str(load_settings().get("json") or "")
    return p if p and os.path.exists(p) else ""


def autofill_run(json_path, pdf_path, xlsx_path, out_path=None, only_page=None,
                 force=False, csv_path=None):
    """无界面跑一遍「框内取文字补 LBD 编号」，逐页打印结果；给了 out 就另存一份。

    跟界面上「补全编号」用的是同一套函数，方便先拿真实文件核对。
    """
    dbg = DebugJson(json_path)
    sheets = xlsx_sheet_names(xlsx_path) if (xlsx_path and os.path.exists(xlsx_path)) else []
    if not sheets:
        # 没给标签表（或表读不出来）：只用框内文字取号，号码不跟表核对
        print("没给标签表 —— 只用框内文字取号（框里没印号的会留空）")
    cache = {}

    def rows_of(nm):
        if nm not in cache:
            try:
                cache[nm] = xlsx_lbd_rows(xlsx_path, nm) if sheets else []
            except Exception as e:
                print("  读分表 %s 失败：%s" % (nm, e))
                cache[nm] = []
        return cache[nm]

    try:
        ptext = PdfText(pdf_path)
    except Exception as e:
        print("打不开 PDF：%s" % e)
        return 2
    order = drawing_pages(dbg)
    pages = [only_page] if only_page else order
    print("JSON %s：%d 页；标签表分表 %d 个；PDF %d 页"
          % (os.path.basename(json_path), len(order), len(sheets), ptext.count()))
    mod = {}
    tot = {"filled": 0, "auto": 0, "missed": 0, "kept": 0}
    check_all = []
    for k, pg in enumerate(pages):
        idx = order.index(pg) if pg in order else -1
        sheet = sheets[idx] if 0 <= idx < len(sheets) else ""
        num_set = {n for n, _l in (rows_of(sheet) if sheet else [])}
        pm = PageModel(dbg, pg)
        now_map = {i: (s.get("name") or "")
                   for i, s in enumerate(pm.shapes) if s.get("label") == "Node"}
        if force:
            for s in pm.shapes:
                if s.get("label") == "Node":
                    s["name"] = ""
                    s["_auto"] = False
                    s["_miss"] = False
                    s["_check"] = False
        items = ptext.items(pg)
        hit, _cnt = page_sheet_by_text(items, sheets)
        if not sheets and hit:
            # 没给标签表：分表名就用页面上印的（这样名字里还带 INV 前缀）
            sheet = hit
        st = autofill_shapes(pm.shapes, items, sheet, num_set, pm.width, pm.height)
        if csv_path:
            rr, _dry = check_rows(pm.shapes, items, sheet, num_set, pm.width, pm.height)
            for r in rr:
                r["page"] = pg
                r["sheet"] = sheet
                r["now"] = now_map.get(r["ix"], "")
                check_all.append(r)
        for key in tot:
            tot[key] += st.get(key, 0)
        note = ""
        if hit and sheet and _clean_key(hit) != _clean_key(sheet):
            note = "  [警告] 页面文字印的是 %s" % hit
        print("[%2d/%2d] 第 %s 页 分表=%-10s 号码表=%-3d 出名字 %d 个"
              "（框内 %d / 推 %d / 缺 %d）%s"
              % (k + 1, len(pages), pg, sheet or "?", len(num_set),
                 st["filled"] + st["auto"], st["filled"], st["auto"],
                 st["missed"], note))
        for s in pm.shapes:
            if s.get("label") == "Node":
                flag = "红" if s.get("_miss") else ("紫" if s.get("_check")
                                                   else ("黄" if s.get("_auto") else "绿"))
                print("        %s %s" % (flag, (s.get("name") or "（空）")))
        for nm, x, y in (st.get("sus_list") or []):
            print("        [要核] %s 同一排号不连续（x=%d y=%d）" % (nm or "（空）", x, y))
        for nm in (st.get("auto_list") or []):
            print("        [要核] %s 图上没编号，按标签表顺序推的" % (nm or "（空）"))
        mod[pg] = pm.result()
    print("合计：框内取到 %d，按标签表推 %d，没取到 %d，已有名字跳过 %d"
          % (tot["filled"], tot["auto"], tot["missed"], tot["kept"]))
    if out_path:
        dbg.save(out_path, mod)
        print("已另存：%s" % out_path)
    if csv_path:
        import csv
        cols = ["page", "sheet", "ix", "cx", "cy", "now", "cand", "sug", "src"]
        head = {"page": "页号", "sheet": "分表", "ix": "框序号", "cx": "中心X",
                "cy": "中心Y", "now": "现有名字", "cand": "框内候选(号@距离px)",
                "sug": "建议名字", "src": "建议来源"}
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writerow(head)
            for r in check_all:
                w.writerow({c: r.get(c, "") for c in cols})
        n_cand = sum(1 for r in check_all if r.get("cand"))
        n_diff = sum(1 for r in check_all if (r.get("now") or "") != (r.get("sug") or ""))
        print("核对表已写：%s（%d 行；框内有候选 %d；现有≠建议 %d）"
              % (csv_path, len(check_all), n_cand, n_diff))
    return 0


def main():
    fix_std_streams()
    ap = argparse.ArgumentParser(description="LBD 标注工具 v%s" % ANNOTATOR_VERSION)
    ap.add_argument("json", nargs="?", default=None,
                    help="识别结果 debug JSON（不填就打开对话框）")
    ap.add_argument("--selftest", action="store_true", help="无界面自检")
    ap.add_argument("--smoke", action="store_true", help="无显示界面自检（画框/翻页/另存）")
    ap.add_argument("--memtest", action="store_true",
                    help="只载入并显示第一页，然后挂着（外部采样内存用）")
    ap.add_argument("--roundtrip", action="store_true",
                    help="复现「画完翻页再翻回来」，检查框和底图有没有乱")
    ap.add_argument("--autofill", action="store_true",
                    help="无界面按框内文字补 LBD 编号（配 --pdf/--xlsx/--out 用）")
    ap.add_argument("--pdf", default=None, help="配套 PDF（框内取文字用）")
    ap.add_argument("--xlsx", default=None, help="LBD 标签表(xlsx)")
    ap.add_argument("--page", type=int, default=None, help="--autofill 时只跑这一页")
    ap.add_argument("--start-page", type=int, default=None,
                    help="把这份 JSON 的起始页记下来（打开时直接停在这一页；0=清除）。"
                         "不启动界面，写完就退出")
    ap.add_argument("--update-check", action="store_true",
                    help="跑一遍在线更新诊断（DNS/curl/git/HEAD/api 各测一次），"
                         "把报告写到程序目录的 update_check_report.txt，不启动界面")
    ap.add_argument("--force", action="store_true",
                    help="--autofill 时把已有的 Node 名字也重算一遍")
    ap.add_argument("--csv", default=None,
                    help="--autofill 时同时导出核对表 CSV（现有/框内候选/建议）")
    ap.add_argument("--hold", type=float, default=20.0, help="--memtest 挂多久（秒）")
    ap.add_argument("--out", default=None, help="自检时的输出文件")
    a = ap.parse_args()
    global TEST_MODE
    TEST_MODE = bool(a.smoke or a.memtest or a.roundtrip)   # 自检不碰用户的设置/缓存
    if a.update_check:
        rep = update_diagnose(settings=load_settings())
        path = os.path.join(app_dir(), "update_check_report.txt")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(rep)
        except Exception:
            path = "（写不进程序目录）"
        print(rep)
        try:                                    # 打包成 exe 后没有控制台，弹个框给人看
            from PySide6.QtWidgets import QApplication, QMessageBox
            QApplication.instance() or QApplication([])
            QMessageBox.information(None, "更新诊断（把这段发我）",
                                    rep + "\n\n（也写到了：%s）" % path)
        except Exception:
            pass
        return 0
    if a.start_page is not None:
        src = a.json or DEFAULT_JSON
        if not (src and os.path.exists(src)):
            print("--start-page 需要一个存在的 JSON 路径")
            return 2
        before = os.path.getsize(src)
        v = write_json_start_page(src, a.start_page)      # 写进 JSON 本身
        set_start_page(src, 0)                            # 顺手清掉老的本机记录
        after = os.path.getsize(src)
        try:                                              # 写完必须还是合法 JSON
            with open(src, "r", encoding="utf-8") as f:
                json.load(f)
            ok = "JSON 校验通过"
        except Exception as e:
            ok = "⚠ JSON 校验失败：%s" % e
        print("已写入：%s → lbd_start_page = %s（%d → %d 字节；%s）"
              % (os.path.basename(src), v if v else "（已删除）", before, after, ok))
        return 0
    if a.autofill:
        src = a.json or DEFAULT_JSON
        pdf = a.pdf or str(load_settings().get("pdf") or "")
        xl = a.xlsx or str(load_settings().get("xlsx") or "")
        # 标签表可以不给：不给就只用框内文字取号（号码不跟表核对）
        if not (src and os.path.exists(src) and pdf and os.path.exists(pdf)):
            print("--autofill 需要：JSON 路径 + --pdf（--xlsx 可选）")
            return 2
        return autofill_run(src, pdf, xl, a.out, a.page, a.force, a.csv)
    if a.selftest:
        src = a.json or DEFAULT_JSON
        if not os.path.exists(src):
            print("找不到文件：%s" % src)
            return 2
        return selftest(src, a.out)
    if a.smoke:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        return run_gui(a.json or DEFAULT_JSON, smoke=True)
    if a.memtest:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        return run_gui(a.json or DEFAULT_JSON, memtest=a.hold)
    if a.roundtrip:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        return run_gui(a.json or DEFAULT_JSON, roundtrip=True)
    if a.json and not os.path.exists(a.json):
        print("找不到文件：%s" % a.json)
        return 2
    src = a.json
    # 双击 exe 就是空白画布（黑底），不自动打开上次那份、也不记路径：
    # 只有命令行显式给了文件，或者你自己点「打开 JSON」，才会载入东西。
    return run_gui(src)


if __name__ == "__main__":
    sys.exit(main())
LBD_LABEL_RE = re.compile(r"INV\s*\d+\s*[A-Z]\s*\d+\s*[-_ ]?LBD[-_ ]?\s*\d+", re.I)
# "像编号"的：LBD 后面跟着数字（LBD-8 / LBD 1.01.1.C.5），或者只有 INV 前缀没 LBD 字样。
# 注意不能写成"只要出现 LBD 就算" —— 图纸上到处都是 "LBD CLUSTER, TYP." 这种旁注。
LBDISH_RE = re.compile(r"LBD\s*[-_.#]?\s*\d|INV\s*\d+\s*[A-Z]\s*\d+", re.I)
