# -*- coding: utf-8 -*-
"""把标注工具打包成单文件 exe（照搬主程序 build.py 的 PySide6 插件 + ICU 规避做法）。
用法（必须用装了 PyInstaller + PySide6 的 Python 跑）：
    .build-venv\\Scripts\\python.exe _dev\\build_annotator.py
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
APP = os.path.join(HERE, "lbd_annotator.py")
DIST = os.path.join(HERE, "dist")
WORK = os.path.join(HERE, "build_exe")
NAME = "LBD标注工具"
CONSOLE = "--console" in sys.argv          # 调试用：带控制台，错误能直接看到
ONEDIR = "--onedir" in sys.argv            # 打成文件夹（onedir）而不是单文件
NAME = NAME + ("_debug" if CONSOLE else "")
EXE = os.path.join(DIST, NAME + ".exe")
APPDIR = os.path.join(DIST, NAME)
if ONEDIR:
    EXE = os.path.join(APPDIR, NAME + ".exe")
SEP = os.pathsep
PY = sys.executable


def log(msg=""):
    print(msg, flush=True)


def pyside_plugins():
    spec = importlib.util.find_spec("PySide6")
    if not spec:
        raise SystemExit("这个 Python 里没有 PySide6")
    return os.path.join(os.path.dirname(spec.origin), "plugins")


def build_env():
    """剔除 PATH 里带第三方 icuuc.dll 的目录（poppler 就带），否则打出来的 exe
    运行时会报 “DLL load failed while importing QtCore”。"""
    windir = (os.environ.get("WINDIR") or r"C:\Windows").lower()
    keep, dropped = [], []
    for d in (os.environ.get("PATH") or "").split(SEP):
        if not d:
            continue
        if os.path.exists(os.path.join(d, "icuuc.dll")) and not d.lower().startswith(windir):
            dropped.append(d)
        else:
            keep.append(d)
    if dropped:
        log("  已临时从 PATH 剔除带第三方 ICU 的目录：%s" % "; ".join(dropped))
    return dict(os.environ, PYTHONIOENCODING="utf-8", PATH=SEP.join(keep))


def main():
    if not os.path.exists(APP):
        raise SystemExit("找不到 %s" % APP)
    if ONEDIR:
        if os.path.isdir(APPDIR):
            try:
                if os.path.isdir(APPDIR + ".old"):
                    shutil.rmtree(APPDIR + ".old")
                os.replace(APPDIR, APPDIR + ".old")
            except Exception as e:
                log("[失败] 旧文件夹挪不动（%s）：多半是标注工具还在运行，先关掉。" % e)
                return 1
    elif os.path.exists(EXE):
        try:
            os.replace(EXE, EXE + ".old")
        except Exception as e:
            log("[失败] 旧 exe 挪不动（%s）：多半是标注工具还在运行，先关掉。" % e)
            return 1
    os.makedirs(DIST, exist_ok=True)
    pl = pyside_plugins()
    # 图标：兼容"源码在仓库里"和"源码在仓库旁边"两种目录布局
    icon = ""
    for cand in (os.path.join(ROOT, "编排器", "app.ico"),
                 os.path.join(ROOT, "CAD-MAP-main", "编排器", "app.ico")):
        if os.path.exists(cand):
            icon = cand
            break
    args = [PY, "-m", "PyInstaller", "--noconfirm", "--clean",
            "--onedir" if ONEDIR else "--onefile",
            "--console" if CONSOLE else "--windowed",
            "--name", NAME,
            "--distpath", DIST, "--workpath", WORK, "--specpath", WORK,
            "--collect-binaries", "PySide6", "--collect-binaries", "shiboken6",
            # 按框内文字补编号要读 PDF 文字层（pypdf 是在函数里 import 的，
            # 显式写上，免得 PyInstaller 漏掉）
            "--hidden-import", "pypdf",
            "--add-data", os.path.join(pl, "platforms") + SEP + "PySide6/plugins/platforms",
            "--add-data", os.path.join(pl, "styles") + SEP + "PySide6/plugins/styles",
            "--exclude-module", "PIL", "--exclude-module", "numpy",
            "--exclude-module", "tkinter", "--exclude-module", "matplotlib",
            "--exclude-module", "onnxruntime", "--exclude-module", "ultralytics"]
    if os.path.exists(icon):
        args += ["--icon", icon]
    args.append(APP)
    log("[打包] PyInstaller 打包中（约 1~2 分钟）...")
    t0 = time.time()
    p = subprocess.run(args, cwd=HERE, env=build_env())
    if p.returncode != 0 or not os.path.exists(EXE):
        log("[失败] 打包没成功（退出码 %s）" % p.returncode)
        return 1
    old = (APPDIR + ".old") if ONEDIR else (EXE + ".old")
    if os.path.exists(old):
        try:
            if os.path.isdir(old):
                shutil.rmtree(old)
            else:
                os.remove(old)
        except Exception:
            pass
    if ONEDIR:
        total = sum(os.path.getsize(os.path.join(base, f))
                    for base, _d, fs in os.walk(APPDIR) for f in fs)
        log("打包完成（文件夹版）：%s（整个文件夹 %.1f MB，用时 %.0fs）"
            % (APPDIR, total / 1048576.0, time.time() - t0))
    else:
        log("打包完成：%s（%.1f MB，用时 %.0fs）"
            % (EXE, os.path.getsize(EXE) / 1048576.0, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
