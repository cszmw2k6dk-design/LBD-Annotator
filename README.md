# LBD 标注工具（LBD Annotator）

CAD-MAP 项目配套的标注小工具：在 PDF 图纸上框 LBD 区域 / 支架，**按框内文字自动补 LBD 编号**，
导出成主程序能直接用的 JSON。

## 运行

- **用现成的 exe**：到本仓库 Release 页下载 `LBD.exe`（就是 LBD标注工具，单文件、双击即可。
  GitHub 会把中文资产名简化成 `LBD.exe`，从 v0.4 起一直是这个名字）
- **源码直跑**：双击 `run_annotator.bat`（需要 Python 3.10+，并装好 PySide6 / pypdf）

## 主要功能

- 打开识别结果 JSON（agent3-debug 那种）或**直接打开 PDF 从零标注**
- 画框（两点确认 / 拖动都行）、改大小、改名字、删除、复制粘贴、撤销重做、翻页
- **按框内文字补 LBD 编号**：文字落在哪个框里就归哪个框；带 `LBD+编号` 的优先，其次裸数字（`07`），
  再兜底取文字里的数字段（`1.01.1.C.5` → 5）并和标签表号码表核对；拆开的、被截断的文字一律过滤
- 已有名字默认不动，需要覆盖时勾「重算(覆盖已有名字)」
- **导出核对表 CSV**：每个框一行（现有名字 / 框内候选 / 建议名字 / 来源）
- **支架按长度分档**：整册所有支架框一起按「长度差 ≤10% 算同一类」聚类，写进 JSON 的 `raw.strings`
- **导出 YOLO 数据集**：`images/` + `labels/` + `classes.txt` + `data.yaml`
- **训练环境面板**：探测本机 Python、检测/一键安装 `ultralytics`

## 打包

```bat
python 打包成exe.py             :: 单文件 exe（默认）
python 打包成exe.py --onedir    :: 文件夹版（不自解压，杀软误报率低）
python 打包成exe.py --console   :: 带控制台的调试版
```

## 依赖

- **PySide6**（界面）、**pypdf**（读 PDF 文字层）
- **poppler**（`pdftoppm.exe`，渲染 PDF 底图；放程序目录的 `poppler\` 下或加入 PATH 都认）
- 训练/推理另外需要 Python 环境里的 `ultralytics`

## 说明

- 设置文件和渲染缓存放在 `%LOCALAPPDATA%\LBD标注工具\`
- 杀软可能把打包出来的 exe 误报成木马（PyInstaller 常见现象），把 exe 所在目录加进信任区即可
