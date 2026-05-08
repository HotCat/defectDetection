# Windows 10 手动部署指南

以下步骤在全新 Windows 10 (64位) 工作站上从零搭建缺陷检测系统。

---

## 1. 安装 Miniconda

1. 下载 Miniconda 安装包：
   - https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Windows-x86_64.exe （清华镜像，国内快）
   - 或官方：https://docs.conda.io/en/latest/miniconda.html
2. 运行安装程序，全部默认选项即可
   - 勾选 "Add Miniconda to my PATH environment variable"（方便命令行使用）
3. 安装完成后打开 **Anaconda Prompt**（从开始菜单搜索）

验证：
```
conda --version
python --version
```

---

## 2. 创建 Python 虚拟环境

```bash
conda create -n defect python=3.11 -y
conda activate defect
```

---

## 3. 安装 PyTorch（CPU 版本）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

验证：
```bash
python -c "import torch; print(torch.__version__)"
```

---

## 4. 安装项目依赖

将项目代码复制到工作站（或 git clone），进入项目目录：

```bash
cd C:\Users\你的用户名\defectDetection
pip install -r requirements.txt
```

> 如果下载慢，使用清华 pip 镜像：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 5. 安装 MindVision 相机 SDK

1. 从 MindVision 官网下载 SDK：
   - http://www.mindvision.com.cn/  → 技术支持 → SDK下载
   - 或使用随相机附带的 U 盘/光盘中的 SDK 安装包
2. 运行 `MVCAMSDK.exe` 安装程序
3. 默认安装路径：`C:\Program Files\MindVision\MVCAMSDK`
4. 安装完成后确认 DLL 存在：
   ```
   dir "C:\Program Files\MindVision\MVCAMSDK\Runtime\Win64_x64\MVCAMSDK_X64.dll"
   ```
5. 将 Runtime 路径添加到系统 PATH：
   - 右键"此电脑" → 属性 → 高级系统设置 → 环境变量
   - 在"系统变量"中找到 `Path`，编辑，添加：
     ```
     C:\Program Files\MindVision\MVCAMSDK\Runtime\Win64_x64
     ```
   - 确定保存

> 如果不使用 MindVision 相机（仅用图片测试），可跳过此步骤。程序在无相机时不会崩溃。

---

## 6. 运行程序

```bash
conda activate defect
cd C:\Users\你的用户名\defectDetection
python app.py
```

如果一切正常，PySide6 窗口会弹出。

---

## 7. 使用流程

1. **连接相机**：USB 连接 MindVision 工业相机，上电
2. **Live View 模式**：启动后默认为 Live View，点击 "Grab" 测试相机画面
3. **设置模板**：将完美产品放在相机下方，点击 "Set Template"
4. **训练**：点击 "Train"，等待训练完成（CPU 约 1-3 分钟）
5. **检测**：
   - 点击 "Inspect"：单次手动检测
   - 点击 "Auto Inspect"：连续自动检测
6. **调整阈值**：工具栏的 Threshold spinner 控制合格/缺陷判定边界

---

## 8. （可选）安装 VC++ 运行时

如果运行时报缺少 `vcruntime140.dll` 等错误：

1. 下载微软 VC++ 运行时：
   - https://aka.ms/vs/17/release/vc_redist.x64.exe
2. 运行安装
3. 重启电脑

---

## 9. （可选）创建桌面快捷方式

创建 `DefectDetection.bat` 文件放到桌面：

```bat
@echo off
call C:\Users\你的用户名\miniconda3\Scripts\activate.bat defect
cd /d C:\Users\你的用户名\defectDetection
python app.py
pause
```

双击即可启动。

---

## 10. （可选）打包为 EXE

如果需要生成独立可执行文件（无需安装 Python）：

```bash
conda activate defect
pip install pyinstaller
pyinstaller defect_detection.spec
```

生成的程序在 `dist\DefectDetection\` 目录下，可将整个文件夹复制到其他 Windows 10 机器运行（仍需安装 MindVision SDK 和 VC++ 运行时）。

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| `ModuleNotFoundError: No module named 'torch'` | 确认已 `conda activate defect` 再运行 |
| `CUDA visible devices warning` | 正常，程序强制使用 CPU |
| 相机打不开 | 确认 SDK 已安装且 Runtime 路径在系统 PATH 中 |
| 画面全黑 | 检查相机镜头盖是否取下，调整曝光时间 |
| `vcruntime140.dll not found` | 安装 VC++ 运行时（步骤 8） |
| pip 下载超时 | 使用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
