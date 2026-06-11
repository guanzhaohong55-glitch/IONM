# IONM JSON Waveform Viewer

这个目录中的 JSON 是 Cadwell 导出的术中神经监测病例数据，主要结构如下：

- `Cases`: 病例列表，本数据集每个 JSON 通常只有 1 个病例。
- `Cases[].Events`: 手术过程事件，包含 `Timestamp`、`Type`、`Message`、`Deleted`。
- `Cases[].Modes`: 监测模式，例如 Lower SSEP、MEP、TOF、DNEP。
- `Modes[].Trials`: 每次刺激/采集的试次，包含试次时间、刺激参数、波形。
- `Trials[].Traces`: 每个通道的一条波形，核心字段是：
  - `TraceData`: 原始采样点。
  - `TraceDataScalar`: 原始数值乘以该系数后得到伏特 V。
  - `TraceDataLength`: 采样点数，当前数据多为 640。
  - `Sweep`: 单条波形窗口长度，SSEP/MEP 多为 0.1s，TOF 多为 0.02s。
  - `Cursors`: 峰潜伏期/幅值标记，例如 `N37`、`P45`、`P`、`T`。

推荐运行方式：

```powershell
.\run_viewer.bat
```

也可以手动运行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe ionm_viewer.py
```

界面会自动扫描当前目录下的病例 `*.json`，并跳过 `.venv` 等依赖目录。波形数值会按 `TraceData * TraceDataScalar` 转换为电压，并按模式自动显示为 `uV` 或 `mV`；横轴按 `Sweep / TraceDataLength` 换算为毫秒。

导出文件中的时间戳按 Unix 微秒解释后与事件文本中的手术记录时间相符；界面同时显示绝对时间和以病例 `StartDate` 为零点的相对手术时间。
