# YirenIngest — 《异人之下》攻略视频摄取

把攻略视频/音频转成结构化文本：**转写 → 术语精修 → 章节 + 全文总览**。

本仓库从 EduNUWA 的 M1 Ingest 拆分而来，去掉 PDF/PPTX 与 TTS 音色样本，只保留视频/音频主链路，领域改为《异人之下》PVP。代码不依赖原项目。

```
视频/音频 → 提取音频 → 长音频切分 → ASR 转写 → LLM 精修 → 章节+总览摘要
                                      (whisper/funasr/cloud)
```

---

## 1. 环境要求

| 项目 | 要求 |
|---|---|
| Python | 3.10（已验证 3.10.20） |
| 系统 | Windows 11 已验证；Linux/macOS 理论可用但未实测 |
| ffmpeg | 必需，需在 PATH 中 |
| 网络 | 首次运行需下载 ASR 模型（数 GB） |
| GPU | 可选。NVIDIA GPU 会显著加速；无 GPU 自动降级 CPU |
| LLM Key | DeepSeek API Key（精修 + 章节摘要） |

仓库**不包含**：`.env`、`data/`、模型权重、样本视频。这些都要在新机器上自行准备。

---

## 2. 安装

先获取代码（私有仓库需要登录/授权）：

```bash
git clone https://github.com/<your-account>/YirenIngest.git
cd YirenIngest
```

以下命令都在仓库根目录执行。

### 2.1 创建 Python 环境

推荐 conda：

```bash
conda create -n yiren python=3.10 -y
conda activate yiren
```

也可以使用 venv：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
```

### 2.2 安装 PyTorch（GPU 用户）

如果你有 NVIDIA GPU，先安装匹配 CUDA 的 torch / torchaudio（以 CUDA 12.8 为例）：

```bash
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

CUDA 版本按 `nvidia-smi` 和 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择。

无 GPU 可跳过这一步，直接执行下一步安装 CPU 版；如果希望明确装 CPU 轮子（避免 Linux 上默认 CUDA 包过大），可以执行：

```bash
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2.3 安装项目依赖

```bash
pip install -r requirements.txt
```

`torchaudio` 必须与 `torch` 版本一致；FunASR 的音频前端会 import 它。

### 2.4 安装 ffmpeg

- Windows：`winget install Gyan.FFmpeg` 或 `choco install ffmpeg -y`
- Ubuntu / Debian：`sudo apt install ffmpeg`
- macOS：`brew install ffmpeg`

安装后在终端确认：

```bash
ffmpeg -version
```

### 2.5 配置 .env

```powershell
# Windows PowerShell
copy .env.example .env
```

```bash
# Linux / macOS
cp .env.example .env
```

编辑 `.env`，填入 DeepSeek Key：

```dotenv
DEEPSEEK_API_KEY=sk-...
```

可选：

- 中国大陆访问 HuggingFace 慢时，取消注释 `HF_ENDPOINT=https://hf-mirror.com`
- 使用 Cloud ASR 时配置 `DASHSCOPE_API_KEY`

`.env` 必须放在仓库根目录，且已被 gitignore，不会提交。

---

## 3. 验证安装

```bash
python -c "import torch, torchaudio, funasr, faster_whisper, moviepy; print(torch.__version__, torchaudio.__version__, 'cuda=', torch.cuda.is_available())"

python -m unittest discover -s tests -v
```

第一条输出 `cuda= True` 表示 GPU 可用；`False` 表示会走 CPU（更慢，但功能可用）。

---

## 4. 首次运行与模型下载

- **FunASR（默认）**：首次运行自动从 ModelScope 下载 `paraformer-zh` / `fsmn-vad` / `ct-punc`
- **Whisper**：首次运行自动从 HuggingFace 下载所选模型；`large-v3` 体积最大
- 下载完成后缓存在用户目录，后续运行不再下载

默认缓存位置：`~/.cache/huggingface`（Whisper）与 `~/.cache/modelscope`（FunASR）。如果目标机器不能联网，可以从本机复制这两个目录。

请预留足够磁盘空间和网络时间。

---

## 5. 使用

### 5.1 CLI

准备好自己的视频/音频文件（仓库不附带样本）：

```bash
python -m yiren_ingest.run --series_id zhangchulan_guide --source "./videos/ep01.mp4"
```

批量目录：

```bash
python -m yiren_ingest.run --series_id zhangchulan_guide --source_dir "./videos" --force_reprocess
```

只跑 ASR，关闭 LLM 精修和摘要：

```bash
python -m yiren_ingest.run \
  --series_id asr_test \
  --source "./videos/ep01.mp4" \
  --asr_backend funasr \
  --no-refine --no-summary
```

切换到 Whisper：

```bash
python -m yiren_ingest.run \
  --series_id asr_test \
  --source "./videos/ep01.mp4" \
  --asr_backend whisper --asr_model large-v3 \
  --no-refine --no-summary
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--source` / `--source_dir` | 单个/多个文件，或递归扫描目录 |
| `--output` | 输出目录，默认 `data/series/{series_id}` |
| `--asr_backend` | `funasr`（默认）/ `whisper` / `cloud` / `auto` |
| `--asr_device` | `auto`（默认）/ `cuda` / `cpu` |
| `--asr_model` | Whisper 模型：`large-v3` / `medium` / `small` |
| `--no-refine` / `--no-summary` | 关闭 LLM 精修 / 章节摘要 |
| `--force_reprocess` | 忽略幂等缓存，强制重跑 |

### 5.2 Web UI

```bash
python -m yiren_ingest.launcher
```

浏览器会自动打开。操作顺序：

1. 填「系列 ID」（如 `zhangchulan_guide`）
2. 添加视频所在文件夹
3. 选择 ASR 引擎（默认 FunASR）
4. 按需关闭「LLM 精修」/「章节 + 全文总览」做纯 ASR 对比
5. 点击窗口底部的绿色「启动流水线」

启动后进入任务监控面板；完成后点「＋ 新任务」回到配置页。

可选参数：

```bash
python -m yiren_ingest.launcher --port 8080 --no-browser
```

Linux 下文件夹选择框需要 `python3-tk`（`sudo apt install python3-tk`）。无桌面环境（纯 SSH 服务器）请直接使用 CLI。

---

## 6. 输出

默认输出到 `data/series/{series_id}/transcripts/TR_{series}_{seq}.json`：

```json
{
  "transcript_id": "TR_zhangchulanguide_001",
  "series_id": "zhangchulan_guide",
  "source_file": "ep01.mp4",
  "source_audio": "videos/ep01.mp4",
  "language": "zh",
  "asr_backend": "whisper-large-v3",
  "asr_quality": {"estimated_cer": 0.13, "low_confidence_segments": 0, "refined": true},
  "overview": {"title": "...", "summary": "...", "key_points": ["..."]},
  "chapters": [
    {"chapter_id": "ch_001", "title": "...", "start": 0.0, "end": 60.0,
     "summary": "...", "key_points": ["..."]}
  ],
  "segments": [
    {"segment_id": "seg_0001", "start": 0.0, "end": 6.5,
     "text": "...", "speaker": "narrator", "confidence": 0.85}
  ]
}
```

其它产物：

- `_ingest_index.json` — 幂等索引（文件名 + 大小 + 修改时间）
- `_logs/ingest_{upload_id}.log` — 运行日志
- `_tmp/` — 临时提取音频，成功后清理

---

## 7. 术语库（最重要）

单一真相源：`yiren_ingest/domain_terms.py`

| 变量 | 用途 |
|---|---|
| `GAME_DOMAIN_TERMS` | Whisper `initial_prompt` |
| `GAME_HOTWORDS` | FunASR 热词 |
| `TERM_CORRECTIONS` | ASR 同音错字纠正表（如 炁/气、险招/显招） |
| `LLM_TERM_REFERENCE` | 精修/摘要 prompt 的完整术语参考 |

术语来自《异人之下》PVP 热词库（资源判定、操作博弈、身外身、对局模式、角色机制、主播黑话、同音易错）。修改后下一次运行自动生效；已处理文件需加 `--force_reprocess` 才会重跑。

---

## 8. 设备与性能

- 默认 `device: auto`：有 CUDA 用 GPU，否则自动走 CPU
- 默认 ASR 引擎 FunASR（中文游戏攻略表现稳定、速度较快）
- Whisper `large-v3` 精度最高但显存占用大；显存不足可改 `medium` / `small`
- 后处理（精修 + 摘要）走网络 API，默认并发 10；遇到限流可调小

不要修改已提交的 `yiren_ingest/hardware_config.yaml` 来适配本机，创建 `yiren_ingest/hardware_config.local.yaml` 覆盖（已 gitignore）：

```yaml
asr:
  model: "medium"
  device: "cpu"
refine:
  workers: 4
```

---

## 9. 目录结构

```
yiren_ingest/
├── run.py                     CLI
├── launcher.py                Web UI（HTTP + SSE）
├── pipeline_window.py         进度状态机
├── domain_terms.py            领域术语库（改这里）
├── llm_client.py              OpenAI 兼容 LLM 客户端
├── asr_pipeline/              whisper / funasr / cloud 三引擎
├── audio_processing/          音轨提取 + 长音频切分
├── ingest_orchestrator/       编排 / 异步任务 / 进度追踪
├── text_cleaning/             规则清洗 + LLM 精修
├── summarizer/                章节 + 全文总览
├── utils.py                   .env 加载、原子写
└── hardware_config.yaml       跨机器默认配置
```

---

## 10. 当前边界

- 只处理视频里的**语音**，不解析画面（不抽帧、不做 OCR/VLM）
- 只支持视频/音频，不支持 PDF/PPTX
- 不生成 TTS 音色样本
- 章节摘要超长输入（>12 万字）会降级保留 transcript，不调用 LLM
- 仓库不附带测试媒体；`samples/` 目录仅保留占位文件

---

## 11. 测试

```bash
python -m unittest discover -s tests -v
```

单元测试只覆盖纯函数与配置，不需要模型、视频或 API Key。

---

## 12. 常见问题

| 现象 | 处理 |
|---|---|
| `ffmpeg not found` | 安装 ffmpeg 并确认在 PATH；Windows 重开终端 |
| `No module named 'torchaudio'` | 安装与 torch 同版本的 torchaudio |
| FunASR 模型下载失败 | 检查网络 / ModelScope；重试 |
| Whisper 模型下载失败 | 在 `.env` 配置 `HF_ENDPOINT=https://hf-mirror.com` |
| 模型已下载但机器离线 | 可设 `HF_HUB_OFFLINE=1`；FunASR 直接使用本地 ModelScope 缓存 |
| 有 GPU 但 `cuda=False` | 装错成 CPU 版 torch；按 PyTorch 官网重装 CUDA 版 |
| 显存不足 / OOM | 改用 FunASR，或 Whisper 换 `medium`；用 local 配置覆盖 |
| 启动时提示 API Key | 确认 `.env` 在仓库根目录且变量名为 `DEEPSEEK_API_KEY` |
| UI 端口被占用 | 启动器会自动换端口；也可 `--port 8080` |
| Windows 控制台中文乱码 | 执行 `chcp 65001` 后重开 |
