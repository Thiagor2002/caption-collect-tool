# caption-collect-tool

从 `test_videos` 中选定的视频里提取**讲解字幕**，并在字幕基础上**整理成文章**的 Python 命令行工具。

- 语音转写：[faster-whisper](https://github.com/SYSTRAN/faster-whisper)（CTranslate2 实现，纯 CPU 即可运行）
- 音频解码：PyAV（自带 ffmpeg 库，**无需在系统里安装 ffmpeg**）
- 成文：LLM 模式（任意 OpenAI 兼容接口，长视频自动 map-reduce）或本地抽取式模式（离线、无需 Key）

## 功能特性

| 能力 | 说明 |
| --- | --- |
| 字幕提取 | 输出带时间轴的 `.srt`、纯文本讲解稿 `.txt`、结构化 `.json`（含转写元信息） |
| 词级断句 | 开启词级时间戳后按标点/停顿/长度切分，单条字幕不超过 40 字、8 秒，`.srt` 自动折行 |
| 口语清洗 | 去除口水词、音效标签、多余空格与重复标点，过滤 Whisper 循环幻觉产生的重复句 |
| 繁简统一 | Whisper 输出常混入繁体，工具用 OpenCC 统一转为简体（`--keep-traditional` 可关闭） |
| 术语校对 | 内置同音错字词典（人名/书名/成语等专有名词），命中数写入文章的「校对记录」便于复核；可用外部词典补充本期术语 |
| 补读点 | ASR 在长段讲解里常整段不带标点；工具在 **jieba 词边界**上补逗号/句号（连接词前断句），只加标点不改用字，`--no-pad-punct` 可关闭 |
| 分期分章 | 拼接了多期节目的长视频，按开场白/告别语自动切分期，每期再按停顿与字数分节 |
| 文章生成 | LLM 模式生成结构化文章；无 Key 时自动回退本地抽取式整理（TextRank 提取要点、目录、每期要点） |
| 增量运行 | 已有 `transcript.json` 时跳过转写，可用 `article` 子命令只重跑校对与成文 |
| 批量处理 | `--all` 一次处理视频目录中的全部视频，单个失败不影响其余 |

## 环境要求

已在如下环境验证：WSL Ubuntu + conda 环境 `myenv`（Python 3.11），16 核 CPU / 7GB 内存，无 GPU。

```bash
# 1. 进入 WSL 并激活环境
wsl -d Ubuntu
source ~/anaconda3/etc/profile.d/conda.sh && conda activate myenv

# 2. 安装依赖（使用清华镜像更快）
cd /mnt/d/Repositories/caption-collect-tool
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 快速开始

在 WSL 中（推荐）：

```bash
cd /mnt/d/Repositories/caption-collect-tool

python -m caption_tool list                      # 查看视频与已生成结果
python -m caption_tool run 1 --model small       # 转写第 1 个视频并成文（默认）
python -m caption_tool run 1 --model medium      # 追求字幕质量，速度约为 small 的 1/3
python -m caption_tool run --all --writer local  # 批量处理，纯本地成文
python -m caption_tool article 苏共兴亡史        # 只重跑校对 + 成文（改完词典后用，秒级）
```

也可以直接在 Windows PowerShell 里一行调用：

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/d/Repositories/caption-collect-tool && source ~/anaconda3/etc/profile.d/conda.sh && conda activate myenv && python -m caption_tool run 1"
```

长视频建议先试跑前几分钟，确认字幕质量后再跑全片：

```bash
python -m caption_tool run 1 --limit-seconds 300 --output-dir /tmp/preview
```

## 命令说明

### `list` — 列出视频

```bash
python -m caption_tool list
```

```
视频目录：/mnt/d/Repositories/caption-collect-tool/test_videos（共 2 个）
  [1] 苏共兴亡史-文.mp4  233MB  1小时15分40秒  已生成: 字幕/文章
  [2] 马克思主义发现史-文.mp4  204MB  1小时34分22秒  已生成: 字幕/文章
```

### `run` — 转写 + 成文

```bash
python -m caption_tool run [视频...] [选项]
```

视频可用**序号**（`1`）、**文件名**（`苏共兴亡史-文.mp4`）、**去尾缀名**（`苏共兴亡史`）或**通配符**（`*兴亡*`）指定；也可用 `--all` 处理全部。

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--all` | - | 处理视频目录中的全部视频 |
| `--videos-dir` | `test_videos` | 视频目录 |
| `--output-dir` | `outputs` | 输出目录 |
| `--model` | `small` | Whisper 模型：`tiny`/`base`/`small`/`medium`/`large-v3` |
| `--language` | `zh` | 语言代码，`auto` 表示自动检测 |
| `--device` | `cpu` | `cpu` 或 `cuda` |
| `--compute-type` | `int8` | CPU 推荐 `int8`，GPU 推荐 `float16` |
| `--beam-size` | `5` | 解码 beam size，越小越快 |
| `--cpu-threads` | `0` | CPU 线程数，`0` 为自动 |
| `--no-vad` | - | 关闭 VAD 静音过滤 |
| `--keep-traditional` | - | 保留繁体输出，不做繁转简 |
| `--initial-prompt` | 内置中文讲座提示词 | 自定义提示词，可写入课程术语引导识别 |
| `--limit-seconds` | - | 只处理前 N 秒，用于快速试跑 |
| `--force` | - | 已有字幕时强制重新转写 |
| `--writer` | `auto` | 成文模式：`auto`/`llm`/`local` |
| `--chunk-chars` | `12000` | LLM 单段处理的最大字数，超出则分段整理后汇总 |
| `--title` | 由文件名推导 | 文章标题（单视频时生效） |
| `--corrections` | 自动查找 `corrections.txt` | 外部纠错词典，每行 `误写=正确`，与内置词典合并 |
| `--no-proofread` | - | 关闭术语校对，直接输出原始转写文本 |
| `--no-pad-punct` | - | 关闭补读点，完全保留 ASR 原始标点 |

`run` 与 `article` 共用上述成文/校对选项，`article` 额外接受输出目录名（缺省处理 `outputs` 下全部结果）。

### `article` — 只重新生成文章

已有字幕时，切换成文模式或调整标题无需重新转写：

```bash
python -m caption_tool article                        # 对 outputs 下全部结果重新成文
python -m caption_tool article 苏共兴亡史 --writer llm  # 指定单个，用 LLM 重写
```

## 输出结构

每个视频在 `outputs/` 下有一个同名目录：

```
outputs/
└── 苏共兴亡史/
    ├── transcript.json   # 原始转写结果（未校对、未补标点），复用与重跑的依据
    ├── transcript.srt    # 校对后的带时间轴字幕（逐条文字不补读点），可直接导入播放器
    ├── transcript.txt    # 校对 + 补读点后的纯文本讲解稿，按自然段分段
    ├── article.md        # 校对后整理的文章（目录 / 分期分章 / 每期要点 / 校对记录）
    ├── corrections.txt   # 可选：本视频的专属纠错词典，存在则自动加载
    └── 校对报告.md       # 可选：人工复核记录（已修正项 + 存疑项时间码），非工具生成
```

`transcript.json` 始终保存原始转写，`srt` / `txt` / `article.md` 是由它加上校对规则派生的视图。
因此改完词典只需重跑 `article`（秒级），不必重新转写。

词典查找顺序：`--corrections 指定路径` → `outputs/<视频>/corrections.txt` → 仓库根 `corrections.txt`。
文件格式为每行一条 `误写=正确`，`#` 开头为注释；内置词典只收录专有名词与"根本不成词"的同音串，
普通词一律不改，避免把正确文本改错。

## LLM 成文配置（可选）

不配置任何 Key 也能用（自动走本地抽取式模式）。要使用 LLM：

```bash
cp .env.example .env
```

编辑 `.env`：

```ini
LLM_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://api.deepseek.com/v1   # 任意 OpenAI 兼容接口
LLM_MODEL=deepseek-chat
```

`--writer auto`（默认）会在 Key 缺失或调用失败时自动回退到本地模式，不会中断流程；`--writer llm` 则强制使用 LLM 并在失败时报错。

字幕超过 `--chunk-chars` 时采用 map-reduce：先分段生成详细笔记，再汇总成完整文章，因此 1 小时以上的讲座也能处理。

## 模型与性能参考

同一台机器实测（WSL 16 核 CPU / 7GB 内存、`int8`、`beam_size=5`、词级时间戳；测试片长 75 分 40 秒与 94 分 22 秒）：

| 模型 | 体积 | 中文效果 | 实测速度 | 全片耗时 | 适用场景 |
| --- | --- | --- | --- | --- | --- |
| `tiny` / `base` | 小 | 一般 | 最快 | - | 快速预览、内容筛选 |
| `small` | 484MB | 可用，专名同音错字多 | 约 3.0x 实时 | 24 分 54 秒 | **默认**，日常批量整理 |
| `medium` | 1.5GB | 明显更好 | 约 1.0–1.3x 实时（约用满 4 核） | 75 分钟片 1小时12分；94 分钟片 1小时11分 | 追求字幕质量 |
| `large-v3` | 3GB+ | 最好 | 很慢 | - | 有 GPU 时使用 |

`medium` 首次运行还需下载 1.5GB 模型（约 800KB/s，需十几分钟）；模型只下载一次，之后走缓存。
同一片子 `small` 转写出 690 条字幕需 95 处术语校对，`medium` 只需 16 处——专名错字大幅下降，
剩下的多是同音词与引文，靠 `corrections.txt` 修。

首次运行会从 Hugging Face 下载模型（缓存于 `~/.cache/huggingface`）。若下载缓慢或卡住，可改用镜像并关闭 xet 传输：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

## 常见问题

- **转写结果有重复句**：工具已自动过滤连续重复的循环幻觉；仍严重时加 `--no-vad` 对比或换更大的模型。
- **同音错字**（如"兴亡史"识别为"新网史"、"逝世"识别为"事事"）：三道防线依次使用——
  ① 内置词典会自动改写专名类误写，并在文章末尾输出「校对记录」表列出命中次数；
  ② 把本期特有的误写补进 `outputs/<视频>/corrections.txt`（`误写=正确`），重跑 `article` 即可，无需重新转写；
  ③ 仍大面积出错时换 `--model medium`，或用 `--initial-prompt` 把课程术语写进提示词引导识别，或配好 `.env` 后用 `--writer llm` 由大模型据上下文改写。
- **校对会不会改错**：内置规则只收录"原写法根本不成词"的串（人名、书名、成语误写），
  不做普通词的全局替换；外部词典由你自己维护，且文章内的「校对记录」列出每一处替换的命中次数，可逐条复核。
- **长段没有标点**：`medium` 在连续讲解的长片段里标点率会明显下降（本次 94 分钟的片子有 47/61 段几乎无标点）。工具默认在 jieba 词边界补逗号/句号，只加标点不动用字；要完全保留 ASR 原样加 `--no-pad-punct`。要真正顺句可读，仍建议 `--writer llm`。
- **"它/他"混用**：抽查全部指人候选后，绝大多数"它"确实指主义/理论/著作（官方译文亦如此，如"它是一切社会关系的总和"），因此不做全局替换；个别说人的位置请在 `corrections.txt` 里用**完整短语**逐条指定。
- **输出混入繁体字**：默认已用 OpenCC 转简体，若未安装 `opencc-python-reimplemented` 则跳过该步骤（`pip install -r requirements.txt` 可补齐）。
- **专业术语识别不准**：见上条 `--initial-prompt`。
- **内存吃紧**：用 `--limit-seconds` 分段处理，或换 `small`/`base` 模型。
- **想重跑某一步**：加 `--force` 重新转写；只改文章用 `article` 子命令。

## 项目结构

```
caption_tool/
├── __main__.py     # python -m caption_tool 入口
├── cli.py          # 命令行解析与流程编排（list / run / article）
├── config.py       # 路径常量、.env 解析、LLM 配置
├── audio.py        # PyAV 音频解码与媒体信息探测
├── asr.py          # faster-whisper 转写封装（模型缓存、进度回调）
├── transcript.py   # 字幕数据模型、清洗断句、srt/txt/json 导出
├── proofread.py    # 同音错字校对词典与命中统计（可外部词典扩充）
└── article.py      # 成文：LLM（map-reduce）与本地抽取式（分期分章 + TextRank）
```

## License

见 [LICENSE](LICENSE)。
