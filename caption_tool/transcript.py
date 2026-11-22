"""字幕数据模型：清洗、断句合并与导出（srt / txt / json）。"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# 音效/舞台提示，例如 [音乐]、(掌声)、【片头】、♪...♪
TAG_RE = re.compile(
    r"[\[\(（【{](?:音乐|掌声|笑声|片头|片尾|字幕|广告|背景音|blues|music|applause|laughter)[^\]\)）】}]{0,20}[\]\)）】}]",
    re.IGNORECASE,
)
NOTE_RE = re.compile(r"♪[^♪]{0,40}♪")
CONTROL_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
SPACE_RE = re.compile(r"[ \t\u3000\u00a0\u2000-\u200a\u202f\u205f]+")
CJK_SPACE_RE = re.compile(
    r"(?<=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])[ \t]+(?=[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef])"
)
AFTER_PUNCT_SPACE_RE = re.compile(r"(?<=[，。！？；：、])\s+")
BEFORE_PUNCT_SPACE_RE = re.compile(r"\s+(?=[，。！？；：、])")
LEADING_FILLER_RE = re.compile(r"^(?:嗯+|呃+|啊+|哦+|唉+|哎+|那么，?|这个，?|就是说，?|然后，?)[，、,]?")
TRAILING_FILLER_RE = re.compile(r"[，、,]?(?:嗯+|呃+|啊+|哦+|唉+|哎+)[，、,。！？]?$")
PUNCT_REPEAT_RE = re.compile(r"([。！？；，、：])\1+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
ASCII_PUNCT_MAP = {",": "，", ";": "；", ":": "：", "?": "？", "!": "！"}

SENTENCE_END = "。！？!?;"
BREAK_PUNCT = "。！？；!?;，、,："

_OPENCC = None


@dataclass
class Utterance:
    """一句（或一小段）讲解字幕。"""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    source: str
    utterances: list[Utterance] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(u.text for u in self.utterances)

    @property
    def full_text(self) -> str:
        return "".join(u.text for u in self.utterances)

    def char_count(self) -> int:
        return sum(len(u.text) for u in self.utterances)

    def to_json(self) -> dict:
        return {
            "meta": {**self.meta, "source": self.source, "exported_at": _now()},
            "utterances": [asdict(u) for u in self.utterances],
        }


def simplify(text: str) -> str:
    """繁体转简体（安装了 opencc-python-reimplemented 时生效）。"""
    global _OPENCC
    if _OPENCC is None:
        try:
            from opencc import OpenCC

            _OPENCC = OpenCC("t2s")
        except Exception:  # noqa: BLE001
            _OPENCC = False
    if _OPENCC:
        try:
            return _OPENCC.convert(text)
        except Exception:  # noqa: BLE001
            return text
    return text


def normalize_punct(text: str) -> str:
    """中文语境下把半角标点转成全角。"""
    if not CJK_RE.search(text):
        return text
    return "".join(ASCII_PUNCT_MAP.get(ch, ch) for ch in text)


def clean_text(text: str, *, do_simplify: bool = True) -> str:
    """清洗 ASR 文本：去提示音标签、口水词、多余空格，统一简繁与标点。

    标点空格清理必须放在标点全角化之后，否则半角 ``", "`` 会转成 ``"， "`` 而残留空格。
    """
    text = CONTROL_RE.sub("", text or "")
    text = TAG_RE.sub("", text)
    text = NOTE_RE.sub("", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = SPACE_RE.sub(" ", text)
    text = CJK_SPACE_RE.sub("", text)
    text = text.strip()
    if do_simplify:
        text = simplify(text)
    text = normalize_punct(text)
    text = AFTER_PUNCT_SPACE_RE.sub("", text)
    text = BEFORE_PUNCT_SPACE_RE.sub("", text)
    text = text.strip()
    text = LEADING_FILLER_RE.sub("", text).strip()
    text = TRAILING_FILLER_RE.sub("", text).strip()
    text = PUNCT_REPEAT_RE.sub(r"\1", text)
    return text.strip().lstrip("，、,；;：: 　").rstrip("、 　")


def iter_words(segments) -> list[Word]:
    """从 ASR 片段取出词级时间戳；无词级信息时退化为片段级。"""
    words: list[Word] = []
    for seg in segments:
        seg_words = getattr(seg, "words", None)
        if seg_words:
            for item in seg_words:
                token = (item.word or "").strip()
                if token:
                    words.append(Word(start=float(item.start), end=float(item.end), text=token))
        else:
            token = (seg.text or "").strip()
            if token:
                words.append(Word(start=float(seg.start), end=float(seg.end), text=token))
    return words


def _is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def _join(tokens: list[str]) -> str:
    """按中英文混排规则拼接词元：中日韩文字与全角标点之间不加空格。"""
    out = ""
    for token in tokens:
        if not token:
            continue
        if not out:
            out = token
            continue
        out += token if (_is_wide(out[-1]) and _is_wide(token[0])) else " " + token
    return out


def build_utterances(
    segments,
    *,
    max_chars: int = 40,
    max_seconds: float = 8.0,
    min_chars: int = 6,
    pause: float = 0.5,
    simplify_text: bool = True,
) -> list[Utterance]:
    """按标点、停顿与长度把词级时间戳合并成句子级字幕。"""
    words = iter_words(segments)
    utterances: list[Utterance] = []
    current: list[Word] = []

    def emit(chunk: list[Word]) -> None:
        text = clean_text(_join([w.text for w in chunk]), do_simplify=simplify_text)
        if len(text) >= 2:
            utterances.append(Utterance(start=round(chunk[0].start, 3), end=round(chunk[-1].end, 3), text=text))

    def flush() -> None:
        if current:
            emit(current[:])
            current.clear()

    def flush_at_punct() -> None:
        """超长时优先在最近一个标点处断句，避免把词切断。"""
        acc = 0
        cut = None
        for idx, word in enumerate(current):
            acc += len(word.text)
            if word.text[-1:] in BREAK_PUNCT and acc >= min_chars:
                cut = idx + 1
        if cut is None:
            flush()
            return
        emit(current[:cut])
        del current[:cut]

    for word in words:
        if current:
            length = sum(len(w.text) for w in current)
            span = current[-1].end - current[0].start
            gap = word.start - current[-1].end
            ended = current[-1].text[-1:] in SENTENCE_END
            long_enough = length >= min_chars
            if length >= max_chars or span >= max_seconds:
                flush_at_punct()
            elif long_enough and (ended or gap >= pause):
                flush()
        current.append(word)
    flush()
    return drop_repeats(utterances)


def drop_repeats(utterances: list[Utterance], window: int = 3) -> list[Utterance]:
    """丢弃 Whisper 循环幻觉产生的连续重复句。"""
    result: list[Utterance] = []
    for item in utterances:
        recent = [u.text for u in result[-window:]]
        if len(item.text) >= 4 and item.text in recent:
            if result:
                result[-1].end = item.end
            continue
        result.append(item)
    return result


def group_paragraphs(utterances: list[Utterance], *, max_chars: int = 240, gap: float = 1.5, protect=None) -> list[list[Utterance]]:
    """把句子按停顿与长度聚成自然段。

    只在句末标点或明显停顿处断开，避免把词/句切到两个段落里。
    ``protect(前句, 后句)`` 返回 True 时禁止在此断段（例如断点落在待校对词内部）。
    """
    paragraphs: list[list[Utterance]] = []
    current: list[Utterance] = []
    size = 0
    for item in utterances:
        if current:
            prev = current[-1]
            settled = prev.text[-1:] in SENTENCE_END
            overflow = size >= max_chars * 1.6
            paused = item.start - prev.end > gap
            # 数字/英文单词常被切分到相邻两条字幕，断开处绝不能再分段落
            split_token = prev.text[-1:].isdigit() and item.text[:1].isdigit()
            split_token = split_token or (prev.text[-1:].isascii() and prev.text[-1:].isalnum()
                                          and item.text[:1].isascii() and item.text[:1].isalnum())
            split_token = split_token or (prev.text[-1:] in HANGING)  # 别把"不只是"这类悬空词切到段尾
            if protect is not None and (settled or overflow) and (size >= max_chars or paused or overflow):
                split_token = split_token or bool(protect(prev.text, item.text))
            if (settled or overflow) and (size >= max_chars or paused or overflow) and not split_token:
                paragraphs.append(current)
                current, size = [], 0
        current.append(item)
        size += len(item.text)
    if current:
        paragraphs.append(current)
    return paragraphs


def format_clock(seconds: float) -> str:
    """文章里用的 时:分:秒（不含毫秒）。"""
    total = int(round(max(0.0, seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp(seconds: float, comma: bool = True) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    sep = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def wrap_text(text: str, width: int = 24) -> str:
    """字幕换行：优先在标点处折行，避免单行过长。"""
    if len(text) <= width:
        return text
    lines: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if len(buf) >= width:
            cut = max((buf.rfind(p) for p in "，、；：。！？ "), default=-1)
            if cut >= width * 0.5:
                lines.append(buf[: cut + 1].strip())
                buf = buf[cut + 1 :]
            else:
                lines.append(buf.strip())
                buf = ""
    if buf.strip():
        lines.append(buf.strip())
    return "\n".join(line for line in lines if line)


def to_srt(transcript: Transcript, width: int = 24) -> str:
    lines: list[str] = []
    for idx, item in enumerate(transcript.utterances, start=1):
        lines.append(str(idx))
        lines.append(f"{format_timestamp(item.start)} --> {format_timestamp(item.end)}")
        lines.append(wrap_text(item.text, width))
        lines.append("")
    return "\n".join(lines)


PAD_STRONG = {
    "因为", "但是", "所以", "所以说", "而且", "并且", "然而", "因此", "于是", "另外", "同时",
    "总之", "那么", "如果", "虽然", "不仅", "更别说", "反而", "其实", "当然", "也就是说",
    "就是说", "由此", "综上", "此外", "第一", "第二", "第三", "其次",
}
PAD_SOFT = {"大家", "我们", "你们", "他们", "她们", "它", "他", "她"}
PAD_CLAUSE_END = "着了过呢吧嘛啦呀也"   # 只有在小句收尾之后才允许在代词前断句
HANGING = set("不是的和与或就都也把被从对为以而及等在有么这那我你他她它要能会可所不")
PAD_MAX_RUN = 40       # 超过这么多字没有读点，就在词边界补一个逗号
PAD_SOFT_RUN = 24


def pad_punctuation(text: str) -> str:
    """在长句里补读点，改善 ASR 漏标点导致的整段无标点。

    只在分词边界上落笔（jieba），因此不会把词切开：
    强连接词前补句号；小句已收尾且累计够长时，在主语代词前补句号；超长处补逗号。
    仅做可读性加工，不改动任何用字。
    """
    if not text:
        return text
    import jieba

    out: list[str] = []
    run = 0
    for token in jieba.cut(text):
        if token[-1:] in "。！？；…":
            out.append(token)
            run = 0
            continue
        if token[-1:] in "，、：":
            out.append(token)
            run = 0
            continue
        if token in PAD_STRONG and run >= 10:
            out.append("。" if run >= 20 else "，")
            out.append(token)
            run = len(token)
            continue
        if token in PAD_SOFT and run >= PAD_SOFT_RUN and (out and out[-1][-1:] in PAD_CLAUSE_END):
            out.append("。")
            out.append(token)
            run = len(token)
            continue
        if run >= PAD_MAX_RUN:
            out.append("，")
            run = 0
        out.append(token)
        run += len(token)
    result = "".join(out).rstrip("，、：")
    if result and result[-1] not in "。！？；…":
        result += "。"
    return result


def to_txt(transcript: Transcript, transform=None, protect=None, pad: bool = False) -> str:
    """整段文本。``transform`` 用于段级再加工（例如跨条修复被切开的词）。"""
    parts: list[str] = []
    for paragraph in group_paragraphs(transcript.utterances, protect=protect):
        text = _join([u.text for u in paragraph])
        if transform:
            text = transform(text)
        parts.append(pad_punctuation(text) if pad else text)
    return "\n\n".join(parts) + "\n"


def save_transcript(transcript: Transcript, out_dir: str | Path, stem: str, skip: tuple[str, ...] = (),
                    transform=None, protect=None, pad: bool = False) -> dict[str, Path]:
    """写出 transcript.srt / transcript.txt / transcript.json，返回路径字典。

    ``skip`` 用于跳过其中某些格式（例如只更新字幕、保留原始 json）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "srt": out_dir / f"{stem}.srt",
        "txt": out_dir / f"{stem}.txt",
        "json": out_dir / f"{stem}.json",
    }
    if "srt" not in skip:
        paths["srt"].write_text(to_srt(transcript), encoding="utf-8")
    if "txt" not in skip:
        paths["txt"].write_text(to_txt(transcript, transform=transform, protect=protect, pad=pad), encoding="utf-8")
    if "json" not in skip:
        paths["json"].write_text(
            json.dumps(transcript.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return paths


def load_transcript(json_path: str | Path) -> Transcript:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    meta = dict(data.get("meta", {}))
    source = meta.pop("source", str(json_path))
    meta.pop("exported_at", None)
    utterances = [Utterance(**item) for item in data.get("utterances", [])]
    return Transcript(source=source, utterances=utterances, meta=meta)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))
