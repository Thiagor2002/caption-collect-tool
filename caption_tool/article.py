"""成文模块：LLM 生成（OpenAI 兼容接口）与本地抽取式整理两种模式。"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import format_duration
from .config import LLMConfig
from .transcript import Transcript, Utterance, HANGING, format_clock, group_paragraphs, pad_punctuation

CN_NUM = "一二三四五六七八九十"

# 多期节目拼接的视频：每期通常以问候开场、以"下个视频见"收尾
EPISODE_START_RE = re.compile(r"^(?:大家好|各位(?:同学|朋友|观众)?|哈喽|你好)")
EPISODE_END_RE = re.compile(r"(?:下个视频见|下期视频见|下集再见|我们下期见|下期再见)")

STOPWORDS = {
    "我们", "你们", "他们", "她们", "自己", "这个", "那个", "这些", "那些", "什么", "怎么", "这样", "那样",
    "一个", "就是", "因为", "所以", "但是", "可是", "如果", "那么", "然后", "现在", "今天", "这里", "那里",
    "大家", "同学", "老师", "比如说", "比如", "例如", "其实", "当然", "应该", "可以", "不是", "没有", "还是",
    "已经", "的时候", "一样", "非常", "特别", "觉得", "知道", "看到", "来说", "方面", "问题", "东西", "事情",
    "继续", "本期", "视频", "讲解", "这一段", "这部分", "内容", "方式", "东西方",
    "这种", "那种", "不可", "一切", "任何", "只是", "所谓",
    "the", "and", "that", "this", "with", "for", "you", "are",
}


class LLMError(RuntimeError):
    """LLM 调用失败。"""


@dataclass
class ArticleResult:
    markdown: str
    mode: str  # "llm" | "local"
    note: str = ""


# --------------------------------------------------------------------------- #
# 通用文本分析
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    import jieba

    tokens = []
    for tok in jieba.lcut(text):
        tok = tok.strip().lower()
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        if not any(ch.isalnum() for ch in tok):
            continue
        tokens.append(tok)
    return tokens


def textrank_scores(token_lists: list[list[str]], damping: float = 0.85, iterations: int = 30) -> np.ndarray:
    """句子级 TextRank 打分。"""
    n = len(token_lists)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.ones(1, dtype=np.float32)

    sets = [set(t) for t in token_lists]
    norms = np.array([np.log(len(s) + 1) + 1e-6 for s in sets], dtype=np.float32)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        si = sets[i]
        if not si:
            continue
        for j in range(i + 1, n):
            sj = sets[j]
            if not sj:
                continue
            overlap = len(si & sj)
            if overlap:
                value = overlap / (norms[i] + norms[j])
                sim[i, j] = value
                sim[j, i] = value

    weights = sim.sum(axis=1)
    weights[weights == 0] = 1e-6
    norm = sim / weights[:, None]
    scores = np.full(n, 1.0 / n, dtype=np.float32)
    for _ in range(iterations):
        prev = scores
        scores = (1 - damping) / n + damping * norm.T.dot(scores)
        if np.abs(scores - prev).sum() < 1e-6:
            break
    return scores


def split_sections(utterances: list[Utterance], *, target_chars: int = 1500, min_chars: int = 500) -> list[list[Utterance]]:
    """按累计字数与停顿把字幕切成若干章节。

    达到 ``target_chars`` 后优先在停顿处断开；若一路没有停顿，则在 ``target_chars * 1.4``
    处强制断句，避免出现超长章节。
    """
    hard_limit = int(target_chars * 1.4)
    sections: list[list[Utterance]] = []
    current: list[Utterance] = []
    size = 0
    for item in utterances:
        if current and size >= target_chars and (item.start - current[-1].end >= 1.2 or size >= hard_limit):
            sections.append(current)
            current, size = [], 0
        current.append(item)
        size += len(item.text)
    if current:
        if sections and size < min_chars // 2:
            sections[-1].extend(current)
        else:
            sections.append(current)
    return sections


def section_keywords(utterances: list[Utterance], topk: int = 3) -> list[str]:
    from collections import Counter

    counter: Counter[str] = Counter()
    for item in utterances:
        counter.update(tokenize(item.text))
    return [word for word, _ in counter.most_common(topk * 3) if word not in STOPWORDS][:topk]


def split_episodes(utterances: list[Utterance], *, min_chars: int = 500) -> list[list[Utterance]]:
    """把拼接在一起的多期节目按开场问候切成若干期。"""
    episodes: list[list[Utterance]] = []
    current: list[Utterance] = []
    size = 0
    for item in utterances:
        if current and size >= min_chars and EPISODE_START_RE.match(item.text):
            episodes.append(current)
            current, size = [], 0
        current.append(item)
        size += len(item.text)
    if current:
        if episodes and size < min_chars * 0.4:
            episodes[-1].extend(current)
        else:
            episodes.append(current)
    return episodes


def pick_highlights(utterances: list[Utterance], scores, count: int = 5, min_len: int = 10) -> list[int]:
    """在给定范围内挑出互不相邻的高分句下标（相对于 utterances）。"""
    if len(utterances) == 0:
        return []
    order = np.argsort(-np.asarray(scores, dtype=np.float32))
    picked: list[int] = []
    for raw in order:
        idx = int(raw)
        if len(utterances[idx].text) < min_len:
            continue
        if any(abs(idx - other) <= 2 for other in picked):
            continue
        picked.append(idx)
        if len(picked) >= count:
            break
    return sorted(picked)


def cn_index(n: int) -> str:
    if n <= 10:
        return CN_NUM[n - 1]
    if n < 20:
        return "十" + CN_NUM[n - 11]
    tens, ones = divmod(n, 10)
    return CN_NUM[tens - 1] + "十" + (CN_NUM[ones - 1] if ones else "")


def _trim(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in ("，", "、", "；", " "):
        pos = cut.rfind(sep)
        if pos > limit * 0.5:
            cut = cut[:pos]
            break
    return cut.rstrip("，、；： ") + "……"


TITLE_SUFFIXES = ("-文", "_文", " 文", "（文）", "(文)", "- 文")


def clean_stem(name: str) -> str:
    """去掉扩展名与常见尾缀，得到可用作标题/目录名的字符串。"""
    stem = name.rsplit(".", 1)[0] if "." in name else name
    changed = True
    while changed:
        changed = False
        for suffix in TITLE_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem.strip(" -_·、,，.")


def derive_title(transcript: Transcript, fallback: str = "讲解整理") -> str:
    source = transcript.meta.get("source", "") or transcript.source
    return clean_stem(Path(source).name) or fallback


# --------------------------------------------------------------------------- #
# 本地抽取式成文
# --------------------------------------------------------------------------- #
def _splits_token(prev_text: str, next_text: str) -> bool:
    """相邻两条字幕是否在数字/英文单词或悬空虚词中间被切开。"""
    if not prev_text or not next_text:
        return False
    a, b = prev_text[-1:], next_text[:1]
    if a in HANGING:
        return True
    if a.isdigit() and b.isdigit():
        return True
    return a.isascii() and a.isalnum() and b.isascii() and b.isalnum()


def article_paragraphs(utterances: list[Utterance], *, target: int = 300, transform=None, protect=None,
                       pad: bool = False) -> list[str]:
    """按句末标点把字幕拼成自然段，避免一句话被拆到两段。

    ``transform`` 在拼好后按整段处理，用于修复被切分到相邻两条字幕里的词；
    ``protect(前段尾, 后段首)`` 返回 True 时禁止在此断段（避免把待校对词切两半）；
    ``pad`` 在字幕停顿处补读点（ASR 长段无标点时可读性明显改善）。
    """
    paragraphs: list[str] = []
    buffer: list[str] = []
    size = 0
    for index, item in enumerate(utterances):
        buffer.append(item.text)
        size += len(item.text)
        settled = item.text.rstrip()[-1:] in "。！？…"
        if not (settled and size >= target or size >= target * 2):
            continue
        following = utterances[index + 1].text if index + 1 < len(utterances) else ""
        if following and (_splits_token(item.text, following) or (protect and protect(item.text, following))):
            continue  # 数字/英文/待校对词被切在两条字幕之间，不在此断段
        paragraphs.append("".join(buffer))
        buffer, size = [], 0
    if buffer:
        if paragraphs and size < target // 3:
            paragraphs[-1] += "".join(buffer)
        else:
            paragraphs.append("".join(buffer))
    if transform:
        paragraphs = [transform(text) for text in paragraphs]
    if pad:
        paragraphs = [pad_punctuation(text) for text in paragraphs]
    return paragraphs


def sentence_at(utterances: list[Utterance], index: int, limit: int = 110, transform=None) -> str:
    """从给定字幕起向后取到句末，用于摘取可读的要点句。"""
    parts: list[str] = []
    size = 0
    for item in utterances[index:]:
        parts.append(item.text)
        size += len(item.text)
        if item.text.rstrip()[-1:] in "。！？…" or size >= limit:
            break
    text = "".join(parts)
    if transform:
        text = transform(text)
    return _trim(text, limit)


def build_local_article(
    transcript: Transcript,
    title: str | None = None,
    proofread_hits=None,
    proofread_rules: dict[str, str] | None = None,
    pad: bool = False,
) -> ArticleResult:
    utterances = transcript.utterances
    if not utterances:
        raise ValueError("字幕为空，无法生成文章")

    from .proofread import correct_text, format_report, key_straddles

    hits = proofread_hits if proofread_hits is not None else Counter()

    def fix(text: str) -> str:
        """段级再校对：修复被切分到相邻两条字幕里的词，命中并入同一份记录。"""
        if proofread_rules is None:
            return text
        return correct_text(text, proofread_rules, hits)

    def protect(prev_text: str, next_text: str) -> bool:
        if proofread_rules is None:
            return False
        return key_straddles(prev_text, next_text, proofread_rules)

    title = title or derive_title(transcript)
    token_lists = [tokenize(u.text) for u in utterances]
    scores = [float(value) for value in textrank_scores(token_lists)]

    meta = transcript.meta
    episodes = split_episodes(utterances)
    stats = meta.get("proofread") or {}
    fixed = int(stats.get("fixed_utterances", 0) or 0)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        f"> 本文由字幕工具自动整理：来源 `{meta.get('source', transcript.source)}`，"
        f"音频时长 {format_duration(meta.get('audio_seconds', 0))}，"
        f"共 {len(utterances)} 条字幕、{transcript.char_count()} 字，"
        f"按开场白识别为 {len(episodes)} 期内容。"
    )
    lines.append(">")
    lines.append(
        f"> 转写模型 `{meta.get('model_size', '未知')}`，语言 `{meta.get('language', '未知')}`，"
        f"耗时 {format_duration(meta.get('asr_seconds', 0))}。"
        + (f"术语校对修正 {fixed} 条字幕、{sum(hits.values())} 处替换。" if fixed else "")
        + "成文采用本地抽取式模式，保留讲者原始表述，仅做口语清洗、断句、分期与分节，不改写观点。"
        + ("文中读点按字幕停顿补齐（`--pad-punct`），未改动任何用字。" if pad else "")
    )
    lines.append("")

    # 内容提要：全文高分句
    highlights = pick_highlights(utterances, scores, count=6)
    if highlights:
        lines.append("## 内容提要")
        lines.append("")
        for idx in highlights:
            lines.append(f"- {sentence_at(utterances, idx, 100, transform=fix)}")
        lines.append("")

    episode_titles: list[str] = []
    for number, episode in enumerate(episodes, start=1):
        keywords = section_keywords(episode, topk=3)
        label = f"第{cn_index(number)}期"
        episode_titles.append(f"{label}｜{'·'.join(keywords)}" if keywords else label)
    if len(episodes) > 1:
        lines.append("## 目录")
        lines.append("")
        for number, episode in enumerate(episodes, start=1):
            lines.append(f"{number}. {episode_titles[number - 1]}（{format_clock(episode[0].start)} 起）")
        lines.append("")

    cursor = 0
    for number, episode in enumerate(episodes, start=1):
        lines.append(f"## {episode_titles[number - 1]}")
        lines.append("")
        lines.append(
            f"> {format_clock(episode[0].start)} – {format_clock(episode[-1].end)}｜"
            f"{len(episode)} 条字幕｜{sum(len(u.text) for u in episode)} 字"
        )
        lines.append("")

        skipped, body = strip_boilerplate(episode)
        sections = split_sections(body, target_chars=1800, min_chars=600)
        multi = len(sections) > 1
        episode_words = set(section_keywords(episode, topk=4))
        inner = cursor + skipped
        for sec_number, section in enumerate(sections, start=1):
            if multi:
                ranked = section_keywords(section, topk=4)
                keywords = [word for word in ranked if word not in episode_words][:2] or ranked[:2]
                heading = "·".join(keywords) if keywords else "内容"
                lines.append(f"### {sec_number}、{heading}")
                lines.append("")
            for paragraph in article_paragraphs(section, transform=fix, protect=protect, pad=pad):
                lines.append(paragraph)
                lines.append("")
            section_scores = scores[inner : inner + len(section)]
            inner += len(section)
            for idx in pick_highlights(section, section_scores, count=2, min_len=12):
                lines.append(f"> 要点：{sentence_at(section, idx, 110, transform=fix)}")
            lines.append("")
        cursor += len(episode)

    if hits:
        lines.append("## 校对记录")
        lines.append("")
        lines.append("下表列出本次自动替换的 ASR 误写（命中次数），可据此复核：")
        lines.append("")
        lines.append(format_report(hits, proofread_rules))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "同目录文件：`transcript.json` 为未校对的原始转写（可反复重跑校对与成文），"
        "`transcript.srt` / `transcript.txt` 为校对后的字幕。"
        "如需更流畅的成文效果，可配置 LLM 后使用 `--writer llm` 重新生成。"
    )
    lines.append("")
    return ArticleResult(markdown="\n".join(lines), mode="local", note="本地抽取式整理，未调用任何外部模型")


def strip_boilerplate(episode: list[Utterance]) -> tuple[int, list[Utterance]]:
    """去掉每期的开场问候与结尾告别，返回（跳过条数, 正文）。"""
    start, end = 0, len(episode)
    while start < end and (EPISODE_START_RE.match(episode[start].text) or len(episode[start].text) < 6):
        start += 1
    while end > start and (EPISODE_END_RE.search(episode[end - 1].text) or len(episode[end - 1].text) < 6):
        end -= 1
    if start >= end:
        return 0, list(episode)
    return start, episode[start:end]


# --------------------------------------------------------------------------- #
# LLM 成文
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "你是一名中文编辑，负责把视频讲解的语音转写字幕整理成可直接发布的文章。"
    "要求：忠实于讲者原意，不得添加字幕中没有的事实、数据或观点；"
    "去除口头禅、重复和语音识别错误造成的破碎表达；使用简体中文与规范标点；"
    "输出 Markdown，结构清晰（标题、导语、分节小标题、正文、要点小结）。"
)

MAP_PROMPT = (
    "下面是讲座字幕的第 {index}/{total} 部分。请把它整理成详细的中文笔记，"
    "保留关键概念、人物、时间、数据与讲者的论证顺序，不要遗漏重要信息，也不要杜撰。"
    "输出为带小标题的要点列表，字数控制在 800 字以内：\n\n{chunk}"
)

REDUCE_PROMPT = (
    "下面是对同一场讲座分段整理得到的笔记。请把它们合并成一篇完整、连贯、可直接发布的中文文章。\n"
    "要求：\n"
    "1. Markdown 格式，包含一级标题、导语、若干二级小标题与正文、结尾要点小结；\n"
    "2. 忠于笔记内容，不要引入笔记之外的信息；\n"
    "3. 合并重复内容，理顺逻辑顺序，语言书面化但不失讲者的表达风格；\n"
    "4. 篇幅充分展开，不少于 1500 字。\n\n"
    "讲座标题参考：{title}\n\n笔记：\n{notes}"
)

SINGLE_PROMPT = (
    "请把下面的讲座字幕整理成一篇完整、连贯、可直接发布的中文文章。\n"
    "要求：Markdown 格式，包含一级标题、导语、若干二级小标题与正文、结尾要点小结；"
    "忠于原文，不要添加字幕中没有的信息；语言书面化，去除口语冗余；篇幅不少于 1200 字。\n"
    "讲座标题参考：{title}\n\n字幕：\n{chunk}"
)


def _chat(cfg: LLMConfig, messages: list[dict]) -> str:
    url = f"{cfg.base_url}/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = ""
    for attempt in range(cfg.max_retries + 1):
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=cfg.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
            if not content.strip():
                raise LLMError(f"模型返回空内容：{json.dumps(body, ensure_ascii=False)[:300]}")
            return content.strip()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")[:300]
            last_error = f"HTTP {exc.code}: {detail}"
        except LLMError as exc:
            last_error = str(exc)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < cfg.max_retries:
            time.sleep(2 * (attempt + 1))
    raise LLMError(last_error or "调用失败")


def _chunks(transcript: Transcript, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in group_paragraphs(transcript.utterances, max_chars=max_chars // 3):
        text = "".join(u.text for u in paragraph)
        if current and size + len(text) > max_chars:
            pieces.append("\n".join(current))
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        pieces.append("\n".join(current))
    return pieces


def build_llm_article(
    transcript: Transcript,
    cfg: LLMConfig | None = None,
    title: str | None = None,
    chunk_chars: int = 12000,
    progress=None,
) -> ArticleResult:
    cfg = cfg or LLMConfig.from_env()
    if not cfg.available:
        raise LLMError(f"LLM 配置不完整，缺少：{', '.join(cfg.missing_fields())}")
    if not transcript.utterances:
        raise ValueError("字幕为空，无法生成文章")

    title = title or derive_title(transcript)
    chunks = _chunks(transcript, chunk_chars)
    notify = progress or (lambda msg: None)

    if len(chunks) == 1:
        notify(f"调用 LLM（{cfg.model}）生成文章……")
        markdown = _chat(
            cfg,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": SINGLE_PROMPT.format(title=title, chunk=chunks[0])},
            ],
        )
        return ArticleResult(markdown=markdown, mode="llm", note=f"单次调用，模型 {cfg.model}")

    notes: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        notify(f"LLM 分段整理 {index}/{len(chunks)}（{len(chunk)} 字）……")
        notes.append(
            _chat(
                cfg,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": MAP_PROMPT.format(index=index, total=len(chunks), chunk=chunk)},
                ],
            )
        )
    notify(f"LLM 汇总成文（{len(notes)} 段笔记）……")
    markdown = _chat(
        cfg,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": REDUCE_PROMPT.format(title=title, notes="\n\n".join(notes))},
        ],
    )
    return ArticleResult(
        markdown=markdown,
        mode="llm",
        note=f"map-reduce：{len(chunks)} 段分节笔记后汇总，模型 {cfg.model}",
    )


def build_article(
    transcript: Transcript,
    writer: str = "auto",
    cfg: LLMConfig | None = None,
    title: str | None = None,
    chunk_chars: int = 12000,
    progress=None,
    proofread_hits=None,
    proofread_rules: dict[str, str] | None = None,
    pad: bool = False,
) -> ArticleResult:
    """writer: auto | llm | local。auto 在缺少 LLM 配置或调用失败时回退本地模式。"""
    notify = progress or (lambda msg: None)
    cfg = cfg or LLMConfig.from_env()

    def local() -> ArticleResult:
        return build_local_article(
            transcript,
            title=title,
            proofread_hits=proofread_hits,
            proofread_rules=proofread_rules,
            pad=pad,
        )

    if writer == "local":
        return local()
    if writer == "llm":
        return build_llm_article(transcript, cfg=cfg, title=title, chunk_chars=chunk_chars, progress=progress)
    if writer != "auto":
        raise ValueError(f"未知的成文模式：{writer}")

    if not cfg.available:
        notify(f"未检测到 LLM 配置（缺少 {', '.join(cfg.missing_fields())}），使用本地抽取式成文。")
        return local()
    try:
        return build_llm_article(transcript, cfg=cfg, title=title, chunk_chars=chunk_chars, progress=progress)
    except LLMError as exc:
        notify(f"LLM 生成失败（{exc}），回退本地抽取式成文。")
        result = local()
        result.note = f"LLM 失败已回退本地模式：{exc}"
        return result
