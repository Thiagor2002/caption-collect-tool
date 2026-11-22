"""校对层：修正 ASR 高频同音错字，并输出可审计的替换记录。

设计原则：

* 只收录**高置信度**规则——专有名词的误写、以及根本不是词语的同音串
  （例如"托罗斯基""恶嫖遍业"）。常见词（如"援助""岛内"）一律不做全局替换，
  以免把正确文本改错。
* 校对是"视图级"操作：``transcript.json`` 始终保存原始转写结果，
  srt / txt / article 由原始数据 + 本模块派生，因此校对规则可以随时增删重跑。
* 每次应用都会记录命中次数，文章末尾附「校对记录」，便于人工复核。

外部词典：``--corrections corrections.txt``，每行 ``错误=正确``，``#`` 开头为注释，
同键覆盖内置规则。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from .transcript import AFTER_PUNCT_SPACE_RE, BEFORE_PUNCT_SPACE_RE

# "19 53 年" -> "1953 年"，"10 %" -> "10%"
_SP = r"[ \t\u00a0\u2000-\u200a\u202f\u205f\u3000]"
DIGIT_JOIN_RE = re.compile(rf"(?<=\d){_SP}+(?=[\d%．.])")
# "4. 4" -> "4.4"
DECIMAL_JOIN_RE = re.compile(rf"(?<=\d\.){_SP}+(?=\d)")
# "第 11 次" -> "第11次"（中文与数字之间不留空格）
CJK_DIGIT_RE = re.compile(rf"(?<=[\u4e00-\u9fff]){_SP}+(?=\d)|(?<=\d){_SP}+(?=[\u4e00-\u9fff])")


def normalize_spacing(text: str) -> str:
    """去掉全角标点两侧的空格，并合并被空格拆开的数字。

    早期版本的字幕清洗在标点全角化之前就做了空格处理，`transcript.json` 里会残留
    ``"， "`` 这类空格；校对是视图层的最后一道工序，在这里统一补正。
    Whisper 也常把年份、小数写成 ``19 53 年`` / ``4. 4``，一并合并。
    """
    if not text:
        return text
    text = BEFORE_PUNCT_SPACE_RE.sub("", AFTER_PUNCT_SPACE_RE.sub("", text))
    text = DIGIT_JOIN_RE.sub("", text)
    text = DECIMAL_JOIN_RE.sub("", text)
    return CJK_DIGIT_RE.sub("", text)

# --------------------------------------------------------------------------- #
# 内置规则
# --------------------------------------------------------------------------- #
# 人名 / 书名 / 文献 / 机构
NAMES: dict[str, str] = {
    "托罗斯基": "托洛茨基",
    "托勒斯基": "托洛茨基",
    "托洛茨斯基": "托洛茨基",
    "脱落司机": "托洛茨基",
    "伯列斯聂夫": "勃列日涅夫",
    "伯列斯孽夫": "勃列日涅夫",
    "伯列日聂夫": "勃列日涅夫",
    "伯连尔斯聂夫": "勃列日涅夫",
    "薄烈日孽夫": "勃列日涅夫",
    "布列日涅夫": "勃列日涅夫",
    "赫鲁小夫": "赫鲁晓夫",
    "哈佛小夫": "赫鲁晓夫",
    "克鲁晓夫": "赫鲁晓夫",
    "斯达林": "斯大林",
    "斯拉林": "斯大林",
    "傅大林": "斯大林",
    "死大瘾": "斯大林",
    "烈宁": "列宁",
    "马克夫主义": "马克思主义",
    "马列斯主义": "马列主义",
    "王阳宁": "王阳明",
    "传习路": "传习录",
    "传席录": "传习录",
    "列宁拳击": "列宁全集",
    "列宁拳体": "列宁全集",
    "斯大林拳击": "斯大林全集",
    "烈宁遗族": "列宁遗嘱",
    "列宁遗族": "列宁遗嘱",
    "到岸文件": "档案文件",
}

# 术语 / 成语 / 固定搭配（均为原写法不成词的情形）
WORDS: dict[str, str] = {
    "微信使馆": "唯心史观",
    "唯心使观": "唯心史观",
    "无彩阶级": "无产阶级",
    "修讯主义": "修正主义",
    "羞愁主义": "修正主义",
    "多档之": "多党制",
    "恶嫖遍业": "饿殍遍野",
    "恶漂遍业": "饿殍遍野",
    "烟消云泛": "烟消云散",
    "十本瞩目": "舍本逐末",
    "愚龙混杂": "鱼龙混杂",
    "奥听图说": "道听途说",
    "盖官定位": "盖棺定论",
    "素质高格": "束之高阁",
    "死无对准": "死无对证",
    "无疑例外": "无一例外",
    "致死不余": "至死不渝",
    "大功无私": "大公无私",
    "古往近来": "古往今来",
    "悬而右悬": "玄而又玄",
    "中期一生": "终其一生",
    "逻辑思辩": "逻辑思辨",
    "树碑地转": "树碑立传",
    "世代传奏": "世代传颂",
    "推波出来": "推波助澜",
    "意见胎子复中": "胎死腹中",
    "奋动终身": "奋斗终身",
    "理小信念": "理想信念",
    "功能大众": "工农大众",
    "傻拜天": "傻白甜",
    "全模博弈": "权谋博弈",
    "新谋活动": "阴谋活动",
    "精神使料": "精神食粮",
    "精神实量": "精神食粮",
    "彭博发展": "蓬勃发展",
    "资产积极": "资产阶级",
    "英明还是婚宴": "英明还是昏庸",
    "一切社会关系的统合": "一切社会关系的总和",
    "问题带于改变世界": "问题在于改变世界",
    "不幸中的外星": "不幸中的万幸",
    "政治是一文艺术": "政治是一门艺术",
    "第三条路可入": "第三条路可走",
    "针对斯大林的自杀": "针对斯大林的暗杀",
    "肆于刺杀": "死于刺杀",
    "联合缴杀": "联合绞杀",
    "等要评价": "如何评价",
    "为人否知": "不为人知",
    "思想重视路线": "思想政治路线",
    "子战员": "指战员",
    "击备人": "接班人",
    "帮兄": "帮凶",
    "诸朵": "诸多",
    "拥顿": "拥趸",
    "苏共岛内": "苏共党内",
    "运喊": "蕴含",
    "晕含": "蕴含",
    "规解起来": "归结起来",
    "自热起来": "炽热起来",
    "经度者同意": "经作者同意",
}

TERMS: dict[str, str] = {**NAMES, **WORDS}


def key_straddles(prev_text: str, next_text: str, rules) -> bool:
    """判断断点是否落在某条规则键内部（会把待校对词切到两段）。"""
    if not prev_text or not next_text:
        return False
    tail, head = prev_text[-10:], next_text[:10]
    for key in rules:
        for i in range(1, len(key)):
            if tail.endswith(key[:i]) and head.startswith(key[i:]):
                return True
    return False


def load_rules(path: str | Path | None = None) -> dict[str, str]:
    """内置规则 + 外部词典（外部同键覆盖内置）。"""
    rules = dict(TERMS)
    if not path:
        return rules
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"纠错词典不存在：{file}")
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        wrong, right = (part.strip() for part in line.split("=", 1))
        if wrong and right:
            rules[wrong] = right
    return rules


def compile_rules(rules: dict[str, str]) -> re.Pattern[str]:
    """长规则优先，避免短词先命中把长词切碎。"""
    keys = sorted(rules, key=lambda item: (-len(item), item))
    return re.compile("|".join(re.escape(k) for k in keys))


def correct_text(text: str, rules: dict[str, str] | None = None, hits: Counter | None = None) -> str:
    """按词典替换文本；传入 ``hits`` 会累计每条规则的命中次数。"""
    if not text:
        return text
    mapping = rules if rules is not None else TERMS
    text = normalize_spacing(text)
    return _sub(compile_rules(mapping), mapping, text, hits)


def _sub(pattern: re.Pattern[str], mapping: dict[str, str], text: str, hits: Counter | None) -> str:
    if hits is None:
        return pattern.sub(lambda m: mapping[m.group(0)], text)

    def replace(match: re.Match[str]) -> str:
        wrong = match.group(0)
        hits[wrong] += 1
        return mapping[wrong]

    return pattern.sub(replace, text)


def apply_rules(transcript, rules: dict[str, str] | None = None) -> Counter:
    """就地校对整份字幕，返回各规则的命中次数。"""
    rules = rules if rules is not None else TERMS
    pattern = compile_rules(rules)
    hits: Counter[str] = Counter()
    changed = 0
    for utterance in transcript.utterances:
        original = normalize_spacing(utterance.text)
        fixed = _sub(pattern, rules, original, hits)
        if fixed != utterance.text:
            utterance.text = fixed
            changed += 1
    transcript.meta["proofread"] = {
        "rules": len(rules),
        "fixed_utterances": changed,
        "replacements": int(sum(hits.values())),
    }
    return hits


def format_report(hits: Counter, rules: dict[str, str] | None = None, limit: int = 100) -> str:
    """把命中记录渲染成 Markdown 表格（供文章附录使用）。"""
    rules = rules if rules is not None else TERMS
    used = [(wrong, rules.get(wrong, "?"), int(count)) for wrong, count in hits.most_common(limit) if count]
    if not used:
        return "> 本次校对未命中任何规则。"
    lines = ["| 误写 | 校正 | 命中 |", "| --- | --- | --- |"]
    lines += [f"| {wrong} | {right} | {count} |" for wrong, right, count in used]
    total = int(sum(hits.values()))
    lines.append("")
    lines.append(
        f"共命中 {len(hits)} 条规则、{total} 处替换"
        + (f"（下表列出最多的 {limit} 条）。" if len(hits) > limit else "。")
        + "规则只收录专有名词与不成词的同音串，普通词一律用短语限定，避免误改。"
    )
    return "\n".join(lines)
