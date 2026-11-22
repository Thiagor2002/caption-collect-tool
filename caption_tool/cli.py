"""命令行入口：视频选择 → 语音转写 → 字幕导出 → 成文。"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import time
from collections import Counter
from copy import deepcopy
from functools import partial
from pathlib import Path

from . import __version__
from .article import ArticleResult, LLMError, build_article, clean_stem
from .asr import AsrOptions, describe, transcribe
from .audio import format_duration, probe
from .config import DEFAULT_COMPUTE_TYPE, DEFAULT_LANGUAGE, DEFAULT_MODEL_SIZE, LLMConfig, OUTPUT_DIR, REPO_ROOT, VIDEOS_DIR, VIDEO_SUFFIXES
from .proofread import apply_rules, correct_text, key_straddles, load_rules
from .transcript import Transcript, load_transcript, save_transcript

TRANSCRIPT_STEM = "transcript"
ARTICLE_NAME = "article.md"
_INVALID_CHARS = re.compile(r'[\\/:*?"<>|\r\n]+')


def log(message: str = "") -> None:
    print(message, flush=True)


def sanitize(name: str) -> str:
    return _INVALID_CHARS.sub("_", name).strip() or "video"


def list_videos(videos_dir: Path) -> list[Path]:
    if not videos_dir.is_dir():
        return []
    return sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES and not p.name.startswith(".")
    )


def resolve_videos(tokens: list[str], videos_dir: Path) -> list[Path]:
    """按序号、完整文件名、去扩展名名称或通配符解析视频。"""
    videos = list_videos(videos_dir)
    if not videos:
        raise SystemExit(f"视频目录为空或不存在：{videos_dir}")
    if not tokens:
        raise SystemExit("未指定视频，可用 `list` 查看序号，或使用 --all 处理全部视频")

    selected: list[Path] = []
    for token in tokens:
        match: Path | None = None
        if token.isdigit():
            index = int(token)
            if 1 <= index <= len(videos):
                match = videos[index - 1]
        if match is None:
            for path in videos:
                if token in (path.name, path.stem, clean_stem(path.name)):
                    match = path
                    break
        if match is None:
            for path in videos:
                if fnmatch.fnmatch(path.name, token) or fnmatch.fnmatch(path.stem, token) or token in path.name:
                    match = path
                    break
        if match is None:
            raise SystemExit(f"找不到匹配 `{token}` 的视频，请先执行 `list` 查看可用视频")
        if match not in selected:
            selected.append(match)
    return selected


def make_progress(label: str):
    state = {"last": 0.0}

    def callback(fraction: float) -> None:
        now = time.time()
        if fraction < 1.0 and now - state["last"] < 0.5:
            return
        state["last"] = now
        fraction = min(1.0, max(0.0, fraction))
        filled = int(fraction * 36)
        bar = "█" * filled + "·" * (36 - filled)
        sys.stderr.write(f"\r  {label} [{bar}] {fraction * 100:5.1f}%")
        sys.stderr.flush()
        if fraction >= 1.0:
            sys.stderr.write("\n")
            sys.stderr.flush()

    return callback


def output_paths(output_dir: Path, name: str) -> dict[str, Path]:
    folder = output_dir / sanitize(name)
    return {
        "folder": folder,
        "srt": folder / f"{TRANSCRIPT_STEM}.srt",
        "txt": folder / f"{TRANSCRIPT_STEM}.txt",
        "json": folder / f"{TRANSCRIPT_STEM}.json",
        "article": folder / ARTICLE_NAME,
    }


def get_or_build_transcript(video: Path, paths: dict, opts: AsrOptions, force: bool) -> Transcript:
    """取得原始字幕：已存在 transcript.json 时直接复用，否则转写并落盘。"""
    if paths["json"].is_file() and not force:
        log(f"  已存在字幕，跳过转写（使用 --force 重新转写）：{paths['json'].name}")
        return load_transcript(paths["json"])

    log(f"  模型：{opts.model_size} / {opts.device} / {opts.compute_type}"
        + (f"，仅处理前 {format_duration(opts.max_seconds)}" if opts.max_seconds else ""))
    log("  首次运行会下载模型，请稍候……")
    result = transcribe(video, opts, progress=make_progress("转写"))
    transcript = result.transcript
    transcript.meta["title"] = clean_stem(video.name)
    log(f"  {describe(result)}")
    return transcript


def proofread_copy(transcript: Transcript, args: argparse.Namespace, folder: Path | None = None):
    """返回（校对后的字幕副本, 命中记录, 使用的规则）；``--no-proofread`` 时原样返回。

    词典优先级：``--corrections`` > 输出目录下的 ``corrections.txt`` > 仓库根目录同名文件。
    """
    if args.no_proofread:
        return transcript, Counter(), None
    path = args.corrections
    if not path:
        for candidate in (
            folder / "corrections.txt" if folder else None,
            REPO_ROOT / "corrections.txt",
        ):
            if candidate and Path(candidate).is_file():
                path = str(candidate)
                break
    rules = load_rules(path)
    view = deepcopy(transcript)
    hits = apply_rules(view, rules)
    fixed = view.meta["proofread"]["fixed_utterances"]
    log(f"  校对：{len(rules)} 条术语规则，命中 {sum(hits.values())} 处、修正 {fixed} 条字幕"
        + (f"（词典 {path}）" if path else "（内置词典）"))
    return view, hits, rules


def txt_transform(rules: dict[str, str] | None):
    """段级再校对的转换函数（不重复计数，计数交给成文那一次）。"""
    if rules is None:
        return None
    return partial(correct_text, rules=rules)


def txt_protect(rules: dict[str, str] | None):
    """分段保护：断点落在待校对词内部时不分段。"""
    if rules is None:
        return None
    return partial(key_straddles, rules=rules)


def write_article(result: ArticleResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.markdown.strip() + "\n", encoding="utf-8")


def process_video(video: Path, args: argparse.Namespace, opts: AsrOptions, llm: LLMConfig) -> int:
    name = args.title if args.title else clean_stem(video.name)
    paths = output_paths(Path(args.output_dir), name)
    paths["folder"].mkdir(parents=True, exist_ok=True)
    info = probe(video)

    log(f"\n=== {video.name} ===")
    log(f"  时长 {format_duration(info.duration)}，分辨率 {info.video_size or '未知'}，音频流 {'有' if info.has_audio else '无'}")
    log(f"  输出目录：{paths['folder']}")

    raw = get_or_build_transcript(video, paths, opts, args.force)
    save_transcript(raw, paths["folder"], TRANSCRIPT_STEM, skip=("srt", "txt"))
    log(f"  原始转写：{paths['json'].name}（{len(raw.utterances)} 条字幕）")

    transcript, hits, rules = proofread_copy(raw, args, paths["folder"])
    save_transcript(transcript, paths["folder"], TRANSCRIPT_STEM, skip=("json",),
                    transform=txt_transform(rules), protect=txt_protect(rules), pad=not args.no_pad_punct)
    log(f"  字幕：{paths['srt'].name} / {paths['txt'].name}")

    article = build_article(
        transcript,
        writer=args.writer,
        cfg=llm,
        title=name,
        chunk_chars=args.chunk_chars,
        progress=lambda msg: log(f"  {msg}"),
        proofread_hits=hits,
        proofread_rules=rules,
        pad=not args.no_pad_punct,
    )
    write_article(article, paths["article"])
    log(f"  文章：{paths['article'].name}（{article.mode} 模式，{len(article.markdown)} 字）")
    if article.note:
        log(f"  说明：{article.note}")
    return 0


def build_asr_options(args: argparse.Namespace) -> AsrOptions:
    return AsrOptions(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=None if args.language.lower() in ("auto", "") else args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        initial_prompt=args.initial_prompt,
        cpu_threads=args.cpu_threads,
        max_seconds=args.limit_seconds,
        simplify=not args.keep_traditional,
    )


def cmd_list(args: argparse.Namespace) -> int:
    videos_dir = Path(args.videos_dir)
    videos = list_videos(videos_dir)
    log(f"视频目录：{videos_dir}（共 {len(videos)} 个）")
    for index, video in enumerate(videos, start=1):
        size_mb = video.stat().st_size / 1024 / 1024
        try:
            duration = format_duration(probe(video).duration)
        except Exception as exc:  # noqa: BLE001
            duration = f"探测失败({type(exc).__name__})"
        paths = output_paths(Path(args.output_dir), clean_stem(video.name))
        done = []
        if paths["json"].is_file():
            done.append("字幕")
        if paths["article"].is_file():
            done.append("文章")
        log(f"  [{index}] {video.name}  {size_mb:.0f}MB  {duration}  已生成: {'/'.join(done) or '无'}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    videos = list_videos(Path(args.videos_dir)) if args.all else resolve_videos(args.videos, Path(args.videos_dir))
    opts = build_asr_options(args)
    llm = LLMConfig.from_env()
    if args.writer == "llm" and not llm.available:
        raise SystemExit(f"LLM 配置不完整，缺少：{', '.join(llm.missing_fields())}（可复制 .env.example 为 .env 后填写）")
    log(f"待处理 {len(videos)} 个视频；成文模式：{args.writer}")
    failures = 0
    for video in videos:
        try:
            process_video(video, args, opts, llm)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            log(f"  [错误] {video.name} 处理失败：{type(exc).__name__}: {exc}")
    log(f"\n完成：成功 {len(videos) - failures} 个，失败 {failures} 个，输出目录 {args.output_dir}")
    return 1 if failures else 0


def cmd_article(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"输出目录不存在：{output_dir}")

    targets = args.videos or sorted(
        path.name for path in output_dir.iterdir() if (path / f"{TRANSCRIPT_STEM}.json").is_file()
    )
    if not targets:
        raise SystemExit("outputs 下没有可重新成文的字幕，请先执行 run")

    llm = LLMConfig.from_env()
    if args.writer == "llm" and not llm.available:
        raise SystemExit(f"LLM 配置不完整，缺少：{', '.join(llm.missing_fields())}")

    resolved: list[tuple[str, Path]] = []
    for token in targets:
        json_path = None
        for candidate in (output_dir / sanitize(clean_stem(token)), output_dir / sanitize(token)):
            file = candidate / f"{TRANSCRIPT_STEM}.json"
            if file.is_file():
                json_path = file
                break
        if json_path is None:
            log(f"  [跳过] 找不到 {token} 的 transcript.json")
            continue
        resolved.append((json_path.parent.name, json_path))

    for name, json_path in resolved:
        raw = load_transcript(json_path)
        folder = json_path.parent
        log(f"\n=== {name} ===（{len(raw.utterances)} 条字幕）")
        transcript, hits, rules = proofread_copy(raw, args, folder)
        save_transcript(transcript, folder, TRANSCRIPT_STEM, skip=("json",),
                        transform=txt_transform(rules), protect=txt_protect(rules), pad=not args.no_pad_punct)
        article = build_article(
            transcript,
            writer=args.writer,
            cfg=llm,
            title=args.title or name,
            chunk_chars=args.chunk_chars,
            progress=lambda msg: log(f"  {msg}"),
            proofread_hits=hits,
            proofread_rules=rules,
            pad=not args.no_pad_punct,
        )
        target = folder / ARTICLE_NAME
        write_article(article, target)
        log(f"  文章：{target.name}（{article.mode} 模式，{len(article.markdown)} 字）")
    return 0


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--videos-dir", default=str(VIDEOS_DIR), help="视频目录（默认 test_videos）")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="输出目录（默认 outputs）")
    parser.add_argument("--writer", choices=("auto", "llm", "local"), default="auto", help="成文模式，auto 优先 LLM 并自动回退")
    parser.add_argument("--chunk-chars", type=int, default=12000, help="LLM 单次处理的最大字幕字数，超出则分段整理后汇总")
    parser.add_argument("--title", default=None, help="文章标题（仅处理单个视频时生效）")
    parser.add_argument("--corrections", default=None, help="外部纠错词典（每行 `误写=正确`），与内置规则合并")
    parser.add_argument("--no-proofread", action="store_true", help="关闭字幕术语校对，只保留原始转写文本")
    parser.add_argument("--no-pad-punct", action="store_true",
                        help="不在词边界补读点，完全保留 ASR 原始标点（.srt 的逐条文字始终不受影响）")


def add_asr(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL_SIZE, help="Whisper 模型：tiny/base/small/medium/large-v3（默认 small）")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="语言代码，auto 表示自动检测（默认 zh）")
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda（默认 cpu）")
    parser.add_argument("--compute-type", default=DEFAULT_COMPUTE_TYPE, help="int8/float16/float32 等（CPU 推荐 int8）")
    parser.add_argument("--beam-size", type=int, default=5, help="解码 beam size（默认 5）")
    parser.add_argument("--cpu-threads", type=int, default=0, help="CPU 线程数，0 表示自动")
    parser.add_argument("--no-vad", action="store_true", help="关闭 VAD 静音过滤")
    parser.add_argument("--keep-traditional", action="store_true", help="保留繁体输出，不做繁转简")
    parser.add_argument("--initial-prompt", default=None, help="自定义提示词，用于引导术语与风格")
    parser.add_argument("--limit-seconds", type=float, default=None, help="只处理前 N 秒，用于快速试跑")
    parser.add_argument("--force", action="store_true", help="已有字幕时强制重新转写")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caption_tool",
        description="从视频提取讲解字幕并整理成文章（faster-whisper + LLM/本地抽取式成文）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m caption_tool list\n"
            "  python -m caption_tool run 1 --model small\n"
            "  python -m caption_tool run --all --limit-seconds 600\n"
            "  python -m caption_tool article 苏共兴亡史 --writer local\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"caption_tool {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="列出视频目录中的视频与已生成结果")
    p_list.add_argument("--videos-dir", default=str(VIDEOS_DIR))
    p_list.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="转写字幕并生成文章（完整流程）")
    p_run.add_argument("videos", nargs="*", help="视频序号、文件名或通配符，可多个")
    p_run.add_argument("--all", action="store_true", help="处理视频目录中的全部视频")
    add_common(p_run)
    add_asr(p_run)
    p_run.set_defaults(func=cmd_run)

    p_article = sub.add_parser("article", help="基于已有 transcript.json 重新生成文章")
    p_article.add_argument("videos", nargs="*", help="输出目录名或视频名，缺省表示全部已转写结果")
    add_common(p_article)
    p_article.set_defaults(func=cmd_article)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except Exception:  # noqa: BLE001
                pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LLMError as exc:
        log(f"[错误] {exc}")
        return 2
    except KeyboardInterrupt:
        log("\n已中断")
        return 130
