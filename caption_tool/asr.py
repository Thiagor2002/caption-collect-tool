"""语音转写：faster-whisper 封装。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .audio import MediaInfo, format_duration, load_audio, probe
from .transcript import Transcript, build_utterances

ZH_PROMPT = (
    "以下是一段简体中文的课堂讲座记录，请使用简体中文和规范的标点符号（逗号、句号、问号）输出。"
    "例如：大家好，今天我们讲的主题是苏联的历史。首先，我们来看斯大林时期的政策，"
    "这些政策对后来的发展产生了深远的影响。"
)

_MODEL_CACHE: dict[tuple, object] = {}


@dataclass
class AsrOptions:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = "zh"
    beam_size: int = 5
    vad_filter: bool = True
    initial_prompt: str | None = None
    cpu_threads: int = 0
    max_seconds: float | None = None
    download_root: str | None = None
    simplify: bool = True
    word_timestamps: bool = True


@dataclass
class AsrResult:
    info: MediaInfo
    transcript: Transcript
    elapsed: float
    model_size: str

    @property
    def speed(self) -> float:
        """相对实时的速度倍数（>1 表示快于实时）。"""
        audio_seconds = self.transcript.meta.get("audio_seconds", 0.0)
        return audio_seconds / self.elapsed if self.elapsed > 0 else 0.0


def get_model(opts: AsrOptions):
    from faster_whisper import WhisperModel

    key = (opts.model_size, opts.device, opts.compute_type, opts.download_root)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = WhisperModel(
            opts.model_size,
            device=opts.device,
            compute_type=opts.compute_type,
            cpu_threads=opts.cpu_threads,
            download_root=opts.download_root,
        )
        _MODEL_CACHE[key] = model
    return model


def transcribe(
    video_path: str | Path,
    opts: AsrOptions | None = None,
    progress: Callable[[float], None] | None = None,
) -> AsrResult:
    video_path = Path(video_path)
    opts = opts or AsrOptions()

    info = probe(video_path)
    if not info.has_audio:
        raise RuntimeError(f"{video_path.name} 没有音频流，无法转写")

    limit = opts.max_seconds
    audio_seconds = min(info.duration, limit) if (limit and info.duration) else (limit or info.duration)

    t0 = time.time()
    audio = load_audio(video_path, max_seconds=limit)
    decode_seconds = time.time() - t0
    audio_seconds = len(audio) / 16000.0

    model = get_model(opts)
    prompt = opts.initial_prompt
    if prompt is None and (opts.language or "zh").lower().startswith("zh"):
        prompt = ZH_PROMPT

    t1 = time.time()
    segments, seg_info = model.transcribe(
        audio,
        language=opts.language,
        beam_size=opts.beam_size,
        vad_filter=opts.vad_filter,
        vad_parameters={"min_silence_duration_ms": 400} if opts.vad_filter else None,
        initial_prompt=prompt,
        word_timestamps=opts.word_timestamps,
        condition_on_previous_text=False,
    )
    raw_segments = []
    for segment in segments:  # 生成器：边解码边推进度
        raw_segments.append(segment)
        if progress is not None and audio_seconds > 0:
            progress(min(1.0, float(segment.end) / audio_seconds))
    if progress is not None:
        progress(1.0)
    asr_seconds = time.time() - t1

    utterances = build_utterances(raw_segments, simplify_text=opts.simplify)
    detected_language = getattr(seg_info, "language", opts.language) or opts.language
    transcript = Transcript(
        source=video_path.name,
        utterances=utterances,
        meta={
            "source_path": str(video_path),
            "media_duration": round(info.duration, 2),
            "audio_seconds": round(audio_seconds, 2),
            "truncated": bool(limit and info.duration > limit + 1),
            "model_size": opts.model_size,
            "compute_type": opts.compute_type,
            "device": opts.device,
            "language": detected_language,
            "language_probability": round(float(getattr(seg_info, "language_probability", 0.0) or 0.0), 4),
            "vad_filter": opts.vad_filter,
            "beam_size": opts.beam_size,
            "word_timestamps": opts.word_timestamps,
            "simplified": opts.simplify,
            "initial_prompt": prompt or "",
            "segment_count": len(raw_segments),
            "utterance_count": len(utterances),
            "decode_seconds": round(decode_seconds, 1),
            "asr_seconds": round(asr_seconds, 1),
            "video_size": info.video_size,
        },
    )
    return AsrResult(info=info, transcript=transcript, elapsed=time.time() - t0, model_size=opts.model_size)


def describe(result: AsrResult) -> str:
    meta = result.transcript.meta
    return (
        f"音频 {format_duration(meta['audio_seconds'])}"
        f"（原片 {format_duration(meta['media_duration'])}）"
        f"，字幕 {meta['utterance_count']} 条 / {result.transcript.char_count()} 字"
        f"，耗时 {format_duration(result.elapsed)}"
        f"，速度 {result.speed:.1f}x 实时"
        f"，语言 {meta['language']}({meta['language_probability']:.0%})"
    )
