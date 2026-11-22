"""音频解码与媒体信息探测（基于 PyAV，无需系统安装 ffmpeg）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


@dataclass
class MediaInfo:
    path: Path
    duration: float
    has_audio: bool
    video_size: str = ""


def probe(path: str | Path) -> MediaInfo:
    import av

    path = Path(path)
    with av.open(str(path), metadata_errors="ignore") as container:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        video_stream = next((s for s in container.streams if s.type == "video"), None)
        duration = float(container.duration) / av.time_base if container.duration else 0.0
        if duration <= 0 and audio_stream is not None and audio_stream.duration is not None:
            duration = float(audio_stream.duration * audio_stream.time_base)
        size = ""
        if video_stream is not None:
            size = f"{video_stream.codec_context.width}x{video_stream.codec_context.height}"
        return MediaInfo(path=path, duration=duration, has_audio=audio_stream is not None, video_size=size)


def load_audio(path: str | Path, sampling_rate: int = SAMPLE_RATE, max_seconds: float | None = None) -> np.ndarray:
    """把媒体文件解码成单声道 float32 波形，取值范围 [-1, 1]。

    max_seconds 用于只解码前 N 秒，方便快速试跑长视频。
    """
    import av

    path = Path(path)
    chunks: list[np.ndarray] = []
    total_samples = 0

    with av.open(str(path), metadata_errors="ignore") as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError(f"{path.name} 中没有音频流")
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sampling_rate)

        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                data = out.to_ndarray().reshape(-1)
                chunks.append(data)
                total_samples += data.shape[0]
            if max_seconds is not None and total_samples >= max_seconds * sampling_rate:
                break

        try:
            for out in resampler.resample(None):  # flush 重采样器
                chunks.append(out.to_ndarray().reshape(-1))
        except Exception:  # noqa: BLE001 - 不同 PyAV 版本 flush 行为不一致
            pass

    if not chunks:
        raise RuntimeError(f"{path.name} 解码后没有得到有效音频")

    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    if max_seconds is not None:
        audio = audio[: int(max_seconds * sampling_rate)]
    return np.clip(audio, -1.0, 1.0)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}小时{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"
