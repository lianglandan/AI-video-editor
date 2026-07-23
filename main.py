from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

FILLER_PATTERN = re.compile(
    r"^(?:嗯+|啊+|呃+|额+|唔+|那个|然后|就是|对吧)[，,。.!！?？\s]*$"
)


def normalize_text(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text.lower())


def load_whisper_segments(json_path: Path) -> list[dict[str, Any]]:
    """Load timestamped segments from an OpenAI Whisper JSON result."""
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"无法解析 Whisper JSON：{json_path}") from error

    raw_segments = payload.get("segments") if isinstance(payload, dict) else payload
    if not isinstance(raw_segments, list):
        raise ValueError("Whisper JSON 必须是列表，或包含 segments 列表。")

    segments: list[dict[str, Any]] = []
    for index, item in enumerate(raw_segments):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 个片段格式无效。")
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"第 {index} 个片段需要有效的 start 和 end。") from error

        text = str(item.get("text", "")).strip()
        if end > start and text:
            segments.append({"start": start, "end": end, "text": text})

    if not segments:
        raise ValueError("Whisper JSON 中没有可用的语音片段。")
    return sorted(segments, key=lambda segment: segment["start"])


def keep_speech_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep complete substantive sentences; remove standalone fillers and repeats."""
    kept: list[dict[str, Any]] = []
    previous_text = ""
    previous_end = float("-inf")

    for segment in segments:
        text = str(segment["text"])
        normalized = normalize_text(text)
        if not normalized or FILLER_PATTERN.match(text):
            continue

        duplicate = (
            len(previous_text) >= 4
            and float(segment["start"]) - previous_end < 15
            and SequenceMatcher(None, normalized, previous_text).ratio() >= 0.88
        )
        if duplicate:
            continue

        kept.append(segment)
        previous_text = normalized
        previous_end = float(segment["end"])
    return kept


def build_cut_list(
    segments: list[dict[str, Any]], pause_threshold: float = 1.5, padding: float = 0.08
) -> list[dict[str, float | bool]]:
    """Create retained video ranges, cutting only pauses longer than the threshold."""
    speech = keep_speech_segments(segments)
    if not speech:
        raise ValueError("没有可保留的有效口播片段。")

    ranges: list[list[float]] = []
    previous_speech_end = 0.0
    for segment in speech:
        start = max(0.0, float(segment["start"]) - padding)
        end = float(segment["end"]) + padding
        if ranges and float(segment["start"]) - previous_speech_end <= pause_threshold:
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])
        previous_speech_end = float(segment["end"])

    return [{"start": round(start, 3), "end": round(end, 3), "keep": True} for start, end in ranges]


def render_cut_list(video_path: Path, output_path: Path, cut_list: list[dict[str, float | bool]]) -> None:
    """Render retained ranges with FFmpeg and concatenate them into one MP4."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 FFmpeg，请确认 ffmpeg 已加入 PATH。")
    if not cut_list:
        raise ValueError("cut_list.json 中没有保留片段。")

    filters: list[str] = []
    labels: list[str] = []
    for index, item in enumerate(cut_list):
        start = float(item["start"])
        end = float(item["end"])
        filters.extend(
            [
                f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}]",
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{index}]",
            ]
        )
        labels.append(f"[v{index}][a{index}]")

    filters.append(f"{''.join(labels)}concat=n={len(cut_list)}:v=1:a=1[outv][outa]")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[outv]",
        "-map",
        "[outa]",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def rough_cut(
    video_path: Path,
    whisper_json: Path,
    cut_list_path: Path,
    output_path: Path,
    pause_threshold: float,
    padding: float,
) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"找不到视频文件：{video_path}")
    if not whisper_json.is_file():
        raise FileNotFoundError(f"找不到 Whisper JSON：{whisper_json}")
    if pause_threshold < 0 or padding < 0:
        raise ValueError("停顿阈值和片段留白不能小于 0。")

    cut_list = build_cut_list(load_whisper_segments(whisper_json), pause_threshold, padding)
    cut_list_path.write_text(json.dumps(cut_list, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_cut_list(video_path, output_path, cut_list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI 视频自动粗剪")
    parser.add_argument("video", type=Path, help="原始视频文件")
    parser.add_argument("whisper_json", type=Path, help="Whisper 生成的 JSON 时间轴")
    parser.add_argument("--cut-list", type=Path, default=Path("cut_list.json"))
    parser.add_argument("--output", type=Path, default=Path("final_video.mp4"))
    parser.add_argument("--pause-threshold", type=float, default=1.5, help="删除超过该秒数的停顿")
    parser.add_argument("--padding", type=float, default=0.08, help="语音片段前后保留秒数")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    rough_cut(
        arguments.video,
        arguments.whisper_json,
        arguments.cut_list,
        arguments.output,
        arguments.pause_threshold,
        arguments.padding,
    )
    print(f"已生成：{arguments.cut_list} 和 {arguments.output}")
