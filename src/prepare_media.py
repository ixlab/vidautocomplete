"""Download helper utilities for MSVD media preparation."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

FFMPEG_ERROR = (
    "ffmpeg not found on PATH. Install it first (e.g., `sudo apt install ffmpeg`)")


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(FFMPEG_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MSVD source videos to MP4 and extract first-frame thumbnails"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory that contains the raw MSVD clips (e.g., extracted YouTubeClips/)",
    )
    parser.add_argument(
        "--pattern",
        default="*.avi",
        help="Glob pattern for source files relative to source-dir (default: *.avi)",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("static/msvd-vids"),
        help="Output directory for MP4 files",
    )
    parser.add_argument(
        "--thumbnails-dir",
        type=Path,
        default=Path("static/msvd-imgs"),
        help="Output directory for JPEG thumbnails",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate files even if they already exist",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame index to capture for the thumbnail (default: 0)",
    )
    parser.add_argument(
        "--video-codec",
        default="h264",
        help="FFmpeg video codec to use (default: h264; set to libx264 if your ffmpeg supports it)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers to use for conversion/thumbnail extraction",
    )
    return parser.parse_args()


def iter_sources(source_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted(source_dir.rglob(pattern))


def convert_to_mp4(src: Path, dst: Path, overwrite: bool, codec: str) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
    ]
    if codec == "libx264":
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    elif codec == "h264":
        cmd += [
            "-c:v",
            "h264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-strict",
            "-2",
        ]
    else:
        cmd += ["-c:v", codec]

    cmd += [
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def extract_thumbnail(video_path: Path, thumb_path: Path, frame_index: int, overwrite: bool) -> None:
    if thumb_path.exists() and not overwrite:
        return
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    filter_expr = f"select=eq(n\\,{frame_index})"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        filter_expr,
        "-vframes",
        "1",
        "-q:v",
        "2",
        str(thumb_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    ensure_ffmpeg()
    sources = list(iter_sources(args.source_dir, args.pattern))
    if not sources:
        raise FileNotFoundError(f"No files matching {args.pattern} under {args.source_dir}")
    print(f"Found {len(sources)} source videos. Converting and extracting thumbnails...")

    def process(src_path: Path) -> str:
        base_name = src_path.stem
        mp4_path = args.videos_dir / f"{base_name}.mp4"
        thumb_path = args.thumbnails_dir / f"{base_name}.jpg"
        convert_to_mp4(src_path, mp4_path, args.overwrite, args.video_codec)
        extract_thumbnail(mp4_path, thumb_path, args.start_frame, args.overwrite)
        return base_name

    workers = max(1, args.workers)
    if workers == 1:
        for src in sources:
            process(src)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process, src): src for src in sources}
            for future in as_completed(futures):
                src = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Failed to process {src}: {exc}")
                    raise

    print("Done. Videos ready under", args.videos_dir, "and thumbnails under", args.thumbnails_dir)


if __name__ == "__main__":
    main()
