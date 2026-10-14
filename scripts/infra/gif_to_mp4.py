"""Write an H.264 MP4 next to every GIF in docs/figures (the website plays the MP4 and links the GIF).

Rerun after regenerating an animation: python scripts/infra/gif_to_mp4.py [--force]
Uses ffmpeg from PATH, or the hub copy at ../ffmpeg.exe (Psychograph/ffmpeg.exe).
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "docs" / "figures"


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg") or str(ROOT.parent / "ffmpeg.exe")
    if not Path(exe).exists() and not shutil.which(exe):
        sys.exit("ffmpeg not found (PATH or ../ffmpeg.exe)")
    return exe


def main(force: bool = False):
    exe = ffmpeg()
    for gif in sorted(FIGS.rglob("*.gif")):
        mp4 = gif.with_suffix(".mp4")
        if mp4.exists() and not force and mp4.stat().st_mtime >= gif.stat().st_mtime:
            continue
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(gif), "-movflags", "+faststart",
                        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
                        "-crf", "22", "-preset", "slow", str(mp4)], check=True)
        print(f"{gif.relative_to(ROOT)}: {gif.stat().st_size // 1024} kB -> {mp4.stat().st_size // 1024} kB")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
