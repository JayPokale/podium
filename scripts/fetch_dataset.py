"""Download the dataset audio from the GitHub release and unpack it as WAV into data/dataset/.

  python scripts/fetch_dataset.py
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import soundfile as sf

URL = "https://github.com/JayPokale/podium/releases/download/v1/podium-dataset-v1.zip"
OUT = Path(__file__).resolve().parents[1] / "data" / "dataset"


def main():
    print("downloading", URL)
    data = urllib.request.urlopen(URL).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.endswith("/") or name == "README.md":
                continue
            dest = OUT / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".flac"):
                y, sr = sf.read(io.BytesIO(z.read(name)))
                sf.write(dest.with_suffix(".wav"), y, sr, subtype="PCM_16")
            elif not dest.exists():
                dest.write_bytes(z.read(name))
    print("done:", sum(1 for _ in OUT.rglob("*.wav")), "recordings in", OUT)


if __name__ == "__main__":
    main()
