"""Podium dashboard: FastAPI backend.

  .venv/bin/uvicorn app.server:app --port 8780
"""
import json
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from podium import audio, pipeline  # noqa: E402

DS = ROOT / "data" / "dataset"
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Podium")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _excerpts():
    out = []
    for meta in sorted(DS.glob("*/*/baseline.json")):
        m = json.loads(meta.read_text())
        out.append({"id": f"{m['split']}/{m['excerpt']}", "speaker": m["speaker"], "speech": m["speech"],
                    "transcript": (meta.parent / "transcript.txt").read_text().strip(),
                    "duration": round(m["words"][-1]["end"], 1)})
    return out


def _ref_dir(ref_id):
    d = (DS / ref_id).resolve()
    if DS.resolve() not in d.parents or not (d / "baseline.wav").exists():
        raise HTTPException(404, "unknown reference")
    return d


@lru_cache(maxsize=16)
def _reference(ref_id):
    d = _ref_dir(ref_id)
    return pipeline.describe(audio.load(d / "baseline.wav"), (d / "transcript.txt").read_text())


@app.get("/api/references")
def references():
    return _excerpts()


@app.get("/api/samples")
def samples(ref: str):
    d = _ref_dir(ref)
    out = []
    for m in sorted(d.glob("*.json")):
        if m.stem == "baseline":
            continue
        meta = json.loads(m.read_text())
        out.append({"name": m.stem, "kind": meta["kind"], "voice": meta.get("voice"),
                    "flaws": [{k: f[k] for k in ("type", "severity", "start", "end")} for f in meta["flaws"]]})
    return out


@app.get("/audio/{path:path}")
def audio_file(path: str):
    p = (DS / path).resolve()
    if DS.resolve() not in p.parents or p.suffix != ".wav" or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")


@app.post("/api/analyze")
async def analyze(ref: str = Form(...), sample: str = Form(None), file: UploadFile = File(None)):
    d = _ref_dir(ref)
    transcript = (d / "transcript.txt").read_text()
    if sample:
        p = (d / f"{sample}.wav").resolve()
        if d not in p.parents or not p.exists():
            raise HTTPException(404, "unknown sample")
        y = audio.load(p)
        gold = json.loads((d / f"{sample}.json").read_text())["flaws"]
    elif file is not None:
        raw = await file.read()
        if len(raw) > 40 * 1024 * 1024:
            raise HTTPException(413, "file too large")
        with tempfile.TemporaryDirectory() as tmp:
            src, wav = Path(tmp) / "in", Path(tmp) / "in.wav"
            src.write_bytes(raw)
            # ffmpeg decodes anything the browser records (webm/opus, m4a, mp3) into 16 kHz mono
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar",
                                str(audio.SR), str(wav)], capture_output=True)
            if r.returncode:
                raise HTTPException(400, "could not decode that audio file")
            y = audio.load(wav)
        gold = None
    else:
        raise HTTPException(400, "send a sample name or an audio file")
    if len(y) < audio.SR * 3:
        raise HTTPException(400, "recording is too short")
    ref_d = _reference(ref)
    part = pipeline.describe(y, transcript)
    res = pipeline.analyze(ref_d, part)
    payload = pipeline.plot_payload(ref_d, part, res)
    payload["gold"] = gold
    payload["duration"] = round(len(y) / audio.SR, 2)
    return JSONResponse(payload)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
