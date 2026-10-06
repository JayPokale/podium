"""Build the demo video: narrated slides, real recordings of the dashboard, real dataset audio.

Narration uses an open Piper TTS voice. The dashboard must be running on :8780.
  python scripts/make_video.py   -> build/podium_demo.mp4
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "build"
DS = ROOT / "data" / "dataset"
VOICE = ROOT / "data" / "voices" / "en_US-lessac-medium.onnx"
BROWSER = "/opt/brave-bin/brave"
W, H = 1280, 720
APP = "http://localhost:8780"

STYLE = """
<style>
 html,body{margin:0;width:1280px;height:720px;background:#f6f5f1;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:#1d2330}
 .s{box-sizing:border-box;width:1280px;height:720px;padding:64px 80px;display:flex;flex-direction:column;justify-content:center}
 h1{font-size:64px;margin:0 0 12px;letter-spacing:-.02em} h2{font-size:44px;margin:0 0 28px}
 p,li{font-size:28px;line-height:1.45;margin:6px 0} .m{color:#6a7180} .a{color:#2c4bd6}
 table{border-collapse:collapse;font-size:24px} td,th{border:1px solid #d5d2c8;padding:8px 16px;text-align:left} th{background:#ece9e0}
 .big{font-size:84px;font-weight:700;color:#2c4bd6} .row{display:flex;gap:36px} .card{background:#fff;border:1px solid #e3e1da;border-radius:16px;padding:20px 26px;flex:1}
 .flow{display:flex;gap:14px;align-items:center;font-size:22px;flex-wrap:wrap} .flow div{background:#fff;border:2px solid #2c4bd6;border-radius:12px;padding:12px 16px}
</style>"""


def run(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def slide(name, body):
    html = OUT / f"{name}.html"
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'>{STYLE}</head><body><div class='s'>{body}</div></body></html>")
    png = OUT / f"{name}.png"
    run([BROWSER, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
         f"--screenshot={png}", f"file://{html}"], capture_output=True)
    return png


def tts(name, text):
    wav = OUT / f"{name}.wav"
    run([sys.executable, "-m", "piper", "-m", str(VOICE), "--length-scale", "1.08", "-f", str(wav)],
        input=text.encode(), capture_output=True)
    return wav


def dur(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout
    return float(out)


def concat_audio(name, parts, gap=0.5):
    """parts: list of wav paths or ('silence', seconds); joined with short gaps, 22050 Hz mono."""
    sr = 22050
    chunks = []
    for p in parts:
        if isinstance(p, tuple):
            chunks.append(np.zeros(int(p[1] * sr)))
        else:
            y, s = sf.read(p)
            if y.ndim > 1:
                y = y.mean(1)
            if s != sr:
                import librosa
                y = librosa.resample(y, orig_sr=s, target_sr=sr)
            chunks.append(y)
        chunks.append(np.zeros(int(gap * sr)))
    wav = OUT / f"{name}_mix.wav"
    sf.write(wav, np.concatenate(chunks), sr)
    return wav


def still(name, png, wav, tail=0.8):
    mp4 = OUT / f"seg_{name}.mp4"
    t = dur(wav) + tail
    run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), "-i", str(wav), "-t", f"{t:.2f}",
         "-vf", f"scale={W}:{H},format=yuv420p", "-r", "30", "-c:v", "libx264", "-preset", "veryfast",
         "-c:a", "aac", "-ar", "44100", "-ac", "2", "-af", "apad", str(mp4)])
    return mp4


def clip(name, video, wav):
    """Screen recording + narration; the last frame is held if the narration is longer."""
    mp4 = OUT / f"seg_{name}.mp4"
    t = max(dur(video), dur(wav) + 0.8)
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-t", f"{t:.2f}",
         "-vf", f"scale={W}:{H},tpad=stop_mode=clone:stop_duration=60,format=yuv420p", "-r", "30",
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-ar", "44100", "-ac", "2", "-af", "apad",
         str(mp4)])
    return mp4


def record(name, steps):
    """Drive the real dashboard in a headless browser and record it."""
    from playwright.sync_api import sync_playwright
    vdir = OUT / f"rec_{name}"
    vdir.mkdir(exist_ok=True)
    for f in vdir.glob("*.webm"):
        f.unlink()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=BROWSER)
        ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(vdir),
                            record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        steps(page)
        ctx.close()
        b.close()
    return next(vdir.glob("*.webm"))


def smooth_scroll(page, to, ms=2500):
    page.evaluate("""([to, ms]) => new Promise(r => { const y0 = scrollY, t0 = performance.now();
      const f = t => { const k = Math.min(1, (t - t0) / ms); scrollTo(0, y0 + (to - y0) * (1 - Math.cos(Math.PI * k)) / 2);
      k < 1 ? requestAnimationFrame(f) : r(); }; requestAnimationFrame(f); })""", [to, ms])


def wait_result(page):
    page.wait_for_function("() => /found|Error/.test(document.getElementById('status').textContent)", timeout=120000)


def y_of(page, sel):
    return page.evaluate(f"() => document.querySelector('{sel}').getBoundingClientRect().top + scrollY - 20")


def snippet(rel_wav, a, b, name):
    y, sr = sf.read(DS / rel_wav)
    out = OUT / f"{name}.wav"
    sf.write(out, y[int(a * sr): int(b * sr)], sr)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    segs = []
    say = {}
    say["intro"] = ("This is Podium, a contrastive speech analytics system, built for Track C of the Multimodal A I "
                    "Hackathon. It listens to how you deliver a speech, compares it word by word with a great delivery "
                    "of the same text, and pinpoints exactly where, and why, your delivery drifts.")
    segs.append(still("intro", slide("s_intro", "<h1>🎙️ Podium</h1><p class='a' style='font-size:34px'>Contrastive speech "
                      "analytics &amp; temporal flaw grounding</p><p class='m'>Multimodal AI Hackathon 2026 · Track C</p>"
                      "<p class='m' style='margin-top:40px'>github.com/JayPokale/podium</p>"), tts("n_intro", say["intro"])))

    say["problem"] = ("Judging spoken delivery is subjective. Two judges hear the same rushed phrase, and disagree on where "
                      "it started, or how bad it was. And there is no dataset that pairs a good delivery with bad "
                      "deliveries of the exact same words. So Podium starts by building one.")
    segs.append(still("problem", slide("s_problem", "<h2>The problem</h2><ul><li>Delivery feedback is vague: "
                      "<i>“work on your pacing”</i></li><li>Judges disagree on <b>where</b> a flaw starts and <b>how bad</b> it is</li>"
                      "<li>No dataset pairs a good delivery with bad deliveries of the <b>same text</b></li></ul>"),
                      tts("n_problem", say["problem"])))

    # dataset, with real audio: an original line, then the same audio with a stutter injected
    meta = json.loads((DS / "test/reagan_challenger_1/stress_stutter_3.json").read_text())
    g = meta["flaws"][0]
    base = json.loads((DS / "test/reagan_challenger_1/baseline.json").read_text())["words"]
    i0, i1 = max(0, g["i"] - 5), min(len(base) - 1, g["i"] + 4)
    a_orig = snippet("test/reagan_challenger_1/baseline.wav", base[i0]["start"] - 0.15, base[i1]["end"] + 0.2, "a_orig")
    mw = meta["words"]
    a_flaw = snippet("test/reagan_challenger_1/stress_stutter_3.wav", mw[i0]["start"] - 0.15, mw[i1]["end"] + 0.2, "a_flaw")
    words = " ".join(w["word"] for w in base[i0:i1 + 1])
    d1 = ("We took three public domain speeches: John F. Kennedy's inaugural and moon speeches, and Ronald Reagan's "
          "Challenger address. Whisper transcribed clean, applause free excerpts, and a wav2vec2 forced aligner timed "
          "every word. Then we built the bad mirror: eight delivery flaws, injected at three severities into the same "
          "audio. Because we inject them, every label is exact to the millisecond. Here is Reagan's original line.")
    d2 = "And here is the same audio, with a stutter injected."
    d3 = ("In total, two hundred and seven recordings, from untouched to egregious, plus the same texts read by open "
          "source synthetic voices, to test that the comparison works across speakers.")
    mix = concat_audio("dataset", [tts("n_d1", d1), a_orig, tts("n_d2", d2), a_flaw, tts("n_d3", d3)])
    segs.append(still("dataset", slide("s_dataset", "<h2>A contrastive dataset, built from scratch</h2><div class='row'>"
                      "<div class='card'><p><b>6 ideal baselines</b></p><p class='m'>JFK 1961 &amp; 1962, Reagan 1986 · public domain"
                      "<br>Whisper → MMS forced alignment</p></div><div class='card'><p><b>The bad mirror</b></p><p class='m'>"
                      "8 flaws × 3 severities<br>rushed · dragged · monotone · mumbled · loud · awkward pause · missing breath · stutter</p></div>"
                      "<div class='card'><p><b>207 recordings</b></p><p class='m'>gradient L0→L4, stress tests, different-speaker TTS readings · "
                      "labels exact to the ms</p></div></div>"
                      f"<p style='margin-top:34px'>🔊 <i>“{words}”</i></p><p class='m'>original → same audio with a stutter injected</p>"),
                      mix))

    say["how"] = ("Both recordings are force aligned to the same transcript, so word i of your reading lines up with word i "
                  "of the reference. For every word, Praat and librosa measure speaker normalised features: pitch in "
                  "semitones relative to your own median, loudness relative to your own level, articulation rate, pauses, "
                  "clarity, and voiced sound inside gaps, which separates a stutter from a silent pause. A word is flagged "
                  "only with two kinds of evidence: it departs from the reference beyond your overall style, and it stands "
                  "out within your own delivery. That second check keeps a different voice from being flagged everywhere.")
    segs.append(still("how", slide("s_how", "<h2>How it works</h2><div class='flow'><div>Reference + you<br><span class='m'>same transcript</span></div>→"
                      "<div>Forced alignment<br><span class='m'>MMS_FA wav2vec2</span></div>→<div>Speaker-normalised features<br>"
                      "<span class='m'>F0 st · dB · articulation rate · pauses · clarity · voiced gaps</span></div>→"
                      "<div>z<sub>ref</sub> ∧ z<sub>self</sub><br><span class='m'>robust median / MAD</span></div>→"
                      "<div>Flaw regions<br><span class='m'>+ causal explanation + rubric</span></div></div>"
                      "<p style='margin-top:40px'>flag = min( deviation from the reference beyond your style , deviation within your own delivery )</p>"),
                      tts("n_how", say["how"])))

    say["app"] = ("Here is the dashboard. I pick Reagan's Challenger excerpt, and a sample with several injected flaws, and "
                  "press analyze. In about ten seconds, Podium returns a rubric, and a time series view: pitch, loudness "
                  "and pace in colour, the reference in grey, warped onto this recording's timeline word by word. Each "
                  "shaded band is a detected flaw. The dotted boxes on top are the injected ground truth, so you can check "
                  "the timing against the real answer. Below, every flaw is explained with the numbers behind it, plus a "
                  "concrete tip. Click one, and you hear it.")

    def app_steps(page):
        page.goto(APP)
        page.wait_for_timeout(2500)
        idx = page.evaluate("() => refs.findIndex(r => r.id === 'test/reagan_challenger_2')")
        page.select_option("#ref", str(idx))
        page.wait_for_timeout(2500)
        page.select_option("#sample", "mirror_L4")
        page.wait_for_timeout(1500)
        page.click("#go")
        wait_result(page)
        page.wait_for_timeout(1500)
        smooth_scroll(page, y_of(page, "#results"), 2000); page.wait_for_timeout(3500)
        smooth_scroll(page, y_of(page, "#chartCard"), 2500); page.wait_for_timeout(9000)
        smooth_scroll(page, y_of(page, "#detail"), 3000); page.wait_for_timeout(6000)
        page.click(".flaw >> nth=2"); page.wait_for_timeout(4000)

    app_vid = record("app", app_steps)
    segs.append(clip("app", app_vid, tts("n_app", say["app"])))

    # the clicked flaw, heard: the 3rd detected region of that analysis
    res = json.loads(subprocess.run(["curl", "-s", "-F", "ref=test/reagan_challenger_2", "-F", "sample=mirror_L4",
                                     f"{APP}/api/analyze"], capture_output=True, text=True).stdout)
    r = res["regions"][2]
    a_r = snippet("test/reagan_challenger_2/mirror_L4.wav", max(0, r["start"] - 0.6), r["end"] + 0.6, "a_region")
    heard = concat_audio("heard", [tts("n_heard", f"{r['label']}, at {r['start']:.1f} seconds."), a_r])
    shot = OUT / "s_heard.png"
    run([BROWSER, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
         "--virtual-time-budget=60000", f"--screenshot={shot}",
         f"{APP}/?ref=test/reagan_challenger_2&sample=mirror_L4"], capture_output=True)
    segs.append(still("heard", shot, heard))

    say["tts"] = ("Now a harder case: a synthetic voice reading the same text, with flaws. Podium reports the overall style "
                  "separately. This voice is a third faster, and flatter, than Reagan, and it is not flagged on every "
                  "sentence for that. It still finds the injected awkward pause, the volume spike, the rushed words, the "
                  "flattened phrase, and the restart.")

    def tts_steps(page):
        page.goto(f"{APP}/?ref=test/reagan_challenger_1&sample=tts_en_US-ryan-medium_L4")
        wait_result(page)
        page.wait_for_timeout(1500)
        smooth_scroll(page, y_of(page, "#results"), 2000); page.wait_for_timeout(5000)
        smooth_scroll(page, y_of(page, "#chartCard"), 2500); page.wait_for_timeout(7000)
        smooth_scroll(page, y_of(page, "#detail"), 3000); page.wait_for_timeout(6000)

    segs.append(clip("tts", record("tts", tts_steps), tts("n_tts", say["tts"])))

    say["rec"] = ("You can also record yourself, straight in the browser, reading the transcript on the left, and get the "
                  "same feedback. Everything runs locally. Your voice never leaves your machine.")

    def rec_steps(page):
        page.goto(APP)
        page.wait_for_timeout(2500)
        page.click("text=Record")
        page.wait_for_timeout(9000)

    segs.append(clip("rec", record("rec", rec_steps), tts("n_rec", say["rec"])))

    ev = json.loads((ROOT / "results" / "evaluation.json").read_text())
    t = ev["test"]["metrics"]
    rows = "".join(f"<tr><td>{k.replace('_', ' ')}</td><td>{t[k]['f1']:.2f}</td><td>{t[k]['mean_iou']:.2f}</td>"
                   f"<td>{t[k]['onset_err_s']:.2f} s</td></tr>" for k in ("long_pause", "mumbled", "stutter", "monotone", "loud", "dragged", "rushed"))
    say["eval"] = ("We tuned thresholds only on the Kennedy speeches, and tested on Reagan, a different speaker and speech. "
                   "On the held out set, detected regions overlap the true flaw by eighty one percent on average, and "
                   "overall F1 is zero point five one. Pauses, mumbling and stutters are located to within six hundredths "
                   "of a second. On untouched recordings of the reference speaker, there are zero false alarms, and the "
                   "rubric falls monotonically, from the clean recording to the most flawed one, on every excerpt.")
    segs.append(still("eval", slide("s_eval", "<h2>Held-out test: Reagan (thresholds tuned on JFK only)</h2><div class='row'>"
                      f"<div><table><tr><th>flaw</th><th>F1</th><th>IoU</th><th>onset err</th></tr>{rows}</table></div>"
                      "<div class='card'><p><span class='big'>0.81</span><br>mean IoU of matched regions</p>"
                      "<p><span class='big'>0</span><br>false alarms / min, same speaker, clean</p>"
                      "<p><b>ρ = −1.00</b> rubric vs flaw level L0→L4</p><p class='m'>overall F1 0.51 (P 0.55 / R 0.48)</p></div></div>"),
                      tts("n_eval", say["eval"])))

    say["limits"] = ("It is not solved. Pacing flaws are the weakest, because slowing a phrase also stretches its pauses, so "
                     "a drag is sometimes reported as an awkward pause. And different speakers remain hard: F1 drops to "
                     "zero point two one. Next come human recorded flawed readings with human labels, and noise models "
                     "learned from paired readings, instead of fixed floors.")
    segs.append(still("limits", slide("s_limits", "<h2>Honest limitations</h2><ul><li>Pacing is weakest (F1 0.29–0.35): "
                      "a dragged phrase stretches its pauses → reported as an awkward pause</li><li>Different speakers: F1 0.21 "
                      "(vs 0.61 same speaker)</li><li>Synthetic flaws are cleaner than human ones</li></ul><p class='a'>Next: "
                      "human-recorded flawed readings · learned noise models · multiple references per text</p>"),
                      tts("n_limits", say["limits"])))

    say["end"] = ("Everything is open source and runs locally: the code, the dataset with its labels, and the evaluation. "
                  "Thanks for watching.")
    segs.append(still("end", slide("s_end", "<h1>🎙️ Podium</h1><p>Code, dataset (207 labelled recordings) and evaluation:</p>"
                      "<p class='a' style='font-size:36px'>github.com/JayPokale/podium</p><p class='m'>Open models: Whisper · "
                      "MMS aligner · Praat · WORLD · Piper. Narration: Piper TTS.</p>"), tts("n_end", say["end"])))

    lst = OUT / "segments.txt"
    lst.write_text("".join(f"file '{s}'\n" for s in segs))
    final = OUT / "podium_demo.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(final)])
    (OUT / "narration.txt").write_text("\n\n".join(say.values()) + "\n\n" + d1 + "\n" + d2 + "\n" + d3)
    print(final, f"{dur(final):.1f} s")


if __name__ == "__main__":
    main()
