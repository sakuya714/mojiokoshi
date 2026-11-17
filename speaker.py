"""話者付き文字起こし: mlx-whisper + pyannote。出力: {name}.speaker.txt"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, mlx_whisper
from mlx_whisper.audio import load_audio, SAMPLE_RATE
from pyannote.audio import Pipeline

src, out_dir, name, lang = sys.argv[1:5]
audio = load_audio(src)  # ffmpegで16kHzモノラルに変換

print("  1/3 文字起こし中…", flush=True)
res = mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
    language=lang, condition_on_previous_text=False, word_timestamps=True,
    hallucination_silence_threshold=2, verbose=None)

print("  2/3 話者分離中…", flush=True)
pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
if torch.backends.mps.is_available():
    pipe.to(torch.device("mps"))
out = pipe({"waveform": torch.from_numpy(np.array(audio, dtype=np.float32))[None], "sample_rate": SAMPLE_RATE})
ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", out)
turns = [(t.start, t.end, spk) for t, _, spk in ann.itertracks(yield_label=True)]

labels = {}
def label(spk):
    if spk not in labels: labels[spk] = "話者" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[len(labels) % 26]
    return labels[spk]

def who(s, e):
    best, ov = None, 0.0
    for ts, te, spk in turns:
        o = min(e, te) - max(s, ts)
        if o > ov: best, ov = spk, o
    return label(best) if best else "不明"

print("  3/3 突き合わせ中…", flush=True)
# 単語ごとに話者を決めて、話者が変わるところで区切る
lines, cur, buf, start = [], None, [], 0.0
for seg in res["segments"]:
    for w in seg.get("words") or [{"word": seg["text"], "start": seg["start"], "end": seg["end"]}]:
        sp = who(w["start"], w["end"])
        if sp != cur and buf:
            lines.append((start, cur, "".join(buf).strip())); buf = []
        if not buf: start, cur = w["start"], sp
        buf.append(w["word"])
if buf: lines.append((start, cur, "".join(buf).strip()))

def ts(x):
    x = int(x); return f"{x//3600:02d}:{x%3600//60:02d}:{x%60:02d}"
path = os.path.join(out_dir, name + ".speaker.txt")
with open(path, "w") as f:
    for s, sp, t in lines:
        if t: f.write(f"[{sp}] {ts(s)} {t}\n")
print(f"  話者数: {len(labels)}", flush=True)
