"""文字起こし本体（mojiokoshi コマンドから呼ばれる）。

1. pyannote で「人が話している区間」を調べる（VAD）
2. Whisper（mlx-whisper / large-v3-turbo）で、その区間を1つずつ切り出して文字起こしする
   - 長い無音の後の小さい声を落とさない（全体を一度に渡すと、無音に引きずられて声を取りこぼすことがある）
   - 無音判定で文を捨てる設定（hallucination_silence_threshold）は使わない。本物の小さい声まで消えるため
   - 聞き取りに失敗した30秒の窓（倍速再生の動画・早口など）は、0.7倍にゆっくりして聞き直す
3. 作り文を消す：話している区間の外にある言葉・Whisper が聞き取りをあきらめた文・
   直前の文のくり返し・Whisper の決まり文句（「ご視聴ありがとうございました」等）だけの行
4. 通常：{name}.txt / {name}.srt を書く。-s：話者分離して {name}.speaker.txt を書く

使い方: python transcribe.py 入力 出力フォルダ 名前 言語 [--speaker]
"""
from __future__ import annotations
import difflib, os, subprocess, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # 区間ごとの「Fetching 4 files」表示を消す

import numpy as np
import mlx_whisper
from mlx_whisper.audio import load_audio, SAMPLE_RATE

MODEL = "mlx-community/whisper-large-v3-turbo"
TEMPO = 0.7        # 聞き取りに失敗した所を聞き直すときの再生速度（0.7倍）
LANG = "ja"
GAP = 2.0          # 話している区間どうしのすき間がこれ未満なら、1つにつないで Whisper に渡す（秒）
PAD = 0.4          # 話している区間の前後に足す余裕（秒）。言葉の頭・お尻が切れないように
# Whisper が無音・雑音から作りがちな決まり文句。これ「だけ」の行は消す
STOCK = ("ご視聴ありがとうございました", "最後までご視聴", "チャンネル登録", "高評価", "字幕", "おやすみなさい")


def speech_regions(audio, diarize: bool):
    """話している区間 [(start, end)] と、話者分離の結果 [(start, end, speaker)] を返す。
    pyannote が使えない（Hugging Face 未ログイン等）ときは (None, None) を返し、捨てる処理を飛ばす。"""
    try:
        import torch
        from pyannote.audio import Model, Pipeline
        from pyannote.audio.pipelines import VoiceActivityDetection
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ pyannote を読めないので、作り文の除去を飛ばす（{e}）", flush=True)
        return None, None
    wav = {"waveform": torch.from_numpy(np.array(audio, dtype=np.float32))[None], "sample_rate": SAMPLE_RATE}
    dev = torch.device("mps") if torch.backends.mps.is_available() else None
    try:
        seg = Model.from_pretrained("pyannote/segmentation-3.0")
        vad = VoiceActivityDetection(segmentation=seg)
        vad.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
        if dev:
            vad.to(dev)
        out = vad(wav)
        ann = getattr(out, "speech", out)
        regions = [(s.start, s.end) for s in ann.get_timeline().support()]
        turns = None
        if diarize:
            pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
            # 一度に処理する量を減らす（既定の32だと、1時間の録画でメモリが足りなくなって止まる）
            pipe.embedding_batch_size = pipe.segmentation_batch_size = 8
            if dev:
                pipe.to(dev)
            last = [-1]

            def hook(step, *_, completed=None, total=None, **__):
                if step == "embeddings" and completed is not None and total:
                    pct = completed * 100 // total
                    if pct // 10 != last[0]:
                        last[0] = pct // 10
                        print(f"    話者分離 {pct}%", flush=True)

            d = pipe(wav, hook=hook)
            da = getattr(d, "exclusive_speaker_diarization", None) or getattr(d, "speaker_diarization", d)
            turns = [(t.start, t.end, spk) for t, _, spk in da.itertracks(yield_label=True)]
            del pipe, d
        del vad, seg, wav
        if dev:
            torch.mps.empty_cache()   # Whisper に使うメモリを空ける
        return regions, turns
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ 話している区間を調べられないので、作り文の除去を飛ばす（{e}）", flush=True)
        return None, None


def keep(t0: float, t1: float, regions) -> bool:
    if regions is None:
        return True
    mid = (t0 + t1) / 2
    return any(s - PAD <= mid <= e + PAD for s, e in regions)


def clips(regions, dur: float):
    """話している区間 → Whisper に渡す区間 [(start, end)]。前後に余裕を足し、GAP 秒未満のすき間はつなぐ。
    区間どうしは重ならない（重ねると、同じ発言を2回書いたり、端の言葉が切れたりする）"""
    out: list[list[float]] = []
    for s, e in regions:
        s, e = max(0.0, s - 0.3), min(dur, e + 0.3)
        if out and s - out[-1][1] < GAP:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def stock(text: str) -> bool:
    t = text.strip(" 。、.,!?！？「」")
    return len(t) <= 20 and any(k in t for k in STOCK)


def bad(seg) -> bool:
    """Whisper が聞き取りをあきらめた文（ゆっくりにして聞き直す目安）。"""
    return seg["avg_logprob"] < -1.0 or seg["compression_ratio"] > 2.4


def junk(seg) -> bool:
    """聞き取りをあきらめた文のうち、捨てるもの：同じ文字のくり返し（「てこてこ…」）と、
    確からしさが低い2文字以下の切れ端。話している区間の中の聞き取りにくい発言そのものは残す"""
    t = seg["text"].strip(" 。、.,!?！？")
    return seg["compression_ratio"] > 2.4 or (seg["avg_logprob"] < -1.0 and len(t) <= 2)


def score(segs) -> float:
    """文字数で重みを付けた平均の確からしさ（高いほど良い）。文が無ければ最低点。"""
    n = sum(len(s["text"].strip()) for s in segs)
    return sum(len(s["text"].strip()) * s["avg_logprob"] for s in segs) / n if n else -99.0


def run(x, off: float, scale: float = 1.0):
    """Whisper で文字起こしし、時刻を元の動画の時刻に直して返す（scale：ゆっくりにした分を戻す）"""
    r = mlx_whisper.transcribe(x, path_or_hf_repo=MODEL, language=LANG, verbose=None,
                               condition_on_previous_text=False, word_timestamps=True)
    for sg in r["segments"]:
        sg["win"] = sg["seek"] / 100 * scale + off     # Whisper が読んだ30秒の窓の頭（seek は 1/100 秒単位）
        sg["start"] = sg["start"] * scale + off
        sg["end"] = sg["end"] * scale + off
        for w in sg.get("words") or []:
            w["start"] = w["start"] * scale + off
            w["end"] = w["end"] * scale + off
    return r["segments"]


def slow(x, tempo: float):
    """音の高さを変えずに、再生速度を tempo 倍にする（ffmpeg の atempo）"""
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "-",
                        "-filter:a", f"atempo={tempo}", "-f", "f32le", "-"],
                       input=x.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32)


def transcribe_clip(x, off: float):
    """1区間を文字起こしする。Whisper が聞き取りに失敗した30秒の窓（倍速再生の動画・早口など）は、
    ゆっくりにして聞き直し、確からしさが上がった方を使う。"""
    segs = run(x, off)
    wins = sorted({s["win"] for s in segs})
    failed = [w for w in wins if any(s["temperature"] > 0 or bad(s) for s in segs if s["win"] == w)]
    if not failed:
        return segs
    end = off + len(x) / SAMPLE_RATE
    edge = {w: (wins[i + 1] if i + 1 < len(wins) else end) for i, w in enumerate(wins)}
    retry = run(slow(x, TEMPO), off, TEMPO)
    for w in failed:
        a = [s for s in segs if s["win"] == w]
        b = [s for s in retry if w <= s["start"] < edge[w]]
        if score(b) > score(a):
            segs = [s for s in segs if s["win"] != w] + b
    return sorted(segs, key=lambda s: s["start"])


def repeated(text: str, prev) -> bool:
    """直前の文のくり返し（Whisper が同じ文を2回書く癖）。短い相づち（「はい」等）は対象外。"""
    strip = lambda x: "".join(ch for ch in x if ch not in " 、。,.!?！？")
    t = strip(text)
    if len(t) < 10:
        return False
    near = strip("".join(s["text"] for s in prev[-3:]))
    m = difflib.SequenceMatcher(None, t, near).find_longest_match(0, len(t), 0, len(near))
    return m.size >= 0.8 * len(t)


def ts_srt(x: float) -> str:
    ms = int(round(x * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def ts_short(x: float) -> str:
    x = int(x)
    return f"{x // 3600:02d}:{x % 3600 // 60:02d}:{x % 60:02d}"


def main():
    global LANG
    src, out_dir, name, LANG = sys.argv[1:5]
    speaker = "--speaker" in sys.argv[5:]
    audio = load_audio(src)  # ffmpeg で 16kHz モノラルにする

    print("  1/3 話している区間を調べ中…" + ("（話者分離も）" if speaker else ""), flush=True)
    regions, turns = speech_regions(audio, speaker)
    if regions == []:
        print("  話している所が見つからなかった", flush=True)

    print("  2/3 文字起こし中…", flush=True)
    dur = len(audio) / SAMPLE_RATE
    if regions is None:            # 区間が分からないときは全体を1回で
        raw = transcribe_clip(np.array(audio, dtype=np.float32), 0.0)
    else:                          # 話している区間ごとに切り出して文字起こし（区間の頭から声が始まるので安定する）
        a = np.array(audio, dtype=np.float32)
        regions = clips(regions, dur)   # 以後は「Whisper に渡した区間」の外の言葉を作り文として消す
        raw, last = [], -1
        for s0, e0 in regions:
            if int(s0 / dur * 10) != last:
                last = int(s0 / dur * 10)
                print(f"    {last * 10}%（{ts_short(s0)} / {ts_short(dur)}）", flush=True)
            raw += transcribe_clip(a[int(s0 * SAMPLE_RATE):int(e0 * SAMPLE_RATE)], s0)

    print("  3/3 仕上げ中…", flush=True)
    # 言葉単位で、話している区間の中にあるものだけ残す。消した文は理由付きで最後に表示する
    segs, dropped = [], []
    for seg in raw:
        if junk(seg):
            dropped.append((seg["start"], "聞き取り失敗", seg["text"].strip() or "（空）"))
            continue
        words = seg.get("words") or [{"word": seg["text"], "start": seg["start"], "end": seg["end"]}]
        kept = [w for w in words if keep(w["start"], w["end"], regions)]
        gone = "".join(w["word"] for w in words if w not in kept).strip()
        if gone:
            dropped.append((seg["start"], "無音の所", gone))
        text = "".join(w["word"] for w in kept).strip()
        why = "決まり文句" if text and stock(text) else "くり返し" if text and repeated(text, segs) else None
        if why:
            dropped.append((seg["start"], why, text))
            continue
        if text:
            segs.append({"start": kept[0]["start"], "end": kept[-1]["end"], "text": text, "words": kept})

    if speaker:
        labels: dict = {}

        def label(spk):
            if spk not in labels:
                labels[spk] = "話者" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[len(labels) % 26]
            return labels[spk]

        def who(s, e):
            best, ov = None, 0.0
            for ts, te, spk in turns or []:
                o = min(e, te) - max(s, ts)
                if o > ov:
                    best, ov = spk, o
            if best is None:  # 重なりが無い言葉は、いちばん近い発言の話者にする
                near = min(turns or [], key=lambda t: min(abs(s - t[1]), abs(e - t[0])), default=None)
                best = near[2] if near else None
            return best

        words = [w for seg in segs for w in seg["words"]]
        spk = [who(w["start"], w["end"]) for w in words]
        # 2人が重なって話す所で、1〜2文字ごとに話者が入れ替わって行が細切れになるのを防ぐ：
        # 2文字以下・1秒未満の短い切れ端は、前の人の発言にくっつける
        i = 0
        while i < len(words):
            j = i
            while j < len(words) and spk[j] == spk[i]:
                j += 1
            short = len("".join(w["word"] for w in words[i:j]).strip()) <= 2 and words[j - 1]["end"] - words[i]["start"] < 1.0
            if short and i > 0:
                spk[i:j] = [spk[i - 1]] * (j - i)
            i = j
        lines, cur, buf, start = [], None, [], 0.0
        for w, sp in zip(words, spk):
            if sp != cur and buf:
                lines.append((start, cur, "".join(buf).strip()))
                buf = []
            if not buf:
                start, cur = w["start"], sp
            buf.append(w["word"])
        if buf:
            lines.append((start, cur, "".join(buf).strip()))
        with open(os.path.join(out_dir, name + ".speaker.txt"), "w", encoding="utf-8") as f:
            for s, sp, t in lines:
                if t:
                    f.write(f"[{label(sp) if sp is not None else '話者?'}] {ts_short(s)} {t}\n")
        print(f"  話者数: {len(labels)}", flush=True)
    else:
        with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(s["text"] for s in segs) + ("\n" if segs else ""))
        with open(os.path.join(out_dir, name + ".srt"), "w", encoding="utf-8") as f:
            for i, s in enumerate(segs, 1):
                f.write(f"{i}\n{ts_srt(s['start'])} --> {ts_srt(s['end'])}\n{s['text']}\n\n")

    if dropped:
        print(f"  作り文として消した所（{len(dropped)}件）:", flush=True)
    for s, why, t in dropped:
        print(f"    {ts_short(s)} [{why}] {t[:40]}", flush=True)


if __name__ == "__main__":
    main()
