"""文字起こし本体（mojiokoshi コマンドから呼ばれる）。

1. pyannote で「人が話している区間」を調べる（VAD）
2. Whisper（mlx-whisper / large-v3-turbo）で、その区間だけを文字起こしする
   - 長い無音の後の小さい声を落とさない（全体を一度に渡すと、無音に引きずられて声を取りこぼすことがある）
   - 無音判定で文を捨てる設定（hallucination_silence_threshold）は使わない。本物の小さい声まで消えるため
3. 作り文を消す：話している区間の外にある言葉と、Whisper の決まり文句（「ご視聴ありがとうございました」等）だけの行
4. 通常：{name}.txt / {name}.srt を書く。-s：話者分離して {name}.speaker.txt を書く

使い方: python transcribe.py 入力 出力フォルダ 名前 言語 [--speaker]
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import mlx_whisper
from mlx_whisper.audio import load_audio, SAMPLE_RATE

MODEL = "mlx-community/whisper-large-v3-turbo"
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
        vad = VoiceActivityDetection(segmentation=Model.from_pretrained("pyannote/segmentation-3.0"))
        vad.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
        out = vad(wav)
        ann = getattr(out, "speech", out)
        regions = [(s.start, s.end) for s in ann.get_timeline().support()]
        turns = None
        if diarize:
            pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
            if dev:
                pipe.to(dev)
            d = pipe(wav)
            da = getattr(d, "exclusive_speaker_diarization", None) or getattr(d, "speaker_diarization", d)
            turns = [(t.start, t.end, spk) for t, _, spk in da.itertracks(yield_label=True)]
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
    """話している区間 → Whisper に渡す区間（前後に余裕を足し、1秒未満のすき間はつなぐ）"""
    out: list[list[float]] = []
    for s, e in regions:
        s, e = max(0.0, s - 0.3), min(dur, e + 0.3)
        if out and s - out[-1][1] < 1.0:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [x for p in out for x in p]


def stock(text: str) -> bool:
    t = text.strip(" 。、.,!?！？「」")
    return len(t) <= 20 and any(k in t for k in STOCK)


def ts_srt(x: float) -> str:
    ms = int(round(x * 1000))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def ts_short(x: float) -> str:
    x = int(x)
    return f"{x // 3600:02d}:{x % 3600 // 60:02d}:{x % 60:02d}"


def main():
    src, out_dir, name, lang = sys.argv[1:5]
    speaker = "--speaker" in sys.argv[5:]
    audio = load_audio(src)  # ffmpeg で 16kHz モノラルにする

    print("  1/3 話している区間を調べ中…" + ("（話者分離も）" if speaker else ""), flush=True)
    regions, turns = speech_regions(audio, speaker)
    if regions == []:
        print("  話している所が見つからなかった", flush=True)

    print("  2/3 文字起こし中…", flush=True)
    kw = {"clip_timestamps": clips(regions, len(audio) / SAMPLE_RATE)} if regions else {}
    res = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL, language=lang, verbose=None,
                                 condition_on_previous_text=False, word_timestamps=True, **kw) if regions != [] else {"segments": []}

    print("  3/3 仕上げ中…", flush=True)
    # 言葉単位で、話している区間の中にあるものだけ残す
    segs, dropped = [], []
    for seg in res["segments"]:
        words = seg.get("words") or [{"word": seg["text"], "start": seg["start"], "end": seg["end"]}]
        kept = [w for w in words if keep(w["start"], w["end"], regions)]
        gone = "".join(w["word"] for w in words if w not in kept).strip()
        if gone:
            dropped.append((seg["start"], gone))
        text = "".join(w["word"] for w in kept).strip()
        if text and stock(text):
            dropped.append((seg["start"], text))
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
            return label(best) if best is not None else "話者?"

        lines, cur, buf, start = [], None, [], 0.0
        for seg in segs:
            for w in seg["words"]:
                sp = who(w["start"], w["end"])
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
                    f.write(f"[{sp}] {ts_short(s)} {t}\n")
        print(f"  話者数: {len(labels)}", flush=True)
    else:
        with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(s["text"] for s in segs) + ("\n" if segs else ""))
        with open(os.path.join(out_dir, name + ".srt"), "w", encoding="utf-8") as f:
            for i, s in enumerate(segs, 1):
                f.write(f"{i}\n{ts_srt(s['start'])} --> {ts_srt(s['end'])}\n{s['text']}\n\n")

    for s, t in dropped:
        print(f"  （話していない所の文を除いた {ts_short(s)}：{t[:30]}）", flush=True)


if __name__ == "__main__":
    main()
