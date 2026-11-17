# mojiokoshi

Apple Silicon の Mac で、動画・音声をローカルだけで文字起こしするコマンド。音声はどこにも送信しない。

- 文字起こし：[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)（Whisper large-v3-turbo）
- 話している区間の検出・話者分離：[pyannote.audio](https://github.com/pyannote/pyannote-audio)（segmentation-3.0・speaker-diarization-community-1）

## 使い方

```bash
moji 会議.mp4            # → 会議.txt / 会議.srt（動画と同じフォルダ）
moji -s 会議.mp4         # 話者分離つき → 会議.speaker.txt
moji -s inbox/*.mp4      # 複数まとめて
LANG_W=en moji talk.mp4  # 日本語以外（既定は ja）
```

`moji` は `mojiokoshi` の短縮名（どちらでも同じ）。`-s` は `--speaker` でもいい。

`-s` の出力例：

```
[話者A] 00:00:00 おはようございます。今日は来週のリリースについて話したいと思います。
[話者B] 00:00:06 わかりました。テストはもう全部終わっていますか?
```

## 取りこぼしと作り文を減らす工夫

Whisper は、無音や雑音から「ご視聴ありがとうございました」のような文を作ったり、長い無音の後の小さい声を落としたりする。これを防ぐため、次のようにしている。

- 人が話している区間を先に調べ、その区間だけを文字起こしする。区間の外に出てきた言葉は捨てる
- 聞き取りに失敗した所（倍速再生の動画の声・早口など）は、0.7倍にゆっくりして聞き直す
- 同じ文字のくり返し、直前の文の2回書き、決まり文句だけの行は捨てる
- 捨てた文は、時刻と理由を付けて最後に表示する（本物の発言を捨てていないか確かめられる）

かかる時間の目安（M3・メモリ16GB の Mac）：1時間の録画で、通常は約25分、`-s` は約35分。メモリが足りないと遅くなるので、長い録画はブラウザなどを閉じてから回す。

## インストール

```bash
git clone https://github.com/sakuya714/mojiokoshi.git
cd mojiokoshi
./install.sh
```

Homebrew が必要。ffmpeg と uv は無ければ入れる。初回実行時にモデル（約1.6GB）をダウンロードする。

## Hugging Face の準備（初回のみ）

話している区間の検出と話者分離に使う。無くても文字起こしはできるが、作り文を捨てる処理が働かない。

1. [Hugging Face](https://huggingface.co) に登録する
2. [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) と [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) で利用規約に同意する
3. Read 権限のアクセストークンを発行し、`.venv/bin/hf auth login` でログインする

トークンは Hugging Face の標準の場所（`~/.cache/huggingface/`）に保存され、このリポジトリには入らない。

## 構成

| パス | 中身 |
|---|---|
| `mojiokoshi` | コマンド本体（zsh）。`~/.local/bin/mojiokoshi` からリンクされる |
| `transcribe.py` | 本体：話している区間の検出（`-s` なら話者分離も）→ 区間ごとに文字起こし → 作り文を捨てる → 書き出し |
| `install.sh` | `.venv` 作成とリンク作成 |
| `.venv/` | 依存ライブラリ（Git 管理外） |

## ライセンス

コードは MIT。使用モデルはそれぞれのライセンスに従う：

- Whisper（OpenAI）：MIT
- pyannote.audio：MIT
- pyannote/speaker-diarization-community-1：CC-BY-4.0
