# mojiokoshi

Apple Silicon の Mac で、動画・音声をローカルだけで文字起こしするコマンド。音声はどこにも送信しない。

- 文字起こし：[mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)（Whisper large-v3-turbo）
- 話者分離：[pyannote.audio](https://github.com/pyannote/pyannote-audio)（speaker-diarization-community-1）

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

## インストール

```bash
git clone https://github.com/sakuya714/mojiokoshi.git
cd mojiokoshi
./install.sh
```

Homebrew が必要。ffmpeg と uv は無ければ入れる。初回実行時にモデル（約1.6GB）をダウンロードする。

## 話者分離の準備（`-s` を使うときだけ・初回のみ）

1. [Hugging Face](https://huggingface.co) に登録する
2. [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) で利用規約に同意する
3. Read 権限のアクセストークンを発行し、`.venv/bin/hf auth login` でログインする

トークンは Hugging Face の標準の場所（`~/.cache/huggingface/`）に保存され、このリポジトリには入らない。

## 構成

| パス | 中身 |
|---|---|
| `mojiokoshi` | コマンド本体（zsh）。`~/.local/bin/mojiokoshi` からリンクされる |
| `transcribe.py` | 話者分離：文字起こし → 話者分離 → 単語ごとに時間で突き合わせ |
| `install.sh` | `.venv` 作成とリンク作成 |
| `.venv/` | 依存ライブラリ（Git 管理外） |

## ライセンス

コードは MIT。使用モデルはそれぞれのライセンスに従う：

- Whisper（OpenAI）：MIT
- pyannote.audio：MIT
- pyannote/speaker-diarization-community-1：CC-BY-4.0
