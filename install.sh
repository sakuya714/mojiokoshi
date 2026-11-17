#!/bin/zsh
# mojiokoshi のセットアップ（Apple Silicon の Mac 用）
set -e
cd "${0:A:h}"
[[ "$(uname -m)" == "arm64" ]] || { echo "Apple Silicon の Mac 専用"; exit 1; }
command -v brew >/dev/null || { echo "Homebrew を先に入れて: https://brew.sh"; exit 1; }
command -v ffmpeg >/dev/null || brew install ffmpeg
command -v uv >/dev/null || brew install uv
uv sync                                   # .venv に whisper・pyannote を入れる
mkdir -p ~/.local/bin
ln -sf "$PWD/mojiokoshi" ~/.local/bin/mojiokoshi
ln -sf "$PWD/mojiokoshi" ~/.local/bin/moji          # 短縮名
[[ ":$PATH:" == *":$HOME/.local/bin:"* ]] || echo '⚠ ~/.local/bin が PATH に無い。~/.zshrc に export PATH="$HOME/.local/bin:$PATH" を足して'
echo "✔ インストール完了。話者分離（--speaker）を使うなら README の「話者分離の準備」をやって"
