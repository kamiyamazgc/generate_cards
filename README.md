# Card‑Maker 📚✂️  
_URL リストまたは単一 URL から自動で「情報カード」Markdown を生成するツール_

## 概要

1. **URL 取得** — `urls.txt` に列挙されたページを取得し、タイトル／公開日／本文を抽出  
2. **LLM で処理**（GPT‑4o‑mini）  
   * 300 字程度の日本語要約  
   * キーワード 8 語抽出  
   * NDC10（日本十進分類）の 3 桁コード分類（バリデーション付き）  
   * 原文が日本語以外の場合のみ全文翻訳  
3. **Markdown ファイル生成**  
   * YAML Front Matter 付き  
   * ファイル名: `YYYY‑MM‑DD‑<slug>.md`  
   * 保存先: `Library/<NDC>_<英語分類名>/`  
4. **日次ダイジェスト** を `Library/_digests/<YYYY-MM-DD>.md` に保存
   * 最新ファイルへの互換リンク `_daily_digest.md` も生成
   * 新規カードの「タイトル／公開日／要約」＋エラーログを掲載

## インストール手順

```bash
git clone <repo-url>
cd generate_cards

# ▼ Python 3.11.x 必須（Whisper の動作に必要）
python -m venv venv
source venv/bin/activate      # Windows は .\venv\Scripts\activate

# 本番 + 開発依存を一括インストール
pip install -r requirements.txt -r requirements-dev.txt
# YouTube URL を処理する場合、`yt-dlp` と `openai-whisper` もインストールされます。

export OPENAI_API_KEY=sk-...  # または実行時に --key
```

> **Codex CI での実行**
> `scripts/setup.sh` がシステム Python を検知し、3.8 未満の場合は
> `uv` 経由で **Python 3.11** と仮想環境 `.venv/` を自動生成します。
> ローカルでも同じ環境を再現したい場合は `bash scripts/setup.sh` を実行してください。

Whisper を使用する音声書き起こしでは、CUDA 対応 GPU や Apple Silicon の MPS が利用可能な場合は自動でそちらを使用します。

## 使い方

```bash
# urls.txt に URL を 1 行ずつ
python generate_cards.py urls.txt --key sk-...
# 単一 URL を直接指定
python generate_cards.py https://example.com --key sk-...
# 翻訳せずに原文だけ保存したい場合
python generate_cards.py https://example.com --no-translate --key sk-...
```

- 生成カードは `Library/` 以下に自動で振り分け
- ダイジェストは `Library/_digests/` 以下に日付別保存
  - 最新版は `Library/_daily_digest.md` としてリンク/コピー
- YouTube URL を指定した場合は音声を取得し、Whisper で文字起こし後にカード化

## フォルダ構成例

```
Library/
  ├── 007_information-science/
  │   └── 2025-05-02-gemini-text-simplification.md
  ├── 180_christianity/
  │   └── 2025-05-09-new-pope.md
  ├── _uncategorized/
  ├── _digests/
  │   └── 2025-05-10.md
  └── _daily_digest.md
```

## 主な CLI オプション

| オプション       | 説明                                |
|------------------|-------------------------------------|
| `--key`          | OpenAI API キーを直接指定           |
| `--test` / `-t`  | API 接続確認（“pong” 応答）だけ実行 |
| `--no-translate` | 非日本語でも翻訳せず原文だけ保存     |

## 依存ライブラリ管理

- 最小依存は `requirements.txt`  
- 開発用パッケージは requirements-dev.txt に分離しています。
- 環境固定したい場合は  
  ```bash
  pip freeze > requirements-lock.txt
  ```

## テスト実行

```bash
python -m pip install pytest
pytest
```


## よくある Q & A

**Q. 速度が遅い / コストが高い**  
A. 要約とキーワードを同時プロンプトにする、URL を非同期で処理する、などで短縮可能。

**Q. NDC 分類が空欄になる**  
A. LLM が判断を保留した場合です。手動で追記するか、ガイドラインを追加してください。

**Q. 英語分類名が変？**
A. `ndc10_3rd.json` の `"en"` フィールドを編集するとフォルダ名に反映されます。

**Q. API エラーで処理が止まる**
A. `openai.chat.completions.create` は `openai.APIError` または
`httpx.HTTPError` が発生した場合、指数バックオフで最大 3 回リトライします。
それでも失敗する場合は例外がそのまま伝播します。

---

ライセンス: MIT
