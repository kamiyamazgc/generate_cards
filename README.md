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
4. **日次ダイジェスト `_daily_digest.md`** を上書き生成  
   * 新規カードの「タイトル／公開日／要約」＋エラーログを掲載  

## インストール手順

```bash
git clone <repo-url>
cd Cards
python -m venv venv
source venv/bin/activate     # Windows は .\venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-... # または実行時に --key
```

## 使い方

```bash
# urls.txt に URL を 1 行ずつ
python generate_cards.py urls.txt --key sk-...
# 単一 URL を直接指定
python generate_cards.py https://example.com --key sk-...
```

- 生成カードは `Library/` 以下に自動で振り分け  
- ダイジェストは `Library/_daily_digest.md` に上書き

## フォルダ構成例

```
Library/
  ├── 007_information-science/
  │   └── 2025-05-02-gemini-text-simplification.md
  ├── 180_christianity/
  │   └── 2025-05-09-new-pope.md
  ├── _uncategorized/
  └── _daily_digest.md
```

## 主な CLI オプション

| オプション       | 説明                                |
|------------------|-------------------------------------|
| `--key`          | OpenAI API キーを直接指定           |
| `--test` / `-t`  | API 接続確認（“pong” 応答）だけ実行 |

## 依存ライブラリ管理

- 最小依存は `requirements.txt`  
- 環境固定したい場合は  
  ```bash
  pip freeze > requirements-lock.txt
  ```

## よくある Q & A

**Q. 速度が遅い / コストが高い**  
A. 要約とキーワードを同時プロンプトにする、URL を非同期で処理する、などで短縮可能。

**Q. NDC 分類が空欄になる**  
A. LLM が判断を保留した場合です。手動で追記するか、ガイドラインを追加してください。

**Q. 英語分類名が変？**  
A. `ndc10_3rd.json` の `"en"` フィールドを編集するとフォルダ名に反映されます。

---

ライセンス: MIT