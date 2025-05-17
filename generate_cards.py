#!/usr/bin/env python3
"""
Create YAML-Front-Matter + Markdown "information cards" from a list of URLs.
Cards are saved to ./Library/{YYYY-MM-DD-short-slug}.md
"""

import os, sys, json, datetime, pathlib, argparse, getpass
from urllib.parse import urlparse

import httpx
import trafilatura
import re
from langdetect import detect, LangDetectException
import yaml
from slugify import slugify
import dateparser
import openai
from tqdm import tqdm
from dateutil import parser as dtparser


# ensure OpenAI key is configured later via CLI/env/prompt

# ---------- config ---------------------------------------------------------

LIBRARY_DIR = pathlib.Path("Library")
MODEL_NAME  = "gpt-4o-mini"
SUMMARY_TOK = 1000        # rough budget: adjust as you like
TRANS_TOK   = 2048

LIBRARY_DIR.mkdir(exist_ok=True)

# --------- digest file constant ---------
DIGEST_FILE = LIBRARY_DIR / "_daily_digest.md"

with open("ndc10_3rd.json", encoding="utf-8") as f:
    _raw = json.load(f)

# Normalise so every entry is {"ja": "...", "en": "..."} (en may be "")
NDC_LABELS: dict[str, dict[str, str]] = {}
for code, val in _raw.items():
    if isinstance(val, str):
        # only Japanese provided
        NDC_LABELS[code] = {"ja": val, "en": ""}
    elif isinstance(val, dict):
        NDC_LABELS[code] = {"ja": val.get("ja", ""), "en": val.get("en", "")}

# ---------- helpers --------------------------------------------------------

def fetch_html(url: str) -> str:
    """Return HTML as Unicode, trying trafilatura's fetch (robust charset) first."""
    html = trafilatura.fetch_url(url)
    if html:                     # success
        return html
    # fallback
    r = httpx.get(url, follow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.text

def extract_meta(url: str, html: str) -> dict:
    """Return dict with title, date, author, text, keywords."""
    data_json = trafilatura.extract(html, url=url,
                                    output_format="json",
                                    with_metadata=True)
    if not data_json:
        return {}
    d = json.loads(data_json)

    # normalise
    pub_dt = dateparser.parse(d.get("date") or "")  # None if absent
    author = d.get("author") or ""
    fam, given = (author.split(maxsplit=1) + [""])[:2] if author else ("", "")
    keywords = [k.strip() for k in (d.get("keywords") or "").split(",") if k.strip()]

    return {
        "title": d.get("title") or "Untitled",
        "publication_date": pub_dt.date().isoformat() if pub_dt else "",
        "author_family": fam,
        "author_given": given,
        "keywords": keywords,
        "text": d.get("text") or "",
    }

# ---------- language & chunk helpers -------------------------------------

def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def chunk_text(text: str, max_chars: int = 4000):
    """Greedy split on sentence boundaries so each chunk fits within token limits."""
    sentences = re.split(r'(?<=[。.!?！？])\s*', text)
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) > max_chars and buf:
            chunks.append(buf)
            buf = s
        else:
            buf += s
    if buf:
        chunks.append(buf)
    return chunks

def translate_full(text: str) -> str:
    """Translate arbitrarily long texts to Japanese by chunking."""
    translated = []
    for chunk in chunk_text(text):
        translated.append(
            ask_openai(
                "次の文章を日本語に正確に全文翻訳してください。\n\n" + chunk,
                TRANS_TOK
            )
        )
    return "\n\n".join(translated)


# ---------- markdown tidy helpers ---------------------------------------

_SENT_END_RE = re.compile(r"([。．.!?！？])\s*\n")

def tidy_markdown_para(text: str) -> str:
    """
    Ensure a blank line between paragraphs for better MD viewers.
    1) Insert an extra newline after sentence‑ending punctuation that
       currently has only a single line break.
    2) Collapse 3+ consecutive blank lines to max 2.
    Works for Japanese '。', Chinese '．', and Western punctuation.
    """
    txt = _SENT_END_RE.sub(r"\1\n\n", text)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


# ---------- keyword extraction via LLM -----------------------------------

KEYWORD_TOP_N = 8

def extract_keywords_llm(summary: str, top_n: int = KEYWORD_TOP_N):
    """
    Ask GPT‑4o‑mini to return top N Japanese keywords, comma‑separated.
    """
    prompt = (
        f"以下の文章の主要なキーワードを{top_n}語抽出し、"
        "日本語カンマ区切りで返答してください。\n\n"
        f"{summary}"
    )
    resp = ask_openai(prompt, max_tokens=128)
    return [kw.strip() for kw in resp.split(",") if kw.strip()]

def ask_openai(prompt: str, max_tokens: int, model: str | None = None) -> str:
    rsp = openai.chat.completions.create(
        model=model or MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return rsp.choices[0].message.content.strip()

# ---------- NDC classification via LLM -----------------------------------

# concise guideline for the LLM so it focuses on true subject classification
NDC_GUIDELINE = (
    "NDC10 分類を決める際は、記事の『主題』を最優先に判断してください。\n"
    "000 総記は『百科事典・図書館・出版・ジャーナリズム一般』など "
    "メタ情報に限定される場合のみ選択します。"
    "\n主な判断基準:\n"
    "  • 宗教・教会 → 180 系\n"
    "  • 医学・保健 → 490 系\n"
    "  • 自然科学(物理/化学/生物) → 400 系\n"
    "  • 情報技術・AI・コンピュータ → 007, 548, 549\n"
    "  • 通信・スマートフォン → 547\n"
    "  • 社会・政治 → 300‑319\n"
    "  • 経済・企業・マーケ → 330‑338\n"
    "  • 技術・工学一般 → 500 系\n"
    "  • 芸術・音楽 → 700‑769\n"
    "一覧に無いか迷う場合は空欄にしてください。"
)

NDC_MODEL_NAME = "gpt-4o-mini"

_NDC_CODES_CSV = ", ".join(sorted(NDC_LABELS.keys()))

def classify_ndc_llm(title: str, summary: str) -> str:
    """
    Ask GPT model for the best 3‑digit NDC10 code.
    1) First prompt: free answer
    2) If result is invalid (not in list), re‑prompt with explicit choices
    Returns "" when still invalid.
    """
    def _ask(prompt: str) -> str:
        resp = ask_openai(prompt, max_tokens=10, model=NDC_MODEL_NAME)
        code = resp.strip()[:3]
        return code if code.isdigit() and code in NDC_LABELS else ""

    # first attempt – free form
    base_prompt = (
        f"{NDC_GUIDELINE}\n\n"
        "次のタイトルと要約に最も適切な日本十進分類法(NDC10)の3桁分類コードを"
        "一つだけ半角数字で答えてください。存在しないコードや4桁以上は不可。"
        "迷う場合は空欄で返してください。\n\n"
        f"タイトル: {title}\n要約: {summary}"
    )
    code = _ask(base_prompt)
    if code:
        return code  # valid within list

    # second attempt – force choice from list
    choice_prompt = (
        f"{NDC_GUIDELINE}\n\n"
        "以下は NDC10 の有効な3桁分類コード一覧です。\n"
        f"{_NDC_CODES_CSV}\n\n"
        "次のタイトルと要約に最も適切なコードを1つだけ選び、半角数字で答えてください。"
        "当てはまらなければ空欄で返してください。\n\n"
        f"タイトル: {title}\n要約: {summary}"
    )
    return _ask(choice_prompt)

def build_card(meta: dict, url: str, access_date: str) -> str:
    """Return full markdown string for one card."""
    domain = urlparse(url).netloc
    # ----- NDC classification
    # （途中で失敗しても str を返す必要があるため、
    #  ここから return するのは厳禁）

    source_lang = detect_lang(meta['text'][:1000])
    needs_translation = source_lang != "ja"

    # summary: always Japanese, regardless of source language
    summary = ask_openai(
        f"次の文章を日本語で300字程度で要約してください。\n\n{meta['text']}",
        SUMMARY_TOK,
    )
    summary = " ".join(summary.split())
    # store summary for later digest
    meta["summary"] = summary

    ndc_stub = classify_ndc_llm(meta["title"], summary)
    # store for downstream use
    meta["ndc"] = ndc_stub

    # automatic keywords from summary using LLM
    auto_keywords = extract_keywords_llm(summary)
    keywords_combined = list(dict.fromkeys((meta["keywords"] or []) + auto_keywords))

    original_text = tidy_markdown_para(meta['text'])
    translation = (
        tidy_markdown_para(translate_full(meta['text']))
        if needs_translation else ""
    )

    front = {
        "title": meta["title"],
        "url": url,
        "publication_date": meta["publication_date"],
        "access_date": access_date,
        "author": [{"family": meta["author_family"], "given": meta["author_given"]}]
                  if meta["author_family"] else [],
        "domain": domain,
        "ndc": ndc_stub,
        "keywords": keywords_combined,
        "summary": summary,
        "has_translation": needs_translation,
    }
    front_matter = yaml.safe_dump(
        front,
        allow_unicode=True,
        sort_keys=False,
        width=4096          # avoid PyYAML auto‑wrapping
    ).strip()
    parts = ["---", front_matter, "---", ""]

    if needs_translation:
        parts += ["## Translation （和訳）", "", translation, ""]

    parts += ["## Original Text", "", original_text]

    # --- ALWAYS return str ---
    body = "\n".join(parts).lstrip()
    return body

def save_card(content: str, meta: dict) -> pathlib.Path:
    """
    Save markdown content to ./Library/ using
    {publication_date}-{slug}.md  where slug preserves Japanese/
    multibyte characters.

    If the title is entirely non‑ASCII and slugify would Latin‑
    transliterate it into unreadable romaji, we instead keep the
    original Unicode (with symbols sanitized) so the filename is still
    human‑legible.  Length is capped to 40 chars to avoid extremely
    long path names.
    """
    ndc_code = meta.get("ndc") or "_uncategorized"

    if ndc_code != "_uncategorized":
        # attach an English‑style slug of the Japanese or English label for readability
        labels      = NDC_LABELS.get(ndc_code, {"ja": "", "en": ""})
        label_en_src = labels.get("en") or labels.get("ja")
        label_en     = slugify(label_en_src, allow_unicode=False) or "misc"
        subdir       = LIBRARY_DIR / f"{ndc_code}_{label_en}"
    else:   
        subdir = LIBRARY_DIR / ndc_code

    subdir.mkdir(parents=True, exist_ok=True)

    date_part = meta["publication_date"] or datetime.date.today().isoformat()
    slug_part = slugify(meta["title"], allow_unicode=True)[:40] or "untitled"
    path = subdir / f"{date_part}-{slug_part}.md"

    path.write_text(content, encoding="utf-8")
    return path

# ---------- main -----------------------------------------------------------

# ---------- cli / entry‑point ---------------------------------------------

def cli():
    parser = argparse.ArgumentParser(
        description="Generate YAML‑Front‑Matter + Markdown information cards from URLs."
    )
    parser.add_argument("url_file", nargs="?", help="Path to txt file containing URLs (one per line)")
    parser.add_argument("--key", help="OpenAI API key (overrides env var)")
    parser.add_argument("-t", "--test", action="store_true",
                        help="Only test the supplied / detected API key and exit")
    args = parser.parse_args()

    # resolve API key
    openai.api_key = args.key or os.getenv("OPENAI_API_KEY") \
                     or getpass.getpass("Enter your OpenAI API key: ").strip()

    if args.test:
        try:
            pong = ask_openai("Say 'pong' in one word.", max_tokens=5)
            print(f"✅ API key works. LLM responded: {pong}")
        except Exception as e:
            print(f"❌ API test failed: {e}")
        return

    if not args.url_file:
        parser.error("url_file is required unless --test is supplied.")

    generate_from_file(args.url_file)

# ---------- orchestrator ---------------------------------------------------

def generate_from_file(url_file: str):
    access_date = datetime.date.today().isoformat()
    with open(url_file, encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip()]

    new_entries = []   # collect (title, pub_date, ndc, summary, rel_path)
    error_entries = [] # collect (url, error_str)
    for url in tqdm(urls, desc="Processing"):
        try:
            html  = fetch_html(url)
            meta  = extract_meta(url, html)
            card  = build_card(meta, url, access_date)
            fp    = save_card(card, meta)
            rel = fp.relative_to(LIBRARY_DIR)
            new_entries.append((
                meta["title"],
                meta["publication_date"],
                meta.get("ndc", ""),
                meta["summary"],
                rel
            ))
            tqdm.write(f"✓ {fp}")
        except Exception as e:
            tqdm.write(f"⚠️  {url}: {e}")
            error_entries.append((url, str(e)))

    # -------- write daily digest -----------
    if new_entries:
        today = datetime.date.today().isoformat()
        def _dt(d):
            try:
                return dtparser.parse(d)
            except Exception:
                return datetime.datetime(1970, 1, 1)
        # enumerate to keep original order for tie‑break
        sorted_entries = sorted(
            enumerate(new_entries),
            key=lambda t: (
                -_dt(t[1][1]).timestamp(),          # publication date desc
                t[1][2],                            # ndc asc
                t[0]                                # original order
            )
        )
        lines = [f"# New Cards created on {today}", ""]
        for _, (title, pubdate, ndc, summ, rel) in sorted_entries:
            lines += [
                f"### [{title}]({rel})",
                f"- Publication date: {pubdate or '―'}",
                "",
                tidy_markdown_para(summ),
                ""
            ]
        if error_entries:
            lines += ["---", "## Error log", ""]
            for url, err in error_entries:
                lines.append(f"- **{url}**: {err}")
        DIGEST_FILE.write_text("\n".join(lines), encoding="utf-8")
        print(f"📝 Digest written to {DIGEST_FILE}")

if __name__ == "__main__":
    cli()