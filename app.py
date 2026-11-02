# """AI-Based Sentiment Analyzer - Flask + SQLite.  Run:  python app.py  ->  http://127.0.0.1:5000"""
# import csv
# import io
# import json
# import os
# import sqlite3
# from collections import Counter, defaultdict
# from datetime import datetime

# from flask import Flask, Response, jsonify, render_template, request

# from analyzer import analyze, engine_name

# BASE = os.path.dirname(os.path.abspath(__file__))
# DB_PATH = os.path.join(BASE, "sentiment.db")
# MAX_BATCH = 5000

# app = Flask(__name__)


# def db():
#     conn = sqlite3.connect(DB_PATH)
#     conn.row_factory = sqlite3.Row
#     return conn


# def init_db():
#     with db() as conn:
#         conn.execute("""CREATE TABLE IF NOT EXISTS analyses(
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             text TEXT NOT NULL, label TEXT NOT NULL, compound REAL NOT NULL,
#             positive REAL, neutral REAL, negative REAL,
#             keywords TEXT, source TEXT, created_at TEXT NOT NULL)""")


# def save(text, result, source="single"):
#     with db() as conn:
#         conn.execute(
#             "INSERT INTO analyses(text,label,compound,positive,neutral,negative,keywords,source,created_at)"
#             " VALUES(?,?,?,?,?,?,?,?,?)",
#             (text, result["label"], result["compound"], result["positive"], result["neutral"],
#              result["negative"], json.dumps(result["keywords"]), source,
#              datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


# @app.route("/")
# def index():
#     return render_template("index.html", engine=engine_name())


# @app.post("/api/analyze")
# def api_analyze():
#     text = (request.get_json(silent=True) or {}).get("text", "").strip()
#     if not text:
#         return jsonify(error="Please enter some text."), 400
#     result = analyze(text)
#     save(text, result)
#     return jsonify(result)


# @app.post("/api/batch")
# def api_batch():
#     """Accepts a CSV/TXT file upload (field 'file') or JSON {"texts": [...]}."""
#     texts = []
#     if "file" in request.files:
#         raw = request.files["file"].read().decode("utf-8-sig", errors="ignore")
#         name = request.files["file"].filename.lower()
#         if name.endswith(".csv"):
#             rows = list(csv.reader(io.StringIO(raw)))
#             if rows:
#                 header = [h.strip().lower() for h in rows[0]]
#                 col = next((header.index(c) for c in ("text", "review", "comment", "content", "tweet", "message")
#                             if c in header), None)
#                 data = rows[1:] if col is not None else rows
#                 col = col or 0
#                 texts = [r[col].strip() for r in data if len(r) > col and r[col].strip()]
#         else:
#             texts = [l.strip() for l in raw.splitlines() if l.strip()]
#     else:
#         texts = [t.strip() for t in (request.get_json(silent=True) or {}).get("texts", []) if t.strip()]

#     if not texts:
#         return jsonify(error="No text found. Upload a CSV with a 'text' column or a .txt with one entry per line."), 400
#     texts = texts[:MAX_BATCH]
#     out = []
#     for t in texts:
#         r = analyze(t)
#         save(t, r, "batch")
#         out.append({"text": t, "label": r["label"], "compound": r["compound"]})
#     counts = Counter(o["label"] for o in out)
#     return jsonify(count=len(out), counts=counts, results=out[:200])


# @app.get("/api/report")
# def api_report():
#     with db() as conn:
#         rows = conn.execute("SELECT * FROM analyses ORDER BY id").fetchall()
#     total = len(rows)
#     counts = Counter(r["label"] for r in rows)
#     avg = round(sum(r["compound"] for r in rows) / total, 3) if total else 0

#     daily = defaultdict(list)
#     bins = [0] * 10
#     pos_words, neg_words = Counter(), Counter()
#     for r in rows:
#         daily[r["created_at"][:10]].append(r["compound"])
#         bins[min(9, int((r["compound"] + 1) / 0.2))] += 1
#         for k in json.loads(r["keywords"] or "[]"):
#             (pos_words if k["score"] > 0 else neg_words)[k["word"]] += 1
#     trend = [{"date": d, "avg": round(sum(v) / len(v), 3), "n": len(v)} for d, v in sorted(daily.items())][-14:]
#     return jsonify(total=total, counts={k: counts.get(k, 0) for k in ("positive", "neutral", "negative")},
#                    average=avg, trend=trend, histogram=bins,
#                    top_positive=pos_words.most_common(8), top_negative=neg_words.most_common(8))


# @app.get("/api/history")
# def api_history():
#     with db() as conn:
#         rows = conn.execute(
#             "SELECT id,text,label,compound,created_at FROM analyses ORDER BY id DESC LIMIT 100").fetchall()
#     return jsonify([dict(r) for r in rows])


# @app.get("/api/export")
# def api_export():
#     with db() as conn:
#         rows = conn.execute("SELECT id,text,label,compound,positive,neutral,negative,created_at "
#                             "FROM analyses ORDER BY id").fetchall()
#     buf = io.StringIO()
#     w = csv.writer(buf)
#     w.writerow(["id", "text", "label", "compound", "positive", "neutral", "negative", "created_at"])
#     w.writerows([tuple(r) for r in rows])
#     return Response(buf.getvalue(), mimetype="text/csv",
#                     headers={"Content-Disposition": "attachment; filename=sentiment_report.csv"})


# @app.post("/api/clear")
# def api_clear():
#     with db() as conn:
#         conn.execute("DELETE FROM analyses")
#     return jsonify(ok=True)


# init_db()

# if __name__ == "__main__":
#     print(f"Sentiment engine: {engine_name()}")
#     app.run(host="127.0.0.1", port=5000, debug=False)"""AI-Based Sentiment Analyzer - Flask + SQLite + Optional Hugging Face Transformers."""
import csv
import io
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime

from flask import Flask, Response, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "sentiment.db")
MAX_BATCH = 5000

app = Flask(__name__)

# ==========================================
# 1. SENTIMENT ENGINE (Lexicon + Optional Transformers)
# ==========================================
_RAW = """
love:3.2 loved:3.0 loves:3.0 loving:2.8 amazing:3.2 awesome:3.1 excellent:3.2 fantastic:3.2 wonderful:3.0
great:2.8 good:1.9 nice:1.8 best:3.0 better:1.6 perfect:3.3 happy:2.7 glad:2.0 pleased:2.0 brilliant:3.0
beautiful:2.8 superb:3.2 outstanding:3.2 impressive:2.5 impressed:2.4 recommend:2.2 recommended:2.2
enjoy:2.3 enjoyed:2.3 enjoyable:2.3 fun:2.3 delightful:2.8 delicious:2.8 smooth:1.6 fast:1.5 quick:1.4
easy:1.6 helpful:2.2 friendly:2.0 reliable:2.1 comfortable:1.9 affordable:1.7 cheap:0.8 worth:1.6 value:1.2
satisfied:2.2 satisfying:2.2 solid:1.6 stunning:3.0 gorgeous:2.9 favorite:2.5 favourite:2.5 incredible:3.1
terrific:3.0 lovely:2.6 neat:1.5 cool:1.8 super:2.3 smart:1.7 useful:1.9 clean:1.4 fresh:1.5 stable:1.4
thanks:1.5 thank:1.5 win:2.2 winning:2.2 success:2.4 successful:2.4 positive:1.9 pleasant:2.0 yay:2.4 wow:2.2
bad:-2.5 terrible:-3.2 awful:-3.2 horrible:-3.2 worst:-3.3 worse:-2.2 poor:-2.3 hate:-3.2 hated:-3.1 hates:-3.0
disappointing:-2.6 disappointed:-2.6 disappointment:-2.6 useless:-2.9 waste:-2.4 broken:-2.4 slow:-1.7
boring:-2.2 ugly:-2.5 dirty:-2.2 rude:-2.6 cheaply:-1.5 defective:-2.8 fail:-2.4 failed:-2.4 failure:-2.5
buggy:-2.2 crash:-2.2 crashes:-2.2 crashed:-2.2 annoying:-2.3 annoyed:-2.2 angry:-2.5 sad:-2.2 unhappy:-2.4
overpriced:-2.2 expensive:-1.2 regret:-2.4 refund:-1.4 scam:-3.2 fake:-2.4 faulty:-2.6 painful:-2.3
mediocre:-1.4 lag:-1.6 laggy:-1.9 problem:-1.6 problems:-1.7 issue:-1.2 issues:-1.4 complaint:-1.8
frustrating:-2.5 frustrated:-2.4 worthless:-3.0 nasty:-2.6 disgusting:-3.0 pathetic:-2.8 poorly:-2.2
unreliable:-2.4 uncomfortable:-1.9 noisy:-1.5 confusing:-1.7 difficult:-1.3 hard:-0.8 negative:-1.9
lame:-1.9 sucks:-2.7 suck:-2.7 trash:-2.8 garbage:-2.9 stupid:-2.5 hopeless:-2.6 damaged:-2.2 late:-1.3
okay:0.4 ok:0.4 fine:0.8 decent:1.2 average:0.0 alright:0.5 acceptable:0.6
"""
LEXICON = {w: float(s) for w, s in (p.split(":") for p in _RAW.split())}

NEGATIONS = {"not", "no", "never", "n't", "cannot", "neither", "nor", "without", "hardly", "barely",
             "isn't", "wasn't", "don't", "doesn't", "didn't", "won't", "can't", "couldn't", "wouldn't",
             "shouldn't", "aren't", "weren't", "haven't", "hasn't", "hadn't", "ain't"}
BOOSTERS = {"very": 1.4, "really": 1.35, "extremely": 1.6, "absolutely": 1.6, "totally": 1.4, "so": 1.3,
            "incredibly": 1.6, "super": 1.4, "highly": 1.4, "too": 1.25, "quite": 1.15, "truly": 1.4,
            "slightly": 0.6, "somewhat": 0.7, "kinda": 0.7, "barely": 0.5, "little": 0.7, "bit": 0.7}
EMOTICONS = {":)": 2.0, ":-)": 2.0, ":D": 2.6, ":-D": 2.6, "=)": 1.8, ";)": 1.4, "<3": 2.6,
             ":(": -2.0, ":-(": -2.0, ":'(": -2.4, ":/": -1.2, ":|": -0.3, ">:(": -2.6}

_WORD = re.compile(r"[A-Za-z']+")
_transformer = None
_transformer_tried = False


def _get_transformer():
    """Lazy-load Hugging Face pipeline if USE_TRANSFORMERS=1 and numpy < 2.0 is present."""
    global _transformer, _transformer_tried
    if _transformer_tried:
        return _transformer
    _transformer_tried = True
    if os.environ.get("USE_TRANSFORMERS", "0") != "1":
        return None
    try:
        from transformers import pipeline  # type: ignore
        _transformer = pipeline("sentiment-analysis",
                                model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True)
        print("[sentiment] Transformer model loaded successfully.")
    except Exception as exc:
        print(f"[sentiment] Transformers unavailable ({exc}); using robust lexicon engine.")
    return _transformer


def engine_name():
    return "lexicon + transformer" if _get_transformer() else "lexicon (offline)"


def _lexicon_score(text):
    raw_tokens = _WORD.findall(text)
    tokens = [t.lower() for t in raw_tokens]
    all_caps_text = text.isupper()
    contribs = []
    but_idx = next((i for i, t in enumerate(tokens) if t in ("but", "however", "although")), None)

    for i, tok in enumerate(tokens):
        if tok not in LEXICON:
            continue
        val = LEXICON[tok]
        for back in (1, 2):
            if i - back >= 0 and tokens[i - back] in BOOSTERS:
                val *= BOOSTERS[tokens[i - back]]
                break
        if any(tokens[j] in NEGATIONS or tokens[j].endswith("n't") for j in range(max(0, i - 3), i)):
            val *= -0.74
        if raw_tokens[i].isupper() and len(raw_tokens[i]) > 1 and not all_caps_text:
            val *= 1.25
        if but_idx is not None:
            val *= 0.5 if i < but_idx else 1.5
        contribs.append((i, tok, val))

    for sym, val in EMOTICONS.items():
        n = text.count(sym)
        if n:
            contribs.append((len(tokens), sym, val * min(n, 3)))

    total = sum(v for _, _, v in contribs)
    if total:
        total += math.copysign(0.3 * min(text.count("!"), 4), total)
    return total, contribs, len(tokens)


def analyze(text):
    text = (text or "").strip()
    total, contribs, n_tokens = _lexicon_score(text)
    compound = total / math.sqrt(total * total + 15)

    pos = sum(v for _, _, v in contribs if v > 0)
    neg = -sum(v for _, _, v in contribs if v < 0)
    neutral_words = max(n_tokens - len(contribs), 0)
    denom = pos + neg + neutral_words or 1
    pos_p, neg_p = pos / denom, neg / denom
    neu_p = max(0.0, 1 - pos_p - neg_p)

    model = _get_transformer()
    if model and text:
        try:
            out = model(text[:512])[0]
            signed = out["score"] if out["label"].upper().startswith("POS") else -out["score"]
            compound = 0.5 * compound + 0.5 * signed
        except Exception:
            pass

    label = "positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral"
    keywords = [{"word": w, "score": round(v, 2)} for _, w, v in contribs]
    return {
        "label": label,
        "compound": round(compound, 4),
        "positive": round(pos_p, 3),
        "neutral": round(neu_p, 3),
        "negative": round(neg_p, 3),
        "confidence": round(min(1.0, abs(compound) * 1.2 + 0.05), 3) if text else 0.0,
        "keywords": keywords,
    }


# ==========================================
# 2. DATABASE & FLASK ROUTES
# ==========================================
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS analyses(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL, label TEXT NOT NULL, compound REAL NOT NULL,
            positive REAL, neutral REAL, negative REAL,
            keywords TEXT, source TEXT, created_at TEXT NOT NULL)""")


def save(text, result, source="single"):
    with db() as conn:
        conn.execute(
            "INSERT INTO analyses(text,label,compound,positive,neutral,negative,keywords,source,created_at)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (text, result["label"], result["compound"], result["positive"], result["neutral"],
             result["negative"], json.dumps(result["keywords"]), source,
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


@app.route("/")
def index():
    return render_template("index.html", engine=engine_name())


@app.post("/api/analyze")
def api_analyze():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="Please enter some text."), 400
    result = analyze(text)
    save(text, result)
    return jsonify(result)


@app.post("/api/batch")
def api_batch():
    texts = []
    if "file" in request.files:
        raw = request.files["file"].read().decode("utf-8-sig", errors="ignore")
        name = request.files["file"].filename.lower()
        if name.endswith(".csv"):
            rows = list(csv.reader(io.StringIO(raw)))
            if rows:
                header = [h.strip().lower() for h in rows[0]]
                col = next((header.index(c) for c in ("text", "review", "comment", "content", "tweet", "message")
                            if c in header), None)
                data = rows[1:] if col is not None else rows
                col = col or 0
                texts = [r[col].strip() for r in data if len(r) > col and r[col].strip()]
        else:
            texts = [l.strip() for l in raw.splitlines() if l.strip()]
    else:
        texts = [t.strip() for t in (request.get_json(silent=True) or {}).get("texts", []) if t.strip()]

    if not texts:
        return jsonify(error="No text found. Upload a CSV with a 'text' column or a .txt with one entry per line."), 400
    texts = texts[:MAX_BATCH]
    out = []
    for t in texts:
        r = analyze(t)
        save(t, r, "batch")
        out.append({"text": t, "label": r["label"], "compound": r["compound"]})
    counts = Counter(o["label"] for o in out)
    return jsonify(count=len(out), counts=counts, results=out[:200])


@app.get("/api/report")
def api_report():
    with db() as conn:
        rows = conn.execute("SELECT * FROM analyses ORDER BY id").fetchall()
    total = len(rows)
    counts = Counter(r["label"] for r in rows)
    avg = round(sum(r["compound"] for r in rows) / total, 3) if total else 0

    daily = defaultdict(list)
    bins = [0] * 10
    pos_words, neg_words = Counter(), Counter()
    for r in rows:
        daily[r["created_at"][:10]].append(r["compound"])
        bins[min(9, int((r["compound"] + 1) / 0.2))] += 1
        for k in json.loads(r["keywords"] or "[]"):
            (pos_words if k["score"] > 0 else neg_words)[k["word"]] += 1
    trend = [{"date": d, "avg": round(sum(v) / len(v), 3), "n": len(v)} for d, v in sorted(daily.items())][-14:]
    return jsonify(total=total, counts={k: counts.get(k, 0) for k in ("positive", "neutral", "negative")},
                   average=avg, trend=trend, histogram=bins,
                   top_positive=pos_words.most_common(8), top_negative=neg_words.most_common(8))


@app.get("/api/history")
def api_history():
    with db() as conn:
        rows = conn.execute("SELECT id,text,label,compound,created_at FROM analyses ORDER BY id DESC LIMIT 100").fetchall()
    return jsonify([dict(r) for r in rows])


@app.get("/api/export")
def api_export():
    with db() as conn:
        rows = conn.execute("SELECT id,text,label,compound,positive,neutral,negative,created_at FROM analyses ORDER BY id").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "text", "label", "compound", "positive", "neutral", "negative", "created_at"])
    w.writerows([tuple(r) for r in rows])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=sentiment_report.csv"})


@app.post("/api/clear")
def api_clear():
    with db() as conn:
        conn.execute("DELETE FROM analyses")
    return jsonify(ok=True)


init_db()

if __name__ == "__main__":
    print(f"Sentiment engine: {engine_name()}")
    app.run(host="127.0.0.1", port=5000, debug=False)
