# TF-IDF + LogisticRegression, wordcloud/барчарти (mlcyberpunk, відносні шляхи, папки)

import os, re, bz2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def _glow(cyberpunk: bool):
    if cyberpunk:
        try:
            import mplcyberpunk as mcp  # noqa: F401
            mcp.add_glow_effects()
        except Exception:
            pass

try:
    from wordcloud import WordCloud
    WORDCLOUD = True
except Exception:
    WORDCLOUD = False

def _read_fasttext(path, max_rows=None):
    texts, labels = [], []
    with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows: break
            line = line.strip()
            if not line or not line.startswith("__label__"): continue
            p = line.split(" ", 1)
            if len(p) != 2: continue
            labels.append(p[0].replace("__label__", "").strip())
            texts.append(p[1])
    return pd.DataFrame({"text": texts, "label": labels})

def _clean(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def run(train_path="data/train.ft.txt.bz2",
        test_path="data/test.ft.txt.bz2",
        max_train=12000, max_test=4000,
        tfidf_max_features=15000,
        out_dir="outputs/text",
        cyberpunk=False):
    img_dir = os.path.join(out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    train = _read_fasttext(train_path, max_rows=max_train)
    test  = _read_fasttext(test_path,  max_rows=max_test)
    train["clean"] = train["text"].apply(_clean)
    test["clean"]  = test["text"].apply(_clean)

    # WordCloud або fallback барчарти
    for lab, sub in train.groupby("label"):
        blob = " ".join(sub["clean"].tolist())
        toks = blob.split()
        if WORDCLOUD and len(toks) > 0:
            wc = WordCloud(width=800, height=400, background_color="black",
                           colormap="Greens", collocations=False).generate(blob)
            plt.figure(figsize=(8,4), dpi=120); plt.imshow(wc); plt.axis("off")
            plt.title(f"[mlcyberpunk] WordCloud — class {lab}"); plt.tight_layout()
            _glow(cyberpunk)
            plt.savefig(os.path.join(img_dir, f"wordcloud_{lab}.png")); plt.close()
        else:
            top = Counter(toks).most_common(30)
            words, counts = zip(*top) if top else ([], [])
            plt.figure(figsize=(8,4), dpi=120)
            plt.bar(range(len(words)), counts)
            plt.xticks(range(len(words)), words, rotation=90)
            plt.title(f"[mlcyberpunk] Top words — class {lab}")
            plt.tight_layout()
            _glow(cyberpunk)
            plt.savefig(os.path.join(img_dir, f"top_words_{lab}.png")); plt.close()

    vec = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1,2), stop_words="english")
    Xtr = vec.fit_transform(train["clean"]); ytr = train["label"].values
    Xte = vec.transform(test["clean"]);  yte = test["label"].values

    clf = LogisticRegression(max_iter=200, solver="liblinear")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc  = float(accuracy_score(yte, pred))
    rep  = classification_report(yte, pred)

    with open(os.path.join(out_dir, "text_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{rep}")

    cm = confusion_matrix(yte, pred)
    plt.figure(figsize=(4,4), dpi=140)
    plt.imshow(cm, aspect='auto')
    plt.title("[mlcyberpunk] Confusion matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    _glow(cyberpunk)
    plt.savefig(os.path.join(img_dir, "confusion_matrix.png")); plt.close()

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# Text classification (mlcyberpunk)\nAccuracy: {acc:.4f}\n")

    print("Done: Part 3 →", out_dir)
