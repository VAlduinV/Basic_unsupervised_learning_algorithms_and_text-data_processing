# Зниження розмірності + KMeans + t-SNE візуалізація (mlcyberpunk, папки, відносні шляхи)

import os, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

def _glow(cyberpunk: bool):
    if cyberpunk:
        try:
            import mplcyberpunk as mcp  # noqa: F401
            mcp.make_scatter_glow()
        except Exception:
            pass

def _make_tsne(perplexity=30):
    """Сумісний конструктор для різних версій sklearn."""
    try:
        return TSNE(n_components=2, random_state=42, perplexity=perplexity,
                    init="pca", learning_rate="auto", n_iter=500)
    except TypeError:
        return TSNE(n_components=2, random_state=42, perplexity=perplexity,
                    init="pca", learning_rate=200)

def run(csv_path="data/russia_losses.csv", out_dir="outputs/pca_tsne_kmeans",
        k=3, tsne_sample=400, cyberpunk=False):
    img_dir = os.path.join(out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    # 1) Дані
    df = pd.read_csv(csv_path)
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.dropna(axis=1, how="all")
    num = num.loc[:, num.std(numeric_only=True) > 0]
    num = num.fillna(num.median(numeric_only=True))
    X = StandardScaler().fit_transform(num.values)

    # 2) KMeans (full)
    t0 = time.time()
    labels_full = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    t1 = time.time()
    sil_full = float(silhouette_score(X, labels_full))

    # 3) PCA(95%) + KMeans
    pca95 = PCA(n_components=0.95, random_state=42)
    X_pca95 = pca95.fit_transform(X)
    t2 = time.time()
    labels_p95 = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_pca95)
    t3 = time.time()
    sil_p95 = float(silhouette_score(X_pca95, labels_p95))

    # 4) PCA-2D (full labels)
    X_p2 = PCA(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(6,5), dpi=120)
    plt.scatter(X_p2[:,0], X_p2[:,1], c=labels_full, s=18)
    plt.title(f"[mlcyberpunk] PCA 2D • KMeans(k={k}) full")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
    _glow(cyberpunk)
    plt.savefig(os.path.join(img_dir, "pca2_kmeans_full.png")); plt.close()

    # 5) PCA-2D (labels після PCA95)
    plt.figure(figsize=(6,5), dpi=120)
    plt.scatter(X_p2[:,0], X_p2[:,1], c=labels_p95, s=18)
    plt.title(f"[mlcyberpunk] PCA 2D • KMeans(k={k}) after PCA(95%)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
    _glow(cyberpunk)
    plt.savefig(os.path.join(img_dir, "pca2_kmeans_pca95.png")); plt.close()

    # 6) t-SNE (sample)
    n = X.shape[0]; s = min(tsne_sample, n)
    idx = np.random.RandomState(42).choice(n, size=s, replace=False)
    X_s, y_s = X[idx], np.array(labels_full)[idx]
    perplexity = max(5, min(30, s//5))
    tsne = _make_tsne(perplexity)
    X_ts = tsne.fit_transform(X_s)
    plt.figure(figsize=(6,5), dpi=120)
    plt.scatter(X_ts[:,0], X_ts[:,1], c=y_s, s=18)
    plt.title("[mlcyberpunk] t-SNE 2D • colors = KMeans(full)")
    plt.xlabel("Dim1"); plt.ylabel("Dim2"); plt.tight_layout()
    _glow(cyberpunk)
    plt.savefig(os.path.join(img_dir, "tsne2_kmeans_full_sample.png")); plt.close()

    # 7) Порівняння
    comp = pd.DataFrame({
        "Setting": ["Full features", "PCA(95%)"],
        "n_features": [X.shape[1], X_pca95.shape[1]],
        "silhouette": [sil_full, sil_p95],
        "fit_time_sec": [t1 - t0, t3 - t2],
    })
    comp.to_csv(os.path.join(out_dir, "pca_comparison.csv"), index=False)

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# PCA/TSNE + KMeans (mlcyberpunk)\n"
                f"| Setting | n_features | silhouette | fit_time_sec |\n"
                f"|---|---:|---:|---:|\n"
                f"| Full features | {X.shape[1]} | {sil_full:.4f} | {t1 - t0:.4f} |\n"
                f"| PCA(95%)      | {X_pca95.shape[1]} | {sil_p95:.4f} | {t3 - t2:.4f} |\n")

    print("Done: Part 1 →", out_dir)
