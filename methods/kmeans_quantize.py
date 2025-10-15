# Квантування зображення (KMeans) для k=64,32,16,8
# mlcyberpunk + розклад по папкам + відносні шляхи

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Dict, Optional
from PIL import Image
from sklearn.cluster import KMeans

def _glow(cyberpunk: bool):
    if cyberpunk:
        try:
            import mplcyberpunk as mcp  # noqa: F401
            mcp.add_glow_effects()
        except Exception:
            pass

def _make_synthetic() -> np.ndarray:
    """Створює синтетичне 256x256 RGB-зображення з блоків + шуму."""
    H, W = 256, 256
    rng = np.random.default_rng(42)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(0, H, 64):
        for j in range(0, W, 64):
            img[i:i+64, j:j+64] = rng.integers(0, 256, size=3, dtype=np.uint8)
    noise = rng.integers(0, 50, size=(H, W, 3), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    return img

def _load_image(path: str) -> np.ndarray:
    """Завантажує RGB-зображення у вигляді np.ndarray."""
    return np.array(Image.open(path).convert("RGB"))

def _quantize_kmeans(arr: np.ndarray, k: int) -> np.ndarray:
    """Квантування кольорів через KMeans до k центрів."""
    h, w, _ = arr.shape
    X = arr.reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=int(k), n_init=5, random_state=42).fit(X)
    centers = km.cluster_centers_.astype(np.uint8)
    return centers[km.labels_].reshape(h, w, 3)

def _safe_tag(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name) or "image"

def _process_one(img: np.ndarray, tag: str, levels: Iterable[int], out_dir: str, cyberpunk: bool) -> Dict[str, str]:
    """Кожне джерело у своїй папці: quantization/<tag>/img/"""
    here = os.path.join(out_dir, tag)
    img_dir = os.path.join(here, "img")
    os.makedirs(img_dir, exist_ok=True)
    saved: Dict[str, str] = {}

    # 1) оригінал
    orig_path = os.path.join(img_dir, f"{tag}_orig.png")
    Image.fromarray(img).save(orig_path)
    saved["orig"] = orig_path

    # 2) квантування
    k_list = [int(k) for k in levels]
    for k in k_list:
        q = _quantize_kmeans(img, k)
        outp = os.path.join(img_dir, f"{tag}_quant_k{k}.png")
        Image.fromarray(q).save(outp)
        saved[f"k{k}"] = outp

    # 3) прев’ю (мінімальний k)
    kmin = min(k_list)
    qmin = _quantize_kmeans(img, kmin)
    plt.figure(figsize=(5, 5), dpi=120)
    plt.imshow(qmin); plt.axis("off")
    plt.title(f"[mlcyberpunk] {tag}: KMeans (k={kmin})")
    plt.tight_layout()
    _glow(cyberpunk)
    preview_path = os.path.join(img_dir, f"{tag}_quant_k{kmin}_preview.png")
    plt.savefig(preview_path); plt.close()
    saved["preview"] = preview_path

    return saved

def run(image_path: Optional[str] = None,
        extra_image_path: Optional[str] = None,
        levels: Iterable[int] = (64, 32, 16, 8),
        out_dir: str = "outputs/quantization",
        process_synthetic: bool = True,
        cyberpunk: bool = False) -> Dict[str, Dict[str, str]]:

    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, Dict[str, str]] = {}

    # user
    if image_path:
        if os.path.exists(image_path):
            results["user"] = _process_one(_load_image(image_path), "user_"+_safe_tag(image_path), levels, out_dir, cyberpunk)
        else:
            print(f"[quant] WARN: not found: {image_path}")

    # extra
    if extra_image_path:
        if os.path.exists(extra_image_path):
            results["extra"] = _process_one(_load_image(extra_image_path), "extra_"+_safe_tag(extra_image_path), levels, out_dir, cyberpunk)
        else:
            print(f"[quant] WARN: not found: {extra_image_path}")

    # synthetic (ВКЛ по умолчанию)
    if process_synthetic:
        results["synthetic"] = _process_one(_make_synthetic(), "synthetic", levels, out_dir, cyberpunk)

    # короткий README
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Quantization (mlcyberpunk)\n")
        f.write(f"- Levels: {', '.join(map(str, levels))}\n")
        f.write("- Each source stored in its own folder under ./quantization/\n")

    print("Done: Part 2 →", out_dir)
    return results

if __name__ == "__main__":
    run()
