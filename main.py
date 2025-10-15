# === Загальні ===
# Зниження розмірності (Dimensionality Reduction): перетворення даних у меншу кількість ознак так, щоб зберегти якнайбільше корисної інформації.
# Кластеризація (Clustering): групування об’єктів без міток так, щоб схожі опинилися разом.
# Класифікація (Classification): навчання за мітками, щоб передбачати клас для нових прикладів.
# Масштабування (StandardScaler): перетворення ознак до нульового середнього та одиничного відхилення (щоб жодна ознака не “домінувала” просто числом).

# === PCA / t-SNE / KMeans ===
# PCA (Principal Component Analysis): лінійний метод зниження розмірності; шукає головні напрямки з максимальною дисперсією.
# Частка поясненої дисперсії: яка частина варіації даних збережена після PCA (ближче до 1 — краще).
# t-SNE: метод 2D-візуалізації, зберігає локальні сусідства; глобальні відстані не інтерпретуємо.
# Perplexity (t-SNE): “ефективне число сусідів” для t-SNE; впливає на вигляд кластерів на картинці.
# KMeans: ділить дані на k груп, мінімізуючи відстань до центрів (centroids).
# k (у KMeans): кількість кластерів; задаємо вручну.
# n_init (у KMeans): скільки разів алгоритм пробує різні ініціалізації центрів і обирає найкращу.

# === Метрики якості ===
# Silhouette Score: від -1 до 1; більший — кластери щільніші всередині та далі один від одного.
# Accuracy: частка правильних передбачень (для класифікації).
# Precision/Recall/F1: метрики класифікації; Precision — точність позитивів, Recall — повнота, F1 — їх середнє гармонійне.
# Confusion Matrix: комутаційна матриця; показує де модель плутає класи.

# === Квантування зображень ===
# Квантування кольорів (Color Quantization): зменшення кількості різних кольорів до k репрезентативних (через KMeans).
# Артефакти квантування: “сходинки” і різкі межі при малому k (8/16), бо палітра бідніша.

# === Текст і NLP ===
# Стоп-слова (Stop-words): дуже частотні слова з малою користю для змісту (the, a, і т.п.); часто їх прибирають.
# n-грами: послідовності з n слів (уніграми — 1 слово, біграми — 2).
# TF-IDF: вага слова = частота у документі × інверсна частота в колекції; підсилює “характерні” слова, послаблює “загальні”.
# Logistic Regression (для тексту): лінійний класифікатор над TF-IDF; швидкий і сильний базовий підхід.
# WordCloud: хмара слів; великі слова — частіші в конкретному класі.
# Формат fastText: кожен рядок = "__label__<клас> " + текст (для train/test у завданні).

# === Практичні зауваги ===
# Після PCA KMeans зазвичай працює швидше (менше вимірів); силует може трохи впасти/підрости — залежить від шуму.
# t-SNE — лише для картинки/інсайту; не використовується як метрика якості.
# Для текстів: якість сильно залежить від чистки, n-грам, балансу класів і параметра C у логістичній регресії.

# === mlcyberpunk стиль ===
import matplotlib.pyplot as plt
try:
    import mplcyberpunk as mcp  # poetry add mplcyberpunk
    plt.style.use("cyberpunk")
    CYBER = True
except Exception:
    CYBER = False

import os
import argparse
from methods import pca_kmeans, kmeans_quantize, text_classify

# ТОЛЬКО ОТНОСИТЕЛЬНЫЕ ПУТИ
DEFAULT_OUT = "outputs"
DEFAULT_IMAGE = "data/images_Army-Drones/photo_2023-10-30_2023-11-06.jpg"
DEFAULT_EXTRA_IMAGE = ""            # опционально
DEFAULT_LOSSES_CSV = "data/russia_losses.csv"
DEFAULT_TRAIN = "data/train.ft.txt.bz2"
DEFAULT_TEST  = "data/test.ft.txt.bz2"

def parse_args():
    p = argparse.ArgumentParser(
        description="Lab 2 (mlcyberpunk): PCA/t-SNE + KMeans, k-means quantization, TF-IDF text"
    )
    p.add_argument("--part", choices=["all", "pca", "quant", "text"], default="all")
    p.add_argument("--out", default=DEFAULT_OUT, help="root output folder (relative)")

    # Part 1
    p.add_argument("--losses-csv", default=DEFAULT_LOSSES_CSV)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--tsne-sample", type=int, default=400)

    # Part 2
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--extra-image", default=DEFAULT_EXTRA_IMAGE)
    p.add_argument("--levels", default="64,32,16,8")
    p.add_argument("--no-synth", action="store_true")  # синтетика включена по умолчанию

    # Part 3
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--test",  default=DEFAULT_TEST)
    p.add_argument("--train-max", type=int, default=12000)
    p.add_argument("--test-max",  type=int, default=4000)
    p.add_argument("--tfidf-max-features", type=int, default=15000)
    return p.parse_args()

def _diag(args):
    print("=== Lab2 config (relative) ===")
    for k in ["part","out","losses_csv","image","extra_image","train","test"]:
        v = getattr(args, k)
        print(f"{k:12s}: {v}   exists={os.path.exists(v) if isinstance(v,str) and v else 'n/a'}")
    print(f"no_synth    : {args.no_synth}")
    print(f"mlcyberpunk : {'ON' if CYBER else 'OFF'}")
    print("==============================")

def main():
    args = parse_args()
    _diag(args)
    os.makedirs(args.out, exist_ok=True)

    if args.part in ("all", "pca"):
        pca_kmeans.run(
            csv_path=args.losses_csv,
            out_dir=os.path.join(args.out, "pca_tsne_kmeans"),
            k=args.k,
            tsne_sample=args.tsne_sample,
            cyberpunk=CYBER
        )

    if args.part in ("all", "quant"):
        levels = [int(x) for x in args.levels.split(",") if x.strip()]
        kmeans_quantize.run(
            image_path=(args.image or None),
            extra_image_path=(args.extra_image or None),
            levels=levels,
            out_dir=os.path.join(args.out, "quantization"),
            process_synthetic=not args.no_synth,  # ON by default
            cyberpunk=CYBER
        )

    if args.part in ("all", "text"):
        text_classify.run(
            train_path=args.train,
            test_path=args.test,
            max_train=args.train_max,
            max_test=args.test_max,
            tfidf_max_features=args.tfidf_max_features,
            out_dir=os.path.join(args.out, "text"),
            cyberpunk=CYBER
        )

if __name__ == "__main__":
    main()
