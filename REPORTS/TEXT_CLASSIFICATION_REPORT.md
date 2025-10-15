# TEXT_CLASSIFICATION_REPORT (mlcyberpunk)

**Модель:** TF‑IDF (уні+біграми, stopwords="english") + LogisticRegression (liblinear).

**Точність (Accuracy): 0.8595**

## Клас-метрики

| class | precision | recall | f1 | support |
|------:|----------:|-------:|---:|--------:|
| 1 | 0.87 | 0.84 | 0.85 | 1951 |
| 2 | 0.85 | 0.88 | 0.87 | 2049 |

**Баланс класів:** приблизно збалансовані (всього 4000 прикладів тесту).

## Інтерпретація
- **Висока базова якість (~86%)** для простої лінійної моделі свідчить, що словоформи/біграми добре відокремлюють класи.
- **Precision/Recall** обох класів близькі → немає суттєвого перекосу в одну сторону.
- **Матриця плутанини** (нижче) показує симетричні помилки: модель іноді плутає близькі за лексикою тексти.

## Візуалізація частот (WordCloud)
- Клас 1: [wordcloud_1](../outputs/text/img/wordcloud_1.png)
- Клас 2: [wordcloud_2](../outputs/text/img/wordcloud_2.png)

## Confusion matrix
[confusion_matrix.png](../outputs/text/img/confusion_matrix.png)

## Рекомендації для покращення
- Збільшити `max_features` TF‑IDF (наприклад, 50–100k) і ввімкнути `sublinear_tf=True`.
- Перебрати `C` у LogisticRegression (логшук GridSearch/RandomizedSearch).
- Додати **характеристики n‑gram символів** (Character n-grams) для орфографічних варіацій.
- Балансування класів: `class_weight='balanced'`, якщо з’явиться дисбаланс.
- Перевірити лематизацію/стемінг, видалення чисел/URL/HTML.
- Для сильнішої моделі: лінійний SVM (`LinearSVC`) на тих самих ознаках.