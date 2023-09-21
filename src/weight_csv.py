import pandas as pd
import numpy as np
from collections import Counter


def find_duplicate_ngrams(sentences, n):
    ngrams = []
    for sentence in sentences:
        words = sentence.split()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            ngrams.append(ngram)
    ngram_counts = Counter(ngrams)
    return ngram_counts


def make_weight(sentence, my_count):
    weight = 0
    for n in [2, 3, 4]:
        words = sentence.split()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            weight += my_count.get(ngram, 2)
    return len(words) / (np.log2(weight))


if __name__ == "__main__":
    raw_data = pd.read_csv("data/기본train.csv")
    data = raw_data[raw_data.type == "train"]
    data.loc[:, "comments"] = data.comments.apply(
        lambda x: x.replace(". ", ".").replace(".", ". ")
    )
    ngram_counts_2 = find_duplicate_ngrams(data.comments.unique(), 2)
    ngram_counts_3 = find_duplicate_ngrams(data.comments.unique(), 3)
    ngram_counts_4 = find_duplicate_ngrams(data.comments.unique(), 4)

    my_count = {k: v for k, v in ngram_counts_2.items() if v > 1}
    my_count.update({k: v for k, v in ngram_counts_3.items() if v > 1})
    my_count.update({k: v for k, v in ngram_counts_4.items() if v > 1})

    comments = raw_data.comments.apply(
        lambda x: x.replace(". ", ".").replace(".", ". ")
    )
    raw_data["weight"] = [make_weight(x, my_count) for x in comments]
    raw_data.to_csv("data/train_weight.csv", index=False)
    pass
