import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
from collections import Counter, defaultdict


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
    sentence = sentence.replace(". ", ".").replace(".", ". ")
    weight = 0
    for n in [2, 3, 4]:
        words = sentence.split()
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            weight += my_count.get(ngram, 2)
    return len(words) / (np.log2(weight))


def make_my_count():
    data = pd.read_csv("data/train.csv")
    data.loc[:, "comments"] = data.comments.apply(
        lambda x: x.replace(". ", ".").replace(".", ". ")
    )
    ngram_counts_2 = find_duplicate_ngrams(data.comments.unique(), 2)
    ngram_counts_3 = find_duplicate_ngrams(data.comments.unique(), 3)
    ngram_counts_4 = find_duplicate_ngrams(data.comments.unique(), 4)

    my_count = {k: v for k, v in ngram_counts_2.items() if v > 1}
    my_count.update({k: v for k, v in ngram_counts_3.items() if v > 1})
    my_count.update({k: v for k, v in ngram_counts_4.items() if v > 1})

    return my_count


if __name__ == "__main__":
    my_count = make_my_count()
    data = pd.concat([pd.read_csv(i) for i in glob("ensemble_csv/*.csv")])

    result = defaultdict(list)
    for img_name in tqdm(data.img_name.unique()):
        result["img_name"] += [img_name]
        comments = data[data.img_name == img_name].comments
        weight = [make_weight(x, my_count) for x in comments]
        result["comments"] += [max(zip(comments, weight), key=lambda x: x[1])[0]]
    pd.DataFrame(result).to_csv("ensemble.csv", index=False)
