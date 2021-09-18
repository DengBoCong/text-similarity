<h1 align="center">Text-Similarity</h1>

<div align="center">

[![Blog](https://img.shields.io/badge/blog-@DengBoCong-blue.svg?style=social)](https://www.zhihu.com/people/dengbocong)
[![Paper Support](https://img.shields.io/badge/paper-repo-blue.svg?style=social)](https://github.com/DengBoCong/nlp-paper)
![Stars Thanks](https://img.shields.io/badge/Stars-thanks-brightgreen.svg?style=social&logo=trustpilot)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=social&logo=appveyor)

[comment]: <> ([![PRs Welcome]&#40;https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square&#41;]&#40;&#41;)

</div>

# Overview

本工具对每种方法都独立成一个文件，方便阅读代码，每个方法参数和关键代码处都进行了详细的注释，方便方法的使用！

+ TF-IDF
+ BM25
+ LSH
+ SIF

# Usages

### TF-IDF

```python
from sim.tf_idf import TFIdf

tokens_list = ["这是 一个 什么 样 的 工具", "..."]
query = ["非常 好用 的 工具"]

tf_idf = TFIdf(tokens_list, split=" ")
print(tf_idf.get_score(query, 0))  # score
print(tf_idf.get_score_list(query, 10))  # [(index, score), ...]
print(tf_idf.weight())  # list or numpy array
```

### BM25

```python
from sim.bm25 import BM25

tokens_list = ["这是 一个 什么 样 的 工具", "..."]
query = ["非常 好用 的 工具"]

bm25 = BM25(tokens_list, split=" ")
print(bm25.get_score(query, 0))  # score
print(bm25.get_score_list(query, 10))  # [(index, score), ...]
print(bm25.weight())  # list or numpy array
```

### LSH

```python
from sim.lsh import E2LSH
from sim.lsh import MinHash

e2lsh = E2LSH()
min_hash = MinHash()

candidates = [[3.6216, 8.6661, -2.8073, -0.44699, 0], ...]
query = [-2.7769, -5.6967, 5.9179, 0.37671, 1]
print(e2lsh.search(candidates, query))  # index in candidates
print(min_hash.search(candidates, query))  # index in candidates
```

### SIF