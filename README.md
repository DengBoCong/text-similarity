<h1 align="center">Text-Similarity</h1>

<div align="center">

[![Blog](https://img.shields.io/badge/blog-@DengBoCong-blue.svg?style=social)](https://www.zhihu.com/people/dengbocong)
[![Paper Support](https://img.shields.io/badge/paper-repo-blue.svg?style=social)](https://github.com/DengBoCong/nlp-paper)
![Stars Thanks](https://img.shields.io/badge/Stars-thanks-brightgreen.svg?style=social&logo=trustpilot)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=social&logo=appveyor)

[comment]: <> ([![PRs Welcome]&#40;https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square&#41;]&#40;&#41;)

</div>

# Overview
+ **Dataset**: 中文/English 语料, ☞  [点这里](https://github.com/DengBoCong/text-similarity/tree/main/corpus)
+ **Paper**: 相关论文详解, ☞  [点这里](https://github.com/DengBoCong/nlp-paper)
+ **The implemented method is as follows:**：
   + TF-IDF
   + BM25
   + LSH
   + SIF/uSIF
   + FastText
   + RNN Base (Siamese RNN, Stack RNN)
   + CNN Base (Fast Text, Text CNN, Char CNN, VDCNN)
   + Bert Base
   + Albert
   + NEZHA
   + RoBERTa
   + SimCSE
   + Poly-Encoder
   + ColBERT
   + RE2（Simple-Effective-Text-Matching）

# Usages
可以选择通过pip进行安装并使用（如下），或者直接下载源码到本地，集成到项目中：
```
pip3 install text-sim
```

```
1：examples目录下有不同模型对应的 preprocess/train/evalute代码，可自行修改
2：如下示例从examples中引入actuator方法，准备好对应的模型配置文件即可执行
3：examples目录下的inference.py为训练好的模型推理代码
4：主体代码放在sim下，TensorFlow和Pytorch两个版本分开存放，引用方式基本保持一致
5：相关工具包括word2vec、tokenizer、data_format统一放在sim的tools下
```

### TF-IDF

```python
# Example
# Sklearn version
from examples.run_tfidf_sklearn import actuator
actuator("./corpus/chinese/breeno/train.tsv", query1="12 23 4160 276", query2="29 23 169 1495")

# Custom version
from examples.run_tfidf import actuator
actuator("./corpus/chinese/breeno/train.tsv", query1="12 23 4160 276", query2="29 23 169 1495")

# 工具调用
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
# Example
from examples.run_bm25 import actuator
actuator("./corpus/chinese/breeno/train.tsv", query1="12 23 4160 276", query2="29 23 169 1495")

# 工具调用
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
+ [A Simple But Tough-To-Beat Baseline For Sentence Embeddings](https://openreview.net/pdf?id=SyK00v5xx)
+ [Unsupervised Random Walk Sentence Embeddings: A Strong but Simple Baseline](https://aclanthology.org/W18-3012.pdf)
```python
sentences = [["token1", "token2", "..."], ...]
vector = [[[1, 1, 1], [2, 2, 2], [...]], ...]
from sim.sif_usif import SIF
from sim.sif_usif import uSIF

sif = SIF(n_components=5, component_type="svd")
sif.fit(tokens_list=sentences, vector_list=vector)

usif = uSIF(n_components=5, n=1, component_type="svd")
usif.fit(tokens_list=sentences, vector_list=vector)
```

### FastText
+ [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)
```python
# TensorFlow version
from examples.tensorflow.run_fast_text import actuator
actuator(execute_type="train", model_type="bert", model_dir="./data/chinese_wwm_L-12_H-768_A-12")

# Pytorch version
from examples.pytorch.run_fast_text import actuator
actuator(execute_type="train", model_type="bert", model_dir="./data/chinese_wwm_pytorch")
```

### RNN Base
+ [Siamese Recurrent Architectures for Learning Sentence Similarity](https://scholar.google.com/scholar_url?url=https://ojs.aaai.org/index.php/AAAI/article/view/10350/10209&hl=zh-CN&sa=T&oi=gsb-gga&ct=res&cd=0&d=7393466935379636447&ei=KQWzYNL5OYz4yATXqJ6YCg&scisig=AAGBfm0zNEZZez8zh5ZB_iG7UTrwXmhJWg)
+ [Learning Text Similarity with Siamese Recurrent Networks](https://aclanthology.org/W16-1617.pdf)
```python
# TensorFlow version
from examples.tensorflow.run_siamese_rnn import actuator
actuator("./data/config/siamse_rnn.json", execute_type="train")

# Pytorch version
from examples.pytorch.run_siamese_rnn import actuator
actuator("./data/config/siamse_rnn.json", execute_type="train")
```

### CNN Base
+ [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
+ [Character-Aware Neural Language Models](https://arxiv.org/pdf/1508.06615.pdf)
+ [Highway Networks](https://arxiv.org/pdf/1505.00387.pdf)
+ [Very Deep Convolutional Networks for Text Classification](https://arxiv.org/pdf/1606.01781.pdf)
```python
# TensorFlow version
from examples.tensorflow.run_cnn_base import actuator
actuator(execute_type="train", model_type="bert", model_dir="./data/chinese_wwm_L-12_H-768_A-12")

# Pytorch version
from examples.pytorch.run_cnn_base import actuator
actuator(execute_type="train", model_type="bert", model_dir="./data/chinese_wwm_pytorch")
```

### Bert Base
+ [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
```python
# TensorFlow version
from examples.tensorflow.run_basic_bert import actuator
actuator(model_dir="./data/chinese_wwm_L-12_H-768_A-12", execute_type="train")

# Pytorch version
from examples.pytorch.run_basic_bert import actuator
actuator(model_dir="./data/chinese_wwm_pytorch", execute_type="train")
```

### Albert
+ [ALBERT: A Lite BERT For Self-superpised Learning Of Language Representations](https://arxiv.org/pdf/1909.11942.pdf)
```python
# TensorFlow version
from examples.tensorflow.run_albert import actuator
actuator(model_dir="./data/albert_small_zh_google", execute_type="train")

# Pytorch version
from examples.pytorch.run_albert import actuator
actuator(model_dir="./data/albert_chinese_small", execute_type="train")
```

### NEZHA
+ [NEZHA: Neural Contextualized Representation For Chinese Language Understanding](https://arxiv.org/pdf/1909.00204.pdf)
```python
# TensorFlow version
from examples.tensorflow.run_nezha import actuator
actuator(model_dir="./data/NEZHA-Base-WWM", execute_type="train")

# Pytorch version
from examples.pytorch.run_nezha import actuator
actuator(model_dir="./data/nezha-base-wwm", execute_type="train")
```

### RoBERTa
+ [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)
```python
# TensorFlow version
from examples.tensorflow.run_basic_bert import actuator
actuator(model_dir="./data/chinese_roberta_L-6_H-384_A-12", execute_type="train")

# Pytorch version
from examples.pytorch.run_basic_bert import actuator
actuator(model_dir="./data/chinese-roberta-wwm-ext", execute_type="train")
```

### SimCSE
+ [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/pdf/2104.08821.pdf)
```python
# TensorFlow version
from examples.tensorflow.run_simcse import actuator
actuator(model_dir="./data/chinese_wwm_L-12_H-768_A-12", execute_type="train", model_type="bert")

# Pytorch version
from examples.pytorch.run_simcse import actuator
actuator(model_dir="./data/chinese_wwm_pytorch", execute_type="train", model_type="bert")
```

### Poly-Encoder
+ [Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring](https://arxiv.org/pdf/1905.01969v2.pdf)
```python
# TensorFlow version
from examples.tensorflow.run_poly_encoder import actuator
actuator(model_dir="./data/chinese_wwm_L-12_H-768_A-12", execute_type="train", model_type="bert")

# Pytorch version
from examples.pytorch.run_poly_encoder import actuator
actuator(model_dir="./data/chinese_wwm_pytorch", execute_type="train", model_type="bert")
```

### ColBERT
+ [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/pdf/2004.12832.pdf)
```python
# TensorFlow version
from examples.tensorflow.run_colbert import actuator
actuator(model_dir="./data/chinese_wwm_L-12_H-768_A-12", execute_type="train", model_type="bert")

# Pytorch version
from examples.pytorch.run_colbert import actuator
actuator(model_dir="./data/chinese_wwm_pytorch", execute_type="train", model_type="bert")
```

### RE2
+ [Simple and Effective Text Matching with Richer Alignment Features](https://arxiv.org/pdf/1908.00300.pdf)
```python
# TensorFlow version
from examples.tensorflow.run_re2 import actuator
actuator("./data/config/re2.json", execute_type="train")

# Pytorch version
from examples.pytorch.run_re2 import actuator
actuator("./data/config/re2.json", execute_type="train")
```


# Cite
```
@misc{text-similarity,
  title={text-similarity},
  author={Bocong Deng},
  year={2021},
  howpublished={\url{https://github.com/DengBoCong/text-similarity}},
}
```

# Reference
+ [bert4keras](https://github.com/bojone/bert4keras/)
+ [albert_zh](https://github.com/brightmart/albert_zh)
+ [HuggingFace](https://huggingface.co/)
+ [Self-Attention with Relative Position Representations](https://arxiv.org/pdf/1803.02155.pdf)
