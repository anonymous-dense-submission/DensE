# DensE: An Enhanced Non-Abelian Group Representation for Knowledge Graph Embedding

This repository is the official pytorch implementation of DensE.

Capturing the composition patterns of relations is a vital task in knowledge graph completion. It also serves as a fundamental step towards multi-hop reasoning over learned knowledge. Previously, rotation-based translational methods, e.g., RotatE, have been developed to model composite relations using the product of a series of complex-valued diagonal matrices. However, RotatE makes several oversimplified assumptions on the composition patterns, forcing the relations to be commutative, independent from entities and fixed in scale. To tackle this problem, we have developed a novel knowledge graph embedding method, named DensE, to provide sufficient modeling capacity for complex composition patterns. In particular, our method decomposes each relation into an SO(3) group-based rotation operator and a scaling operator in the three dimensional (3-D) Euclidean space. The advantages of our method are twofold: (1) For composite relations, the corresponding diagonal relation matrices can be non-commutative and related with entity embeddings; (2) It extends the concept of RotatE to a more expressive setting with lower model complexity and preserves the direct geometrical interpretations, which reveals how relations with distinct patterns (i.e., symmetry/anti-symmetry, inversion and composition) are modeled. Experimental results on multiple benchmark knowledge graphs show that DensE outperforms the current state-of-the-art models for missing link prediction, especially on composite relations.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
wn18: bash run_DensE.sh train DensE wn18 <gpu-id> <dir> 512 1024 200 12.0 0.3 0.1 80000 8 0 -me -mr -adv
wn18rr: bash run_DensE.sh train DensE wn18rr <gpu-id> <dir> 512 512 300 6.0 0.5 0.1 80000 8 0 -me -mr -adv
FB15k-237: bash run_DensE.sh train DensE FB15k-237 <gpu-id> <dir> 1024 256 800 9.0 1.0 0.1 80000 16 0 -me -mr -adv 
YAGO3-10: bash run_DensE.sh train DensE YAGO3-10 <gpu-id> <dir> 1024 512 150 24.0 1.0 0.1 100000 16 0 -me -mr -adv
```
## Evaluation

To evaluate my models, run:

```eval
wn18: bash run_DensE.sh test DensE wn18 <gpu-id> <dir> 512 1024 200 12.0 0.3 0.1 80000 8 0 -me -mr -adv
wn18rr: bash run_DensE.sh test DensE wn18rr <gpu-id> <dir> 512 512 300 6.0 0.5 0.1 80000 8 0 -me -mr -adv
FB15k-237: bash run_DensE.sh test DensE FB15k-237 <gpu-id> <dir> 1024 256 800 9.0 1.0 0.1 80000 16 0 -me -mr -adv
YAGO3-10: bash run_DensE.sh test DensE YAGO3-10 <gpu-id> <dir> 1024 512 150 24.0 1.0 0.1 100000 16 0 -me -mr -adv
```


## Results

Our model achieves the following performance on:

WN18

| Model name | MR | MRR | H@1 | H@3 | H@10|
| ---------- |----|-----|-----|-----|-----|
|  RotatE    |309 |0.949|0.944|0.952|0.959|
|   QuatE	   |388	|0.949|0.941|0.954|0.960|
|   DensE    |285	|0.950|0.945|0.954|0.959|


WN18RR

| Model name | MR | MRR | H@1 | H@3 | H@10|
| ---------- |----|-----|-----|-----|-----|
|   RotatE   |3340|0.476|0.428|0.492|0.571|
|   QuatE    |3472|0.481|0.436|0.500|0.564|
|   DensE    |2935|0.492|0.445|0.508|0.586|

FB15k-237

| Model name | MR | MRR | H@1 | H@3 | H@10|
| ---------- |----|-----|-----|-----|-----|
|   RotatE	 |177	|0.338|0.241|0.375|0.533|
|   QuatE    |176	|0.311|0.221|0.342|0.495|
|   DensE    |161	|0.351|0.256|0.386|0.544|


YAGO3-10

| Model name | MR | MRR | H@1 | H@3 | H@10|
| ---------- |----|-----|-----|-----|-----|
|  RotatE    |1767|0.495|0.402|0.550|0.670|
|   DensE    |1450|0.542|0.468|0.585|0.678|

## Contributing

This respoisitory is a open source software under MIT lisence. If you'd like to contribute, or have any suggestions for this project, please open an issue on this GitHub repository.

## Acknowledgement 

The evaluation code is implemented based on the open source code of [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
