## Baselines for Anchor Link Prediction (Network Alignment)

This project contains baselines, including MNA, PALE, IONE, FINAL, FRUI-P, CROSSMNA and REGAL.

### MNA

+ Citation: Kong, X., Zhang, J., & Yu, P. S. (2013). Inferring anchor links across multiple heterogeneous social networks. In Proceedings of the 22nd ACM international conference on Conference on information & knowledge management - CIKM ’13 (pp. 179–188).

+ Implementation

```shell
python alp_main.py --method mna --graphs data/test.src.net data/test.obj.net --identity-linkage data/test.align --use-net True --neg-ratio 5 --output test.res
```

### FINAL

+ Citation: Zhang, S. (2016). FINAL: Fast Attributed Network Alignment. In Proceedings of the 22th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining - KDD '16 (pp. 421–434).

+ Implementation

```shell
python alp_main.py --method final --graphs data/test.src.net data/test.obj.net --graph-sizes 20 20 --identity-linkage data/test.align --output test --alpha 0.1 --tol 0.1
```

### FRUI-P
+ Citation: Zhou, X., Liang, X., Member, S., Du, X., & Zhao, J. (2018). Structure Based User Identification across Social Networks. IEEE Transactions on Knowledge and Data Engineering - TKDE, 30(6), 1178–1191.

+ Implementation

> Step 1: network embedding (FFVM)
```shell
python ne_main.py --input data/test.src.net --output ffvm.res --batch-size 6 --table-size 100 --rep-size 4 --method ffvmx --neg-ratio 2 --order 1
```

### 

> Step 2: matching (FRUI-P)
```shell
python alp_main.py --method fruip --embeddings test.res.epoch5.node_order1 ffvm.res.epoch5.node --graphs data/test.src.net data/test.obj.net --identity-linkage data/test.align --output test.fruip --epochs 10
```

### PALE

+ Citation: Man, T., Shen, H., Liu, S., Jin, X., & Cheng, X. (2016). Predict anchor links across social networks via an embedding approach. International Joint Conference on Artificial Intelligence - IJCAI (1823–1829).

+ Implementation

> Step 1: network embedding (LINE)

```shell
python ne_main.py --input data/test.src.net --output test.res --batch-size 6 --table-size 100 --rep-size 4 --method line --neg-ratio 2 --order 1
```

> Step 2: matching (PALE)

```shell
python alp_main.py --embeddings test.res.epoch5.node_order1 ffvm.res.epoch5.node --type-model lin --identity-linkage data/test.align --output test.alp --batch-size 4 --input-size 4 --epochs 10 --method pale --neg-ratio 5 --device :/cpu:0
```

### IONE

+ Citation: Liu, L., Cheung, W. K., Li, X., & Liao, L. (2016). Aligning Users Across Social Networks Using Network Embedding. International Joint Conference on Artificial Intelligence - IJCAI (pp. 1774–1780). 

+ Implementation

See [Author's implementation][https://github.com/ColaLL/IONE]

### CROSSMNA

+ Citation: Chu, X., Fan, X., Yao, D., Zhu, Z., Huang, J., & Bi, J. (2019). Cross-Network Embedding for Multi-Network Alignment. The World Wide Web Conference - WWW (pp. 273-284).

+ Implementation

```shell
python alp_main.py --graphs data/test.src.net data/test.obj.net --identity-linkage data/test.align --log-file test.log --batch-size 4 --epochs 10 --table-size 1000 --method crossmna --lr .01 --nd-rep-size 4 --layer-rep-size 8 --neg-ratio 5 --output test.crossmna.out
```

### REGAL

+ Citation: Heimann, M., Shen, H., Safavi, T., & Koutra, D. (2018, October). Regal: Representation learning-based graph alignment. Proceedings of the 27th ACM International Conference on Information and Knowledge Management - CIKM (pp. 117-126)

+ Implementation

```shell
python regal/regal.py --input data/test.src.net data/test.obj.net --output test.regal.out
```
