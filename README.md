## Baselines for Anchor Link Prediction (Network Alignment)

This project contains baselines, including MNA, PALE, IONE, FINAL, FRUI-P, CROSSMNA and REGAL.

### MNA

+ Citation: ``Kong, X., Zhang, J., & Yu, P. S. (2013). Inferring anchor links across multiple heterogeneous social networks. In Proceedings of the 22nd ACM international conference on Conference on information & knowledge management - CIKM ’13 (pp. 179–188).``

+ Implementation

```shell
python alp_main.py --method mna --graphs data/test.src.net data/test.obj.net --identity-linkage data/test.align --use-net True --neg-ratio 5 --output test.res
```

+ Evaluation

```
python eval_mna.py -net-src data/test.src.net -net-end data/test.obj.net -feat-src test.res.epoch5.node_order1 -feat-end ffvm.res.epoch5.node -linkage data/test.align -use-net true -model test.res.pkl -n-cands 1 -eval-type cls -output eval.mna
```

### FINAL

+ Citation: ``Zhang, S. (2016). FINAL: Fast Attributed Network Alignment. In Proceedings of the 22th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining - KDD '16 (pp. 421–434).``

+ Implementation

```shell
python alp_main.py --method final --graphs data/test.src.net data/test.obj.net --graph-sizes 20 20 --identity-linkage data/test.align --output test --alpha 0.1 --tol 0.1
```

### FRUI-P
+ Citation: ``Zhou, X., Liang, X., Member, S., Du, X., & Zhao, J. (2018). Structure Based User Identification across Social Networks. IEEE Transactions on Knowledge and Data Engineering - TKDE, 30(6), 1178–1191.``

+ Implementation

> Step 1: network embedding (FFVM)
```shell
python ne_main.py --input data/test.src.net --output ffvm.res --batch-size 6 --table-size 100 --rep-size 4 --method ffvm --neg-ratio 2 --order 1
```

### 

> Step 2: matching (FRUI-P)
```shell
python alp_main.py --method fruip --embeddings test.res.epoch5.node_order1 ffvm.res.epoch5.node --graphs data/test.src.net data/test.obj.net --identity-linkage data/test.align --output test.fruip --epochs 10
```

### PALE

+ Citation: ``Man, T., Shen, H., Liu, S., Jin, X., & Cheng, X. (2016). Predict anchor links across social networks via an embedding approach. International Joint Conference on Artificial Intelligence - IJCAI (1823–1829).``

+ Implementation

> Step 1: network embedding (LINE)

```shell
python ne_main.py --input data/test.src.net --output test.res --batch-size 6 --table-size 100 --rep-size 4 --method line --neg-ratio 2 --order 1
```

> Step 2: matching (PALE)

```shell
python alp_main.py --embeddings test.res.epoch5.node_order1 ffvm.res.epoch5.node --type-model lin --identity-linkage data/test.align --output test.alp --batch-size 4 --input-size 4 --epochs 10 --method pale --neg-ratio 5 --device :/cpu:0
```

+ Evaluation

```shell
python eval_pale.py -feat-src test.res.epoch5.node_order1 -feat-end ffvm.res.epoch5.node -linkage data/test.align -model test.alp -model-type lin -eval-type cls -n-layer 2 -n-cands 1 -output eval.pale
```

### IONE

+ Citation: ``Liu, L., Cheung, W. K., Li, X., & Liao, L. (2016). Aligning Users Across Social Networks Using Network Embedding. International Joint Conference on Artificial Intelligence - IJCAI (pp. 1774–1780).``

+ Implementation

See [Author's implementation][https://github.com/ColaLL/IONE]

+ Evaluation

```shell
python eval_ione.py -feat-src xxx -feat-end xxx -linkage data/test.align -eval-type cls -n-cands 1 -output eval.ione
```

### CROSSMNA

+ Citation: ``Chu, X., Fan, X., Yao, D., Zhu, Z., Huang, J., & Bi, J. (2019). Cross-Network Embedding for Multi-Network Alignment. The World Wide Web Conference - WWW (pp. 273-284).``

+ Implementation

```shell
python alp_main.py --graphs data/test.src.net data/test.obj.net --identity-linkage data/test.align --log-file test.log --batch-size 4 --epochs 10 --table-size 1000 --method crossmna --lr .01 --nd-rep-size 4 --layer-rep-size 8 --neg-ratio 5 --output test.crossmna.out
```

+ Evaluation

```shell
python eval_crossmna.py -node test.crossmna.out -train-linkage data/test.align -linkage data/test.align -n-cands 1 -eval-type cls -output eval.crossmna
```

### REGAL

+ Citation: ``Heimann, M., Shen, H., Safavi, T., & Koutra, D. (2018, October). Regal: Representation learning-based graph alignment. Proceedings of the 27th ACM International Conference on Information and Knowledge Management - CIKM (pp. 117-126)``

+ Implementation

```shell
python regal/regal.py --input data/test.src.net data/test.obj.net --output test.regal.out
```

+ Evaluation

```shell
python eval_regal.py -node test.regal.out -linkage data/test.align -n-cands 1 -eval-type cls -output eval.regal
```

---

### Notes (Updated 04/23/2020)

Update shell commands for evaluations

Here we provide metrics on ranking and classification issues. For classification issue, we calculate the evaluation results as follows,

```
src2end
Overall: 17
(1,1),0.439630,1
(1,7),5.488685,0
(2,2),1.018289,1
(2,4),1.463131,0
(3,3),0.769002,1
(3,15),2.228442,0
(4,4),2.119972,1
(4,7),5.188379,0
(5,5),1.434842,1
(5,19),0.474886,0
(6,6),2.483092,1
...
```

The first line shows the order of calculation, e.g., 'src2end'. The network who puts back refers that we will sample nodes in that network to compose negative sample in evaluation (We will take example later). The second line shows the overall tasks in the evaluation task. Then we start to record the results of the choosen metric. In classification, each line (from 3rd line to the end) has three columns separated by comma ','. It outputs the decision whether the pair of nodes (e.g.,(1,1)) is anchor node, and the decision score is listed in the second column. The last column records the ground truth provided by test dataset. As we only observe anchor links across networks in test dataset, we need to supplement negative samples. The negative samples are generated by sampling from the target network (e.g., the network listed back in first line). Then you can use any classification metrics by the results, e.g., AUC, F1 score, Accuracy and Recall (recommended scikit learn package in python to caluculate these metrics).

For ranking issue, we calcuate the evaluation results as follows,

```
src2end
Overall: 17
(1,1):1.000000
(2,2):0.500000
(3,3):0.500000
(4,4):0.250000
(5,5):0.250000
(6,6):0.200000
(7,7):0.166667
(8,8):0.250000
(9,9):0.200000
...
mean_mrr:0.41960784313725485, var:0.08660899653979237
```

The first and second lines have the same meaning as mentioned above. From the 3rd line to the end, each line has two columns seperated by colon ':', refering the MRR value (1.000000) corresponded to the pair of nodes (e.g., (1,1)). At the last line, we provide statistic results on mean MRR and its variance (calculated by np.mean and np.var).