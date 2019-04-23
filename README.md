## MNA
> Citation: Kong, X., Zhang, J., & Yu, P. S. (2013). Inferring anchor links across multiple heterogeneous social networks. In Proceedings of the 22nd ACM international conference on Conference on information & knowledge management - CIKM ’13 (pp. 179–188).
```shell
python alp_main.py --graph1 data/test.src.net --graph2 data/test.obj.net --identity-linkage data/test.align --output test.res --method mna

python alp_main.py --graph1 ../../../data/alp-datasets/online-offline/online.adjlist --graph2 ../../../data/alp-datasets/online-offline/offline.adjlist --identity-linkage ../../half/data/online-offline/train-test/online-offline.douban.anchors.ptrain10.train --output online-offline.mna.p10 --method mna
```

## FINAL
> Citation: Zhang, S. (2016). FINAL: Fast Attributed Network Alignment. In Proceedings of the 22th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining - KDD '16 (pp. 421–434).
```shell
python alp_main.py --method final --graph1 data/test.src.net --graph2 data/test.obj.net --graph-size1 20 --graph-size2 20 --identity-linkage data/test.align --output test
```

## FRUI-P
> Citation: Zhou, X., Liang, X., Member, S., Du, X., & Zhao, J. (2018). Structure Based User Identification across Social Networks. IEEE Transactions on Knowledge and Data Engineering, 30(6), 1178–1191.
+ Step 1: network embedding (FFVM)
```shell
python ne_main.py --input data/test.src.net --output ffvm.res --batch-size 6 --table-size 100 --rep-size 4 --method ffvmx --neg-ratio 2 --order 1
```

+ Step 2: matching (FRUI-P)
```shell
python alp_main.py --embedding1 test.res.epoch5.node_order1 --embedding2 ffvm.res.epoch5.node --graph1 data/test.src.net --graph2 data/test.obj.net --identity-linkage data/test.align --output test.fruip --epochs 10 --method fruip
```

## PALE
> Citation: Man, T., Shen, H., Liu, S., Jin, X., & Cheng, X.. Predict anchor links across social networks via an embedding approach. IJCAI International Joint Conference on Artificial Intelligence, 2016, 1823–1829.
+ Step 1: network embedding (LINE)
```shell
python ne_main.py --input data/test.src.net --output test.res --batch-size 6 --table-size 100 --rep-size 4 --method line --neg-ratio 2 --order 1
```

+ Step 2: matching (PALE)
```shell
python alp_main.py --embedding1 test.res.epoch5.node_order1 --embedding2 ffvm.res.epoch5.node --type-model lin --identity-linkage data/test.align --output test.alp --batch-size 4 --input-size 4 --epochs 10 --method pale --neg-ratio 5 --device :/cpu:0

python alp_main.py --embedding1 source.epoch100.node_order1 --embedding2 target.epoch100.node_order1 --type-model mlp --identity-linkage ../../half/data/online-offline/train-test/online-offline.douban.anchors.ptrain10.train --output online-offline.pale.p10 --batch-size 128 --input-size 16 --hidden-size 8 --layers 2 --epochs 10000 --method pale --neg-ratio 15 --device :/cpu:0 --saving-step 10 --is-valid True --early-stop True
```

```shell
python eval_pale.py -feat-src source.epoch100.node_order1 -feat-end target.epoch100.node_order1 -linkage ../../half/data/online-offline/train-test/online-offline.douban.anchors.ptrain10.test -model online-offline.pale.p10 -model-type mlp -n-layer 2 -n-cands 19 -output mrr.pale.p10
```

## IONE

> Citation:
```shell
java -jar ione/IONE.jar --iters 1000000 --dims 8 --root-directory ../../gcn_for_alp/data/node50/ --anchor-file graph.anchors.labels.0.7.train --graph-x-file graph.src.s_0.3.c_0.8 --graph-x-output-file node50.ione.src.emb --graph-y-file graph.obj.s_0.3.c_0.8 --graph-y-output-file node50.ione.obj.emb
```