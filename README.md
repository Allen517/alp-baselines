# MNA
```shell
python alp_main.py --graph1 data/test.src.net --graph2 data/test.obj.net --identity-linkage data/test.align --output test.res --method mna
```

# FINAL
```shell
python alp_main.py --method final --graph1 data/test.src.net --graph2 data/test.obj.net --graph-size1 20 --graph-size2 20 --identity-linkage data/test.align --output test
```

# FRUI-P

+ Step 1: network embedding (FFVM)
```shell
python ne_main.py --input data/test.src.net --output ffvm.res --batch-size 6 --table-size 100 --rep-size 4 --method ffvmx --neg-ratio 2 --order 1
```

+ Step 2: matching (FRUI-P)
```shell
python alp_main.py --embedding1 test.res.epoch5.node_order1 --embedding2 ffvm.res.epoch5.node --graph1 data/test.src.net --graph2 data/test.obj.net --identity-linkage data/test.align --output test.fruip --epochs 10 --method fruip
```

# PALE

+ Step 1: network embedding (LINE)
```shell
python ne_main.py --input data/test.src.net --output test.res --batch-size 6 --table-size 100 --rep-size 4 --method line --neg-ratio 2 --order 1
```

+ Step 2: matching (PALE)
```shell
python alp_main.py --embedding1 test.res.epoch5.node_order1 --embedding2 ffvm.res.epoch5.node --type-model lin --identity-linkage data/test.align --output test.alp --batch-size 4 --input-size 4 --epochs 10 --method pale --neg-ratio 5 --device :/cpu:0
```

