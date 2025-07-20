```shell
anomalib train --config examples/configs/model/efficient_ad.yaml
```

```shell
# add NUM_CLASS
python timm_train.py --data-dir datasets/imagenette/imagenette2 -b 64 --model wide_resnet50_2 --sched cosine --epochs 200 --lr 0.05 --amp --remode pixel --reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --split-bn --jsd --dist-bn reduce

```