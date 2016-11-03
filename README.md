# folder2lmdbs.py

### Script generates two LMDBs from image folder like this

```
/
  /cat
     image_1.jpg
     image_2.png
  /dog
     image_1.jpg
     image_2.png
```

### Bonus. It also calculates mean.binaryproto

Usage:

```
python3 folder2lmdbs.py --src <images> --dst <folder-for-LMDB> --caffe <where-your-caffe-is> --resize <size> [--min <category-size>] [--split <ratio>] [--png]
```

Hypothetical example on CASIA dataset:

```
python3 folder2lmdbs.py --src /data/casia --dst /data/casia-db --caffe /usr/local/caffe --resize 144 --min 20 --png
```

# balancer.py

### Script creates balanced and shrinked version of dataset.

Usage:

```
python3 balancer.py --src <dataset> --top <number-of-categories-to-keep> --dst <balanced-dataset> [--size <category-sizes>]
```

Hypothetical example on CASIA dataset:

```
python3  balancer.py --src /data/casia --top 10000 --dst /data/casia_balanced --size 100




