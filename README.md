# folder2lmdbs

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
python folder2lmdbs.py --src <images> --dest <folder-for-LMDB> --caffe <where-your-caffe-is> --resize <size> [--min <category-size>] [--split <ratio>] [--png]
```

Hypothetical example on CASIA dataset:

```
python folder2lmdbs.py --src /data/casia --dest /data/casia-db --caffe /usr/local/caffe --resize 144 --min 20 --png
```





