# CDTA
> Release for CDTA: Cross-Domain Transfer-Based Attack with Contrastive Learning [AAAI23]

## Download

### Datasets

Download [birds-400](https://github.com/LiulietLee/CDTA/releases/download/v1.1/birds-400.zip), [food-101](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz), [comic books](https://www.kaggle.com/datasets/cenkbircanoglu/comic-books-classification), and [oxford 102 flower](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) datasets. Extract them to the `./dataset` directory.

### Pre-trained target classifiers

Download target classifiers trained on [birds-400](https://github.com/LiulietLee/CDTA/releases/download/v1.0/birds-400.zip), [food-101](https://github.com/LiulietLee/CDTA/releases/download/v1.0/food-101.zip), [comic books](https://github.com/LiulietLee/CDTA/releases/download/v1.0/Comic.Books.zip), and [oxford 102 flower](https://github.com/LiulietLee/CDTA/releases/download/v1.0/Oxford.102.Flower.zip). Extract them to the `./pretrained/target` directory.

### Pre-trained feature extractor

Download the [per-trained feature extractor](https://github.com/LiulietLee/CDTA/releases/download/v1.0/simsiam_bs256_100ep_cst.tar) and put the tar file in the `./pretrained/surrogate` directory.

## Evaluation

```
python eval_cdta.py \
  -d '[dataset]' \
  -a '[target classifier]' \
  --pretrained './pretrained/surrogate/simsiam_bs256_100ep_cst.tar' \
  --eps 0.06274509803921569 \
  --nb-iter 30 \
  --step-size 0.01568627450980392
```

- `[dataset]` can be `birds-400`, `food-101`, `comic books`, or `oxford 102 flower`. 
- `[target classifier]` can be `resnet34`, `densenet161`, `inception_v3`, or `vgg16_bn`.

Or use `eval.sh` to test all target models.

```
bash ./eval.sh
```

## Train feature extractor

```
cd cst
```

```
python main_simsiam.py \
  -a resnet50 \
  -b 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr \
  '[ImageNet path]'
```