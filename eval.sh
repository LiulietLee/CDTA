models=('resnet34' 'inception_v3' 'densenet161' 'vgg16_bn')
datasets=('birds-400' 'food-101' 'comic books' 'oxford 102 flower')

for ((j = 0; j < ${#datasets[@]}; j++)) do
    for ((i = 0; i < ${#models[@]}; i++)) do
        python eval_cdta.py \
        -d ${datasets[$j]} \
        -a ${models[$i]} \
        --pretrained './pretrained/surrogate/simsiam_bs256_100ep_cst.tar' \
        --eps 0.06274509803921569 \
        --nb-iter 30 \
        --step-size 0.01568627450980392
    done
done
