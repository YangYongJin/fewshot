################### Test URL Model with Task-specific Adapters (TSA) ###################
# url + residual adapters (matrix, initialized as identity matrices) + pre-classifier alignment 
# CUDA_VISIBLE_DEVICES=0 python test_extractor_tsa.py --model.pretrained --model.name=imagenet-net --model.dir ./saved_results/sdl \
# --test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode sdl\
# set1: cifar10 mscoco dtd cifar100 cu_birds aircraft ilsvrc_2012 -3~4%
# set2: omniglot quickdraw fungi vgg_flower traffic_sign mnist + 2~6%


CUDA_VISIBLE_DEVICES=6   python test_extractor_tsa_plus.py --model.name=imagenet-net --model.dir ./saved_results/sdl --model.pretrained --source ./saved_results/sdl \
--test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode sdl --data.test omniglot fungi quickdraw vgg_flower traffic_sign mnist --out.method ensemble2 --test.size 200

# 0: bias 0.5 loss **2 eff 0.5
# 1: 0.75
# 2: 0.75
# 3: 0.5 eff loss 
# distillation from low to high - 0.8: 0.2 omniglot 83%
# distillation from serial to res - 0.9: 0.1 cifar 78.8% - lr2 0.5
# distillation from serial to res - 0.9: 0.1 cifar 80.4% - lrw 0.1


# serial + res - super fast adapt