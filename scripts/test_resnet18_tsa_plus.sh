################### Test URL Model with Task-specific Adapters (TSA) ###################
# url + residual adapters (matrix, initialized as identity matrices) + pre-classifier alignment 
# CUDA_VISIBLE_DEVICES=0 python test_extractor_tsa.py --model.pretrained --model.name=imagenet-net --model.dir ./saved_results/sdl \
# --test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode sdl\
# set1: cifar10 dtd mscoco cifar100 cu_birds aircraft ilsvrc_2012 -3~4%
# set2: omniglot quickdraw fungi vgg_flower traffic_sign mnist + 2~6%



CUDA_VISIBLE_DEVICES=7  python test_extractor_tsa_plus.py --model.name=imagenet-net --model.dir ./saved_results/sdl --model.pretrained --source ./saved_results/sdl \
--test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode sdl --data.test cifar10 dtd mscoco cifar100 cu_birds aircraft ilsvrc_2012  --out.method ensemble2 --test.size 200

# 0: max 10 
# 1: loss eff bias 0.75 intra 2.0
# 2: 0.4 eff
# 3: scale intra

# distillation from low to high - 0.8: 0.2 omniglot 83%
# distillation from serial to res - 0.9: 0.1 cifar 78.8% - lr2 0.5
# distillation from serial to res - 0.9: 0.1 cifar 80.4% - lrw 0.1


# best config - tsa distill 0.2 lr: 0.1, lr_beta: 1.0, lrw: 0.01 - tsa eye init new randn init 

# serial + res - super fast adapt

# dtd - 0.5 가 best, cifar은 0.35- 0.2 omni quick은 0.1? mscoco 0.35 +  lr 0.1
# omni - 0.4-0.1 best, quick도..?
# dtd 0.3-0.4 best 인듯?