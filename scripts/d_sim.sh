


CUDA_VISIBLE_DEVICES=7 python domain_sim.py --model.name=imagenet-net --model.dir ./saved_results/sdl --model.pretrained --source ./saved_results/sdl \
--test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode sdl --data.train aircraft ilsvrc_2012 omniglot fungi quickdraw vgg_flower traffic_sign mnist mscoco cifar10 dtd cifar100 cu_birds --data.test cifar10 mnist

