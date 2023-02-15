


python domain_sim.py --model.name=imagenet-net --model.dir ./saved_results/sdl --model.pretrained --source ./saved_results/sdl \
--test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode sdl --data.train ilsvrc_2012 --data.test cifar10 mnist

