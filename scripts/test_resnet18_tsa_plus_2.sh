################### Test URL Model with Task-specific Adapters (TSA) ###################
# url + residual adapters (matrix, initialized as identity matrices) + pre-classifier alignment 
# CUDA_VISIBLE_DEVICES=0 python test_extractor_tsa.py --model.pretrained --model.name=imagenet-net --model.dir ./saved_results/sdl \
# --test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode sdl

CUDA_VISIBLE_DEVICES=3 python test_extractor_tsa_plus.py --model.name=url --model.dir ./saved_results/url \
--test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode mdl --data.test cifar10 quickdraw mscoco traffic_sign fungi --out.method url --test.size 200

# CUDA_VISIBLE_DEVICES=2  python test_extractor_tsa_plus.py --model.name=imagenet-net --model.dir ./saved_results/sdl --model.pretrained --source ./saved_results/sdl \
# --test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode sdl --data.test omniglot quickdraw fungi vgg_flower traffic_sign mnist --out.method ensemble2 --test.size 200