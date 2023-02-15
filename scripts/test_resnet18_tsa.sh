################### Test URL Model with Task-specific Adapters (TSA) ###################
# url + residual adapters (matrix, initialized as identity matrices) + pre-classifier alignment 
# CUDA_VISIBLE_DEVICES=0 python test_extractor_tsa.py --model.pretrained --model.name=imagenet-net --model.dir ./saved_results/sdl \
# --test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode sdl

CUDA_VISIBLE_DEVICES=0 python test_extractor_tsa.py --model.name=imagenet-net --model.dir ./saved_results/sdl --model.pretrained --source ./saved_results/sdl \
--test.tsa-ad-type residual --test.tsa-ad-form matrix --test.tsa-opt alpha+beta --test.tsa-init eye --test.mode sdl --data.test omniglot --out.method tsa --test.size 600