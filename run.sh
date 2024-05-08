# Train
#CUDA_VISIBLE_DEVICES=0 python end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_cifar10.yml
#CUDA_VISIBLE_DEVICES=0 python end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_cifar20.yml
#CUDA_VISIBLE_DEVICES=0 python end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_imagenet10.yml
#CUDA_VISIBLE_DEVICES=0 python end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_imagenet_dogs.yml
#CUDA_VISIBLE_DEVICES=0 python end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_tiny_imagenet.yml
#CUDA_VISIBLE_DEVICES=0 python end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_stl10.yml

# nohup python -u end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_cifar10.yml >cifar10.log 2>&1 &
# nohup python -u end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_cifar20.yml >cifar20.log 2>&1 &
# nohup python -u end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_stl10.yml >stl10.log 2>&1 &
# nohup python -u end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_imagenet_dogs.yml >imagenet_dogs.log 2>&1 &
# nohup python -u end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_imagenet10.yml >imagenet10.log 2>&1 &
# nohup python -u end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_tiny_imagenet.yml >tiny_imagenet.log 2>&1 &

# Test
#CUDA_VISIBLE_DEVICES=0 python test_end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_cifar10.yml
#CUDA_VISIBLE_DEVICES=0 python test_end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_cifar20.yml
#CUDA_VISIBLE_DEVICES=0 python test_end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_imagenet10.yml
#CUDA_VISIBLE_DEVICES=0 python test_end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_imagenet_dogs.yml
#CUDA_VISIBLE_DEVICES=0 python test_end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_tiny_imagenet.yml
#CUDA_VISIBLE_DEVICES=0 python test_end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_stl10.yml

# training and test self-labeling
#CUDA_VISIBLE_DEVICES=0 python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar20.yml
#CUDA_VISIBLE_DEVICES=0 python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_stl10.yml

# nohup python -u selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml > cifar10_sl.log 2>&1 &
# nohup python -u selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar20.yml >cifar20_sl.log 2>&1 &
# nohup python -u selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_imagenet10.yml >imagenet10_sl.log 2>&1 &
# nohup python -u selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_stl10.yml > stl10_sl.log 2>&1 &
