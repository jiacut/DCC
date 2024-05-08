# Train
# nohup python -u end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_cifar10.yml >cifar10.log 2>&1 &
# nohup python -u end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_cifar20.yml >cifar20.log 2>&1 &
# nohup python -u end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_stl10.yml >stl10.log 2>&1 &
# nohup python -u end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_imagenet_dogs.yml >imagenet_dogs.log 2>&1 &
# nohup python -u end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_imagenet10.yml >imagenet10.log 2>&1 &
# nohup python -u end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_tiny_imagenet.yml >tiny_imagenet.log 2>&1 &

# training and test self-labeling
# nohup python -u selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml > cifar10_sl.log 2>&1 &
# nohup python -u selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar20.yml >cifar20_sl.log 2>&1 &
# nohup python -u selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_imagenet10.yml >imagenet10_sl.log 2>&1 &
# nohup python -u selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_stl10.yml > stl10_sl.log 2>&1 &
