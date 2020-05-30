# export CUDA_VISIBLE_DEVICES=0,1,2,3
python -W ignore -m run_apis.validation \
            --load_path 'structures/NAE-K3-S-1' --report_freq 100 \
            --data_path '/home/csgrad/xuangong/data/imagenet/' --save './val'
            # --data_path '/data/ImageNet/' --save './val'
            # --data_path /home/csgrad/xuangong/data/imagenet/ --save ../dense_weight/