# python -m run_apis.retrain --data_path '/media/user/lcl1234/ImageNet/' --load_path 'structures/DenseNAS-A' --save '/home1/weight/chl/dense' --report_freq 10
# python -m run_apis.retrain --data_path '/home1/lcl_work/imagenet' --load_path 'structures/DenseNAS-A' --save '/home1/weight/chl/dense' --report_freq 10

python -W ignore -m run_apis.retrain_lincls_dist \
            --dist_url 'tcp://localhost:10002' --multiprocessing_distributed --world_size 1 --rank 0 \
            --load_path 'structures/ResNet50-K3' --pretrained 'pretrain/resnet50_k3/checkpoint.pth.tar' \
            --report_freq 500 --gpu '0,1' --job_name 'trian_cls' \
            --data_path '/dataset/ImageNet/' --save '/weight/chl/dense'
            # --data_path '/root/data/imagenet' --save '/root/weight/chl/dense'
            # --data_path '/dataset/ImageNet/' --save '/weight/chl/dense'
            # --data_path '/home/dataset/data/ImageNet/' --save '/weight/chl/dense'
            # --data_path '/home/ImageNet/' --save '/weight/chl/dense'
            # --data_path /home/csgrad/xuangong/data/imagenet/ --save ../dense_weight/
            # --data_path '/media/user/lcl1234/ImageNet/' --save '/home1/weight/chl/dense'
            # --data_path '/weight/imagenet' --save '/weight/chl/dense'
            # --data_path '/home1/lcl_work/imagenet' --save '/home1/weight/chl/dense'