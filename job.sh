
source /data/zhongz2/anaconda3/bin/activate th24
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
module load gcc/11.3.0


source $FRCE_DATA_ROOT/anaconda3/bin/activate th23
module load cuda/11.8
module load cudnn/8.8.3-cuda11


python -m torch.distributed.launch \
--nproc_per_node=8 \
main_dino.py --arch vit_small \
--data_path /tmp/zhongz2/images/train \
--epochs 300 \
--output_dir ./saving_dir_mnist_300

VERSION=_v1
VERSION=_v2
VERSION=_v3
PARAM_VERSION=_v1  #--global_crops_scale 0.6 1 --local_crops_scale 0.2 0.6
PARAM_VERSION=_v2  #--global_crops_scale 0.5 1 --local_crops_scale 0.2 0.5
python -m torch.distributed.launch \
--nproc_per_node=8 \
main_dino.py --arch vit_small \
--data_path /tmp/zhongz2/images${VERSION}/train \
--epochs 300 \
--output_dir ./saving_dir_mnist_300${VERSION}${PARAM_VERSION} \
--global_crops_scale 0.5 1 --local_crops_scale 0.2 0.5



CUDA_VISIBLE_DEVICES=0 python eval_image_retrieval_mnist.py \
--imsize 224 \
--data_path /tmp/zhongz2/images \
--pretrained_weights ./saving_dir_mnist/checkpoint.pth \
--savefilename "val_feats_v2.pkl"

CUDA_VISIBLE_DEVICES=0 python eval_image_retrieval_mnist.py \
--imsize 224 \
--data_path /tmp/zhongz2/images \
--pretrained_weights ./saving_dir_mnist_300/checkpoint0160.pth \
--savefilename "val_feats_300.pkl"

CUDA_VISIBLE_DEVICES=0 python eval_image_retrieval_mnist.py \
--imsize 224 \
--data_path /tmp/zhongz2/images \
--pretrained_weights ./saving_dir_mnist_300/checkpoint.pth \
--savefilename "val_feats_300_final.pkl"


CUDA_VISIBLE_DEVICES=0 python eval_image_retrieval_mnist.py \
--imsize 224 \
--data_path /tmp/zhongz2/images_v2 \
--pretrained_weights ./saving_dir_mnist_300_v2/checkpoint180.pth \
--savefilename "val_feats_300_v2.pkl"


CUDA_VISIBLE_DEVICES=0 python eval_image_retrieval_mnist.py \
--imsize 224 \
--data_path /tmp/zhongz2/images_v2 \
--pretrained_weights ./saving_dir_mnist_300_v2/checkpoint.pth \
--savefilename "val_feats_300_v2_final.pkl"

CUDA_VISIBLE_DEVICES=0 python eval_image_retrieval_mnist.py \
--imsize 224 \
--data_path /tmp/zhongz2/images_v1 \
--pretrained_weights ./saving_dir_mnist_300_v1/checkpoint.pth \
--savefilename "val_feats_300_v1_final.pkl"

CUDA_VISIBLE_DEVICES=0 python eval_image_retrieval_mnist.py \
--imsize 224 \
--data_path /tmp/zhongz2/images_v3 \
--pretrained_weights ./saving_dir_mnist_300_v3/checkpoint.pth \
--savefilename "val_feats_300_v3_final.pkl"

CUDA_VISIBLE_DEVICES=0 python eval_image_retrieval_mnist.py \
--imsize 224 \
--data_path /tmp/zhongz2/images_v3 \
--pretrained_weights ./saving_dir_mnist_300_v3_v1/checkpoint.pth \
--savefilename "val_feats_300_v3_v1_final.pkl"

CUDA_VISIBLE_DEVICES=0 python eval_image_retrieval_mnist.py \
--imsize 224 \
--data_path /tmp/zhongz2/images_v3 \
--pretrained_weights ./saving_dir_mnist_300_v3_v2/checkpoint.pth \
--savefilename "val_feats_300_v3_v2_final.pkl"

CUDA_VISIBLE_DEVICES=0 python eval_image_retrieval_mnist.py \
--imsize 224 \
--data_path /tmp/zhongz2/images \
--savefilename "val_feats_official.pkl"

















