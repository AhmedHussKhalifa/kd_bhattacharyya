export dataset=cifar100
NUM_GPUS=1
GPU_ID=4
method=ce


CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} \
--master_addr 127.0.0.2 --master_port 29503 \
--use_env examples/image_classification.py \
--config configs/sample/${dataset}/${method}/wide_resnet40_1-final_run.yaml \
--log log/${dataset}/${method}/wide_resnet40_1/wide_resnet40_1-final_run.log \
-adjust_lr --world_size ${NUM_GPUS} 

