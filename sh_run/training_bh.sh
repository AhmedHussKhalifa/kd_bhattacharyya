
export dataset=cifar100
method=bh
# python3 examples/image_classification.py \
# --config configs/sample/cifar100/ce/densenet_bc_k12_depth100-final_run.yaml \
# --log log/cifar100/ce/densenet_bc_k12_depth100/densenet_bc_k12_depth100.txt \
# -adjust_lr

# python3 examples/image_classification.py \
# --config configs/sample/${dataset}/${method}/wide_resnet40_1_from_wide_resnet40_4-hyperparameter_tuning.yaml \
# --log log/${dataset}/${method}/wide_resnet40_1_from_wide_resnet40_4/wide_resnet40_1_from_wide_resnet40_4-hyperparameter_tuning.log \
# -adjust_lr --temp 4 --alpha 0.5 --index 0.5

NUM_GPUS=2
GPU_ID=0,1

# at Tempurature = 1 loss became Zero 

for temp in 4
do
    for index in 0.5
    do
        CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} \
        --master_addr 127.0.0.2 --master_port 29502 \
        --use_env examples/image_classification.py \
        --config configs/sample/${dataset}/${method}/wide_resnet40_1_from_wide_resnet40_4-final_run.yaml \
        --log log/${dataset}/${method}/wide_resnet40_1_from_wide_resnet40_4/wide_resnet40_1_from_wide_resnet40_4-final_run.log \
        --temp ${temp} --index ${index} --alpha 0.5 \
        -adjust_lr --world_size ${NUM_GPUS} 
    done
done



