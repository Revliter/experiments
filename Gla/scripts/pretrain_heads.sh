python finetune.py \
    --train_layers [[28],[29],[30],[31]] \
    --lr_scaling [1.0,1.0,1.0,1.0] \
    --output_dir '../ckpts/pretrain_heads' \
    --load_head \
    --head_path '../ckpts/baseline/head.pt' \
    --fix_heads \
    &>pretrain_heads.out