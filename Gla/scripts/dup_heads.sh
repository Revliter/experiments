python finetune.py \
    --train_layers [[28],[29],[30],[31]] \
    --lr_scaling [1.0,1.0,1.0,1.0] \
    --output_dir '../ckpts/debug' \
    --use_duplicate_head \
    --fix_heads \
    &>dup_heads.out