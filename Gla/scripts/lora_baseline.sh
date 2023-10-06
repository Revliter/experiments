python baseline.py \
    --train_layers [[28],[29],[30],[31]] \
    --lr_scaling [1.0,1.0,1.0,1.0] \
    --train_last_layer \
    --output_dir '../ckpts/lora_baseline' \
    --lora_weights 'tloen/alpaca-lora-7b' \
    &>lora_baseline.out