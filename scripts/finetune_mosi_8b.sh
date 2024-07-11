CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 11112 train.py \
   --llm_model 8B\
   --llama_model_path ../data/weights/ \
   --max_seq_len 512 \
   --batch_size 4 \
   --accum_iter 4 \
   --epochs 30 \
   --warmup_epochs 2 \
   --blr 9e-3 \
   --weight_decay 0.02 \
   --output_dir ./LaVIN-8B/\
   --adapter_type attn\
   --adapter_dim 8\
   --adapter_scale 1\
   --prompt_format QCM-ALE \
   --temperature 10.\
   --visual_adapter_type router


CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 11111 eval_mosi.py \
    --ckpt_dir ../data/weights/ \
    --llm_model 8B\
    --tokenizer_path ../data/weights/tokenizer.model \
    --adapter_path ./LaVIN-8B/checkpoint-29.pth \
    --adapter_type attn \
    --adapter_dim 8 \
    --adapter_scale 1 \
    --max_batch_size 16\
    --max_seq_len 512 \
    --split valid \
    --temperature 10.\
    --visual_adapter_type router
