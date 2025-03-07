CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 11112 train.py \
  --llm_model 8B\
  --llama_model_path ../data/weights/ \
  --max_seq_len 512 \
  --batch_size 4 \
  --accum_iter 4 \
  --epochs 10 \
  --warmup_epochs 3 \
  --blr 1e-3 \
  --weight_decay 0.02 \
  --output_dir ./LaVIN-8B-3.1-MELD/\
  --adapter_type attn\
  --adapter_dim 8\
  --adapter_scale 1\
  --temperature 10.\
  --visual_adapter_type router \
  --num_workers 10\
  --dataset meld\