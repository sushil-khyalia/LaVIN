# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 11112 train_regression.py \
#    --llm_model 8B\
#    --llama_model_path ../data/weights/ \
#    --max_seq_len 512 \
#    --batch_size 4 \
#    --accum_iter 4 \
#    --epochs 20 \
#    --warmup_epochs 3 \
#    --blr 1e-3 \
#    --weight_decay 0.02 \
#    --output_dir ./LaVIN-8B-3.1-IEMOCAP-Valence/\
#    --adapter_type attn\
#    --adapter_dim 8\
#    --adapter_scale 1\
#    --prompt_format QCM-ALE \
#    --temperature 10.\
#    --visual_adapter_type router \


CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 11112 eval_mosi_regression.py \
   --llm_model 8B\
   --llama_model_path ../data/weights/ \
   --data_path ../data/alpaca_data.json \
   --max_seq_len 512 \
   --batch_size 32 \
   --accum_iter 1 \
   --epochs 20 \
   --warmup_epochs 2 \
   --blr 9e-3 \
   --weight_decay 0.02 \
   --output_dir ./LaVIN-8B/ \
   --adapter_type attn \
   --adapter_dim 8 \
   --adapter_scale 1 \
   --n_prompt 6 \
   --prompt_format QCM-ALE \
   --temperature 10. \
   --visual_adapter_type router \
   --resume /work/skhyalia/LaVIN/LaVIN-8B-3.1-IEMOCAP-Valence/checkpoint-12.pth \