export PYTHONPATH="."
export CUDA_VISIBLE_DEVICES="0"
# train 4-bit 64-rank llama-2-7b with LoftQ on GSM8K using one A100
python train_gsm8k.py \
  --model_name_or_path LoftQ/Llama-2-7b-hf-4bit-64rank \
  --learning_rate 3e-4 \
  --seed 11 \
  --expt_name gsm8k_llama2_7b_4bit_64rank_loftq_26_2 \
  --output_dir exp_results/ \
  --num_train_epochs 6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "epoch" \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --do_train \
  --report_to tensorboard
