#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
# evaluate
#python3.6 test.py --checkpoint "./work/GRU/GRU0/training/best/gru_9293.ckpt-11600" --model_architecture gru --model_size_info 1 100 --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 40

# train quantized model
python3.6 train_quant.py --optimizer_type adam --quant_type fixed --num_bits 4 --model_architecture gru --model_size_info 1 100 --start_checkpoint "./work/GRU/GRU0/training/best/gru_9293.ckpt-11600" --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 40 --learning_rate 0.001,0.0008,0.0005 --how_many_training_steps 5000,5000,5000 --summaries_dir work/GRU/GRU_Q/retrain_logs --train_dir work/GRU/GRU_Q/training

#quant without training
#python3.6 train_quant.py --evaluate True --optimizer_type adam --quant_type fixed --num_bits 4 --model_architecture gru --model_size_info 1 100 --start_checkpoint "./work/GRU/GRU0/training/best/gru_9293.ckpt-11600" --dct_coefficient_count 10 --window_size_ms 40 --window_stride_ms 40 --learning_rate 0.001,0.0008,0.0005 --how_many_training_steps 5000,5000,5000 --summaries_dir work/GRU/GRU_Q/retrain_logs --train_dir work/GRU/GRU_Q/training
