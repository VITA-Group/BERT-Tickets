### Pretrain IMP

python -u LT_pretrain.py --output_dir LT_pretrain_model --model_type bert --model_name_or_path bert-base-uncased --train_data_file pretrain_data/en.train --do_train --eval_data_file pretrain_data/en.valid --do_eval --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --evaluate_during_training --num_train_epochs 1 --logging_steps 10000 --save_steps 10000 --mlm --overwrite_output_dir --seed 57

### Finetune

##### Glue:

python -u glue_trans.py --dir pre --weight_pertub tmp/shuffle_weight.pt --mask_dir tmp/dif_mask/mnli_mask.pt --output_dir tmp/530/mnli --logging_steps 12271 --task_name MNLI --data_dir glue_data/MNLI --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 57

##### SQuAD:

python -u squad_trans.py --dir pre --weight_pertub tmp/shuffle_weight.pt --mask_dir tmp/dif_mask/squad_mask.pt --output_dir tmp/530/squad --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --train_file SQuAD/train-v1.1.json --predict_file SQuAD/dev-v1.1.json --per_gpu_train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 4 --max_seq_length 384 --doc_stride 128 --evaluate_during_training --eval_all_checkpoints --overwrite_output_dir --logging_steps 5500 --save_steps 0 --seed 57

##### Pretrain:

python -u pretrain_trans.py --dir pre --weight_pertub tmp/shuffle_weight.pt --mask_dir tmp/dif_mask/pretrain_mask.pt --output_dir tmp/530/pre --model_type bert --model_name_or_path bert-base-uncased --train_data_file pretrain_data/en.train --do_train --eval_data_file pretrain_data/en.valid --do_eval --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 8 --evaluate_during_training --num_train_epochs 1 --logging_steps 2000 --save_steps 0 --max_steps 20000  --mlm --overwrite_output_dir --seed 57

## Grasp

##### Glue:

python -u glue_grasp.py --tt 0.4 --output_dir tmp/trans_result50/mrpc-squad --logging_steps 114 --task_name MRPC --data_dir glue_data/MRPC --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --max_seq_length 128 --per_gpu_train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 1 --overwrite_output_dir --evaluate_during_training --save_steps 0 --eval_all_checkpoints --seed 10

##### SQuAD:

python -u squad_grasp.py --output_dir tmp/trans_result50/squad-qnli --model_type bert --model_name_or_path bert-base-uncased --do_train --do_eval --do_lower_case --train_file SQuAD/train-v1.1.json --predict_file SQuAD/dev-v1.1.json --per_gpu_train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 2 --max_seq_length 384 --doc_stride 128 --evaluate_during_training --eval_all_checkpoints --overwrite_output_dir --logging_steps 5500 --save_steps 0 --seed 10

##### Pretrain:

python -u pretrain_grasp.py --output_dir tmp/trans_result50/pretrain-squad --model_type bert --model_name_or_path bert-base-uncased --train_data_file pretrain_data/en.train --do_train --eval_data_file pretrain_data/en.valid --do_eval --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 6 --evaluate_during_training --num_train_epochs 1 --logging_steps 1000 --save_steps 0 --max_steps 10000  --mlm --overwrite_output_dir

### OMP:

python oneshot.py --weight pre --model glue --rate 0.5

### RF:

python oneshot.py --weight rand --model glue --rate 0.5

### Suffule Weight:

python pertub_weight.py