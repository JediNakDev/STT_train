#STT

##Train

###Custom Dataset
'''
ngpu=4 # number of GPUs to perform distributed training on.

torchrun --nproc_per_node=${ngpu} train_custom_dataset.py \
--model_name biodatlab/whisper-th-large-v3-combined \
--language Thai \
--sampling_rate 16000 \
--num_proc 2 \
--train_strategy epoch \
--learning_rate 3e-3 \
--warmup 1000 \
--train_batchsize 16 \
--eval_batchsize 8 \
--num_epochs 20 \
--resume_from_ckpt None \
--output_dir op_dir_epoch \
--train_datasets output_data_directory/train_dataset_1 output_data_directory/train_dataset_2 \
--eval_datasets output_data_directory/eval_dataset_1 output_data_directory/eval_dataset_2 output_data_directory/eval_dataset_3
'''

###Hf Dataset
'''
ngpu=4 # number of GPUs to perform distributed training on.

torchrun --nproc_per_node=${ngpu} train_hf_dataset.py \
--model_name biodatlab/whisper-th-large-v3-combined \
--language Thai \
--sampling_rate 16000 \
--num_proc 2 \
--train_strategy steps \
--learning_rate 3e-3 \
--warmup 1000 \
--train_batchsize 16 \
--eval_batchsize 8 \
--num_steps 10000 \
--resume_from_ckpt None \
--output_dir op_dir_steps \
--train_datasets "google/fleurs" \
--train_dataset_configs th_th \
--train_dataset_splits train validation \
--train_dataset_text_columns transcription \
--eval_datasets google/fleurs \
--eval_dataset_configs th_th \
--eval_dataset_splits test \
--eval_dataset_text_columns transcription
'''

##Evaluate

###Custom Dataset
'''
python3 evaluate/evaluate_on_custom_dataset.py \
--is_public_repo True \
--hf_model biodatlab/whisper-th-large-v3-combine \
--language th \
--eval_datasets output_data_directory/eval_dataset_1 output_data_directory/eval_dataset_2 \
--device 0 \
--batch_size 16 \
--output_dir predictions_dir
'''

###Hf Dataset
'''
python3 evaluate/evaluate_on_hf_dataset.py \
--is_public_repo True \
--hf_model biodatlab/whisper-th-large-v3-combined \
--language th \
--dataset "google/fleurs" \
--config th_th \
--split test \
--device 0 \
--batch_size 16 \
--output_dir predictions_dir
'''

##Transcribe
'''
python3 transcribe_audio.py \
--is_public_repo True \
--hf_model biodatlab/whisper-th-large-v3-combined \
--path_to_audio /path/to/audio/file.wav \
--language th \
--device 0
'''
