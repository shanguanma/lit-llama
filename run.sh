#!/usr/bin/env python

stage=0
stop_stage=10000

. parse_options.sh
. path_for_lit_llama.sh
pretrain_model_dir=/workspace2/maduo/model_hub/OPT-LLM
mkdir -p $pretrain_model_dir
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
   echo "download llama model weight"
   git lfs install
   it clone https://huggingface.co/openlm-research/open_llama_7b  $pretrain_model_dir/open-llama/7b   
   tree $pretrain_model_dir/open-llama/7b
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
   echo "covert model to specify format"
   lit_llama_dir=/workspace2/maduo/lit-llama
   python $lit_llama_dir/scripts/convert_hf_checkpoint.py --checkpoint_dir $pretrain_model_dir/open-llama/7b --model_size 7B     
   tree ./checkpoints/lit-llama
   mkdir -p $pretrain_model_dir/lit-llama/7b   
   mv ./checkpoints/lit-llama/* $pretrain_model_dir/lit-llama/7b

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   echo "generate answer base on your prompt "
   lit_llama_dir=/workspace2/maduo/lit-llama
   #prompt="Hello,here is Shenzhen"
   #prompt="Hello, my name is"
   CUDA_VISIBLE_DEVICES=6 python $lit_llama_dir/generate.py \
          --quantize llm.int8 \
          --prompt "Hello, my name is" \
          --checkpoint_path $pretrain_model_dir/lit-llama/7b/7B/lit-llama.pth \
          --tokenizer_path $pretrain_model_dir/lit-llama/7b/tokenizer.model   

fi
