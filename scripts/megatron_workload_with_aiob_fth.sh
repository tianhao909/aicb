#!/bin/sh  # 使用sh脚本解释器

# 定义默认的参数值
frame=Megatron  # 通信框架，默认使用 Megatron
world_size=32  # 全局训练的总设备数，默认为32
tensor_model_parallel_size=8  # 张量并行的大小，默认为8
pipeline_model_parallel=1  # 管道并行的大小，默认为1
global_batch=1024  # 全局批次大小，默认为1024
micro_batch=1  # 微批次大小，默认为1
num_layers=40  # 模型层数，默认为40
seq_length=4096  # 序列长度，默认为4096
hidden_size=5120  # 隐藏层大小，默认为5120
epoch_num=1  # 训练轮数，默认为1
num_attention_heads=40  # 注意力头数，默认为40
aiob_enable=  # AIOB启用标志，默认未启用
use_flash_attn=  # 是否使用Flash Attention，默认未启用
swiglu=  # 是否使用SwigLU激活函数，默认未启用
sp_enable=  # 是否启用序列并行，默认未启用
ffn_hidden_size=  # FFN隐藏层大小，默认为空（未设置）
comp_filepath=  # 计算文件路径，默认为空（未设置）
model_size=13  # 模型大小，默认为13
max_position_embeddings=4096  # 最大位置嵌入数，默认为4096
vocab_size=50257  # 词汇表大小，默认为50257
num_experts=1  # 专家模型的数量，默认为1
moe_enable=  # 是否启用MoE（混合专家模型），默认未启用
recompute_activations=  # 是否重新计算激活值，默认未启用
gpu_type=  # GPU类型，默认为空（未设置）

# 定义用法说明
usage() {
  echo "Usage: \$0 [options]
    options:
      --frame              通信框架，默认为$frame
      --world_size              全局设备数量，默认为$world_size
      --tensor_model_parallel_size                  张量并行大小，默认为$tensor_model_parallel_size
      --pipeline_model_parallel                  管道并行大小，默认为$pipeline_model_parallel
      --global_batch            全局批次大小，默认为$global_batch
      --micro_batch             微批次大小，默认为$micro_batch
      --num_layers              层数，默认为$num_layers
      --seq_length              序列长度，默认为$seq_length
      --hidden_size             隐藏层大小，默认为$hidden_size
      --epoch_num               训练轮数，默认为$epoch_num
      --use_distributed_optimizer 是否使用分布式优化器
      --num_attention_heads     注意力头数，默认为$num_attention_heads
      --aiob_enable             启用AIOB
      --use_flash_attn          使用Flash Attention
      --swiglu                  使用SwigLU激活函数
      --ffn_hidden_size         FFN隐藏层大小
      --comp_filepath           计算文件路径
      --max_position_embeddings 最大位置嵌入，默认为$max_position_embeddings
      -m, --model_size          模型大小，默认为$model_size（可选值：175，22，13，7，moe）
      --moe_enable             启用MoE
      --moe_router_topk         路由到每个token的专家数量
      --expert_model_parallel_size     专家模型并行度
      --num_experts          MoE模型中的专家数量  
      --moe_grouped_gemm        启用分组GEMM
      -h, --help                显示帮助并退出" 1>&2; exit 1;
}

# 解析输入参数
while [ $# -gt 0 ]  # 当命令行参数大于0时，循环解析
do
  case $1 in
    --gpu_type)  # 解析GPU类型参数
      gpu_type=$2; shift;;
    --frame)  # 解析通信框架参数
      frame=$2; shift;;
    --world_size)  # 解析全局设备数量参数
      world_size=$2; shift;;
    --tensor_model_parallel_size|--tp)  # 解析张量并行大小参数
      tensor_model_parallel_size=$2; shift;;
    --pipeline_model_parallel|--pp)  # 解析管道并行参数
      pipeline_model_parallel=$2; shift;;
    --global_batch)  # 解析全局批次大小参数
      global_batch=$2; shift;;
    --micro_batch)  # 解析微批次大小参数
      micro_batch=$2; shift;;
    --num_layers)  # 解析层数参数
      num_layers=$2; shift;;
    --seq_length)  # 解析序列长度参数
      seq_length=$2; shift;;
    --hidden_size)  # 解析隐藏层大小参数
      hidden_size=$2; shift;;
    --epoch_num)  # 解析训练轮数参数
      epoch_num=$2; shift;;
    --num_attention_heads)  # 解析注意力头数参数
      num_attention_heads=$2; shift;;
    --aiob_enable|--aiob)  # 解析是否启用AIOB参数
      aiob_enable=--aiob_enable;;
    --use_flash_attn|--flash_attn)  # 解析是否启用Flash Attention参数
      use_flash_attn=--use_flash_attn;;
    --swiglu)  # 解析是否启用SwigLU激活函数参数
      swiglu=--swiglu;;
    --ffn_hidden_size)  # 解析FFN隐藏层大小参数
      ffn_hidden_size=$2; shift;;
    --sp|--sp-enable)  # 解析是否启用序列并行参数
      sp_enable=--enable_sequence_parallel;;
    --comp_filepath)  # 解析计算文件路径参数
      comp_filepath=$2; shift;;
    -m|--model_size)  # 解析模型大小参数
      model_size=$2; shift;;
    --max_position_embeddings)  # 解析最大位置嵌入数参数
      max_position_embeddings=$2; shift;;
    --moe_enable)  # 解析是否启用MoE参数
      moe_enable=--moe_enable;;
    --moe_router_topk|--topk)  # 解析每个token路由的专家数量参数
      moe_router_topk=$2; shift;;
    --num_experts|--experts)  # 解析专家数量参数
      num_experts=$2; shift;;
    --expert_model_parallel_size|--ep)  # 解析专家模型并行度参数
      expert_model_parallel_size=$2; shift;;
    --grouped_gemm|--moe_grouped_gemm)  # 解析是否启用分组GEMM参数
      grouped_gemm=--moe_grouped_gemm;;
    --recompute_activations|--recompute)  # 解析是否重新计算激活值参数
      recompute_activations=--recompute_activations;;
    -h|--help)  # 解析帮助选项
      usage;;
    (*)  # 如果是其他不识别的选项，则跳出循环
      break;;
  esac
  shift  # 移动到下一个命令行参数
done

# 根据模型大小选择具体的配置
case $model_size in
  175)  # 如果模型大小为175
    model_name=gpt_175B  # 设置模型名称为gpt_175B
    num_layers=96  # 设置层数为96
    hidden_size=12288  # 设置隐藏层大小为12288
    num_attention_heads=96  # 设置注意力头数为96
    tensor_model_parallel_size=8  # 设置张量并行大小为8
    ;;
  22)  # 如果模型大小为22
    model_name=gpt_22B
    num_layers=48
    hidden_size=6144
    num_attention_heads=64
    tensor_model_parallel_size=8
    ;;
  13)  # 如果模型大小为13
    model_name=gpt_13B
    num_layers=40
    hidden_size=5120
    num_attention_heads=40
    ;;
  7)  # 如果模型大小为7
    model_name=gpt_7B
    num_layers=36
    hidden_size=4096
    num_attention_heads=32
    tensor_model_parallel_size=4
    ;;
  405)  # 如果模型大小为405
    model_name=llama_405B
    num_layers=128
    hidden_size=16384
    ffn_hidden_size=53248
    num_attention_heads=128
    ;;
  moe)  # 如果是moe类型的模型
    model_name=Mixtral_8*7B
    num_layers=32
    hidden_size=4096
    num_attention_heads=32
    ffn_hidden_size=14336
    tensor_model_parallel_size=4
    moe_enable=--moe_enable
    grouped_gemm=--moe_grouped_gemm
    ;;
  (*)
    echo "Only support model size 175, 22,13 or 7; using default size 13"  # 如果模型大小不支持，使用默认配置
    model_name=gpt_13B
    num_layers=40
    hidden_size=5120
    num_attention_heads=40
    ;;
esac

# 构造要执行的命令
cmd="python -m workload_generator.AIOB_simAI_workload_generator \
  --gpu_type=$gpu_type \
  --frame=$frame \
  --world_size=$world_size \
  --tensor_model_parallel_size=$tensor_model_parallel_size \
  --pipeline_model_parallel=$pipeline_model_parallel \
  --global_batch=$global_batch \
  --micro_batch=$micro_batch \
  --num_layers=$num_layers \
  --seq_length=$seq_length \
  --hidden_size=$hidden_size \
  --epoch_num=$epoch_num \
  --num_attention_heads=$num_attention_heads \
  --model_name=$model_name \
  --max_position_embeddings=$max_position_embeddings \
  --vocab_size=$vocab_size \
  --use-distributed-optimizer
  ${aiob_enable} \
  ${use_flash_attn} \
  ${swiglu} \
  ${sp_enable} \
  ${recompute_activations} \
  ${ffn_hidden_size:+--ffn_hidden_size=$ffn_hidden_size} \
  ${comp_filepath:+--comp_filepath=$comp_filepath} \
  ${moe_enable} \
  ${moe_router_topk:+--moe_router_topk=$moe_router_topk} \
  ${num_experts:+--num_experts=$num_experts} \
  ${expert_model_parallel_size:+--expert_model_parallel_size=$expert_model_parallel_size} \
  ${grouped_gemm} " \

echo $cmd  # 打印构造的命令，便于调试和确认

$cmd  # 执行构造的命令
