"""
Copyright (c) 2021, Alibaba Group;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# 导入工作负载生成器中模拟模型的模块
import workload_generator.mocked_model.MockedDeepspeed  
from workload_generator.mocked_model.MockedMegatron import *  # 导入MockedMegatron模块中的所有内容
from workload_generator.mocked_model.MockedModel import MockedParam, MockedModel  # 导入MockedParam和MockedModel类
from utils.utils import CommType, get_params, get_comp_out, extract_averages  # 从utils.utils中导入常用工具函数
import os  # 导入操作系统相关模块
from typing import List, Tuple  # 导入List和Tuple类型注解
from collections import deque  # 导入双端队列（deque）
import dataclasses  # 导入dataclasses模块用于简化类的定义
from enum import Enum  # 导入枚举类

try:
    import torch  # 尝试导入torch模块
except ImportError as e:  # 如果导入失败，捕获ImportError异常
    torch = None  # 如果导入失败，则将torch设置为None
    print("Failed to import 'torch'.")  # 输出导入失败的提示信息
import math  # 导入数学模块
import re  # 导入正则表达式模块

# 定义一个数据类Work_Item，用于表示每个工作项的属性
@dataclasses.dataclass
class Work_Item:
    name: str = dataclasses.field(default="none")  # 工作项的名称，默认为"none"
    placeholder: int = dataclasses.field(default=-1)  # 占位符，默认为-1
    forward_compute_time: int = dataclasses.field(default=0)  # 正向计算时间，默认为0
    forward_comm: str = dataclasses.field(default="NONE")  # 正向通信方式，默认为"NONE"
    forward_comm_size: int = dataclasses.field(default=0)  # 正向通信的大小，默认为0
    backward_compute_time: int = dataclasses.field(default=0)  # 反向计算时间，默认为0
    backward_comm: str = dataclasses.field(default="NONE")  # 反向通信方式，默认为"NONE"
    backward_comm_size: int = dataclasses.field(default=0)  # 反向通信的大小，默认为0
    dp_compute_time: int = dataclasses.field(default=0)  # 数据并行计算时间，默认为0
    dp_comm: str = dataclasses.field(default="NONE")  # 数据并行通信方式，默认为"NONE"
    dp_comm_size: int = dataclasses.field(default=0)  # 数据并行通信的大小，默认为0
    process_time: int = dataclasses.field(default=100)  # 进程的执行时间，默认为100

# 获取AI操作的计算时间
def _get_aiob_compute_time(compute_cache, forward_or_backward, stage):
    compute_time_map = compute_cache  # 将compute_cache赋值给compute_time_map，表示计算时间的映射字典
    if stage == "grad":  # 如果stage是"grad"
        prefix = stage + "_" + forward_or_backward  # 构造以"grad_"为前缀的键值
    elif stage == "embedding":  # 如果stage是"embedding"
        prefix = "Emb"  # 设置前缀为"Emb"
    elif stage == "final":  # 如果stage是"final"
        prefix = "attention" + "_" + forward_or_backward  # 构造以"attention_"为前缀的键值
    else:
        prefix = stage + "_" + forward_or_backward  # 其他情况，按stage和forward_or_backward构造前缀

    # 遍历计算时间映射字典
    for key, value in compute_time_map.items():
        if prefix == key:  # 如果前缀匹配到某个键
            compute_time = compute_time_map.get(key)  # 获取该键对应的计算时间
            return compute_time  # 返回计算时间

    print("[warn] can't match any stage", stage)  # 如果没有匹配到前缀，输出警告信息
    return 1  # 如果没有匹配到返回默认值1

# 定义LayerInfo类用于存储每一层的相关信息
class LayerInfo:
    def __init__(self, layer_id, layer_name, param_count):
        self.layer_id = layer_id  # 初始化层的ID
        self.layer_name = layer_name  # 初始化层的名称
        self.param_count = param_count  # 初始化层的参数数量


class SIMAI_workload:  # 定义SIMAI工作负载类
    def __init__(self, model, args, compute_cache=None):  # 构造函数，初始化模型、参数和计算缓存
        self.model = model  # 保存模型
        self.args = args  # 保存参数
        self.compute_cache = compute_cache  # 保存计算缓存（可选）
        self.workload = []  # 初始化空的工作负载列表
        self.seq_len = args.seq_length  # 序列长度，从参数中获取
        self.tp = args.tensor_model_parallel_size  # Tensor模型并行大小，从参数中获取
        self.mbs = args.micro_batch  # 微批量大小，从参数中获取
        if args.moe_enable:  # 如果启用了MoE（混合专家模型）
            self.expert_model_parallel_size = args.expert_model_parallel_size  # 保存专家模型并行大小
            self.num_experts = args.num_experts  # 保存专家数量
            self.topk = args.moe_router_topk  # 保存路由到每个token的专家数

    def get_model_details(self):  # 获取模型的详细信息
        layers = []  # 初始化空的层信息列表
        visited = set()  # 初始化访问的模块集合（避免重复遍历）

        def traverse_model(model):  # 定义递归函数来遍历模型
            if id(model) in visited:  # 如果当前模块已访问过
                return  # 直接返回，避免重复访问
            visited.add(id(model))  # 将当前模块添加到已访问集合

            if self.args.enable_sequence_parallel:  # 如果启用了序列并行
                if (
                    isinstance(model, MegatronColumnLinear)  # 如果模型是MegatronColumnLinear类型
                    or isinstance(model, MegatronRowLinear)  # 或者是MegatronRowLinear类型
                    or isinstance(model, MegatronEmbedding)  # 或者是MegatronEmbedding类型
                    or isinstance(model, FusedLayernorm)  # 或者是FusedLayernorm类型
                ):
                    params = model.parameters()  # 获取模型的所有参数
                    param_count = sum(p.numel() for p in params)  # 计算参数总数
                    layers.append(LayerInfo(model.layer_id, model.name, param_count))  # 将层信息添加到layers列表

                if isinstance(model, GroupedMLP):  # 如果模型是GroupedMLP类型
                    moe_params = model.parameters()  # 获取模型的参数
                    moe_param_count = sum(p.numel() for p in moe_params)  # 计算参数总数
                    layers.append(LayerInfo(model.layer_id, model.name, moe_param_count))  # 将层信息添加到layers列表

            else:  # 如果没有启用序列并行
                if (
                    isinstance(model, MegatronAttention)  # 如果模型是MegatronAttention类型
                    or isinstance(model, MegatronMlp)  # 或者是MegatronMlp类型
                    or isinstance(model, MegatronEmbedding)  # 或者是MegatronEmbedding类型
                ):
                    params = model.parameters()  # 获取模型的参数
                    param_count = sum(p.numel() for p in params)  # 计算参数总数
                    layers.append(LayerInfo(model.layer_id, model.name, param_count))  # 将层信息添加到layers列表

            for child in model.child_modules():  # 遍历模型的所有子模块
                traverse_model(child)  # 递归调用traverse_model来遍历子模块

        traverse_model(model)  # 调用traverse_model函数来遍历整个模型

        return layers  # 返回所有层的信息


    # 定义一个函数，用于获取模型的总参数数量和Mixture of Experts（MoE）参数数量
    def _get_total_params(self):
        total_params = 0  # 初始化总参数计数器
        moe_param_count = 0  # 初始化MoE参数计数器
        layers = self.get_model_details()  # 获取模型的层信息
        for layer in layers:  # 遍历每一层
            total_params += layer.param_count  # 累加每层的参数数量
            if "moe" in layer.layer_name:  # 如果层名称包含"moe"，则累加MoE参数
                moe_param_count += layer.param_count

        return total_params, moe_param_count  # 返回总参数数量和MoE参数数量

    def workload_generate_aiob(self):
        # args.world_size --> total gpus number
        self.ga_num = self.args.global_batch // (self.args.micro_batch * self.args.dp_num)  # 计算全局批次大小 / (微批次大小 * 数据并行数量) 得到每个GPU的全局批次数量
        if self.ga_num < 1:  # 如果计算出的全局批次数量小于1，给出警告
            print(
                "[WARN]: ga num < 1, please confirm global_batch num and micro_batch num"  # 提示用户检查全局批次和微批次的配置
            )
        default_compute_time = 1  # 默认的计算时间设为1，可以作为后续的基准
        compute_time = 0  # 初始化计算时间（这里没有用到，可能为后续扩展保留）
        tp_comm_size = (
            2 * self.args.micro_batch * self.args.seq_length * self.args.hidden_size  # 计算模型的通信数据大小，涉及微批次大小、序列长度和隐藏层大小
        )
        layers = self.get_model_details()  # 获取模型的详细信息（层数等）
        total_params, moe_param_count = self._get_total_params()  # 调用类内的_get_total_params方法，获取总参数数量和MoE参数数量
        # self.workload.append(Work_Item(name="norm", forward_compute_time=0,  # 注释掉的代码，向工作负载中添加一个“norm”操作项，表示前向计算时间为0，通信操作等为默认值
        #                         forward_comm = "BROADCAST", forward_comm_size= 8*self.args.micro_batch*self.args.seq_length,
        #                         backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
        #                         dp_compute_time=default_compute_time, dp_comm="NONE", dp_comm_size=0
        #                         ))
        forward_compute_time = _get_aiob_compute_time(
            self.compute_cache, "forward", "grad"  # 调用外部函数_get_aiob_compute_time计算前向计算时间
        )
        backward_compute_time = _get_aiob_compute_time(
            self.compute_cache, "backward", "grad"  # 调用外部函数_get_aiob_compute_time计算反向计算时间
        )
        self.workload.append(
            Work_Item(  # 向工作负载列表添加一个“grad_gather”操作项
                name="grad_gather",  # 操作名为“grad_gather”
                forward_compute_time=default_compute_time,  # 前向计算时间为默认值1
                forward_comm="NONE",  # 前向通信类型为“NONE”，表示没有通信
                forward_comm_size=0,  # 前向通信数据大小为0
                backward_compute_time=default_compute_time,  # 反向计算时间为默认值1
                backward_comm="NONE",  # 反向通信类型为“NONE”，表示没有通信
                backward_comm_size=0,  # 反向通信数据大小为0
                dp_compute_time=default_compute_time,  # 数据并行计算时间为默认值1
                dp_comm="ALLGATHER",  # 数据并行通信类型为“ALLGATHER”，表示所有进程的数据收集
                dp_comm_size=2 * (total_params - moe_param_count),  # 数据并行通信大小为总参数数目减去MoE参数数目，乘以2
            )
        )
        self.workload.append(
            Work_Item(  # 向工作负载列表添加一个“grad_param_comm”操作项
                name="grad_param_comm",  # 操作名为“grad_param_comm”
                forward_compute_time=default_compute_time,  # 前向计算时间为默认值1
                forward_comm="NONE",  # 前向通信类型为“NONE”，表示没有通信
                forward_comm_size=0,  # 前向通信数据大小为0
                backward_compute_time=default_compute_time,  # 反向计算时间为默认值1
                backward_comm="NONE",  # 反向通信类型为“NONE”，表示没有通信
                backward_comm_size=0,  # 反向通信数据大小为0
                dp_compute_time=default_compute_time,  # 数据并行计算时间为默认值1
                dp_comm="REDUCESCATTER",  # 数据并行通信类型为“REDUCESCATTER”，表示数据散布和聚集
                dp_comm_size=4 * (total_params - moe_param_count),  # 数据并行通信大小为总参数数目减去MoE参数数目，乘以4
            )
        )
        self.workload.append(
            Work_Item(  # 向工作负载列表添加一个“grad_param_compute”操作项
                name="grad_param_compute",  # 操作名为“grad_param_compute”
                forward_compute_time=default_compute_time,  # 前向计算时间为默认值1
                forward_comm="NONE",  # 前向通信类型为“NONE”，表示没有通信
                forward_comm_size=0,  # 前向通信数据大小为0
                backward_compute_time=forward_compute_time + backward_compute_time,  # 反向计算时间为前向计算时间与反向计算时间的总和
                backward_comm="NONE",  # 反向通信类型为“NONE”，表示没有通信
                backward_comm_size=0,  # 反向通信数据大小为0
                dp_compute_time=default_compute_time,  # 数据并行计算时间为默认值1
                dp_comm="NONE",  # 数据并行通信类型为“NONE”，表示没有通信
                dp_comm_size=0,  # 数据并行通信数据大小为0
            )
        )

        if not self.args.enable_sequence_parallel:  # 如果未启用序列并行，则进行以下操作
            self.workload.append(
                Work_Item(
                    name="layernorm",  # 任务名称为 layernorm
                    forward_compute_time=default_compute_time,  # 正向计算时间为默认值
                    forward_comm="NONE",  # 正向通信方式为无
                    forward_comm_size=0,  # 正向通信大小为0
                    backward_compute_time=default_compute_time,  # 反向计算时间为默认值
                    backward_comm="ALLREDUCE",  # 反向通信方式为 ALLREDUCE
                    backward_comm_size=2 * total_params,  # 反向通信大小为 2倍的总参数数
                    dp_compute_time=default_compute_time,  # 数据并行计算时间为默认值
                    dp_comm="NONE",  # 数据并行通信方式为无
                    dp_comm_size=0,  # 数据并行通信大小为0
                )
            )
            
        if args.tensor_model_parallel_size == 1:  # 如果张量模型并行的大小为1
            emd_backward_comm = "NONE"  # 设置反向传播通信方式为无
        else:  # 否则
            emd_backward_comm = "ALLREDUCE"  # 设置反向传播通信方式为 ALLREDUCE
        
        # 添加嵌入梯度的工作项
        self.workload.append(
            Work_Item(
                name="embedding_grads",  # 任务名称为 embedding_grads
                forward_compute_time=default_compute_time,  # 正向计算时间为默认值
                forward_comm="NONE",  # 正向通信方式为无
                forward_comm_size=0,  # 正向通信大小为0
                backward_compute_time=default_compute_time,  # 反向计算时间为默认值
                backward_comm=emd_backward_comm,  # 使用之前定义的反向传播通信方式
                backward_comm_size=tp_comm_size,  # 反向通信大小为 tensor 并行的通信大小
                dp_compute_time=default_compute_time,  # 数据并行计算时间为默认值
                dp_comm="NONE",  # 数据并行通信方式为无
                dp_comm_size=0,  # 数据并行通信大小为0
            )
        )

        # 如果专家模型并行数不等于数据并行数
        if self.args.expert_model_parallel_size != self.args.dp_num:
            # 添加工作项 moe_grad_norm1
            self.workload.append(Work_Item(name="moe_grad_norm1", forward_compute_time=default_compute_time,
                                    forward_comm="NONE", forward_comm_size=0,
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
                                    dp_compute_time=default_compute_time, dp_comm="ALLGATHER_DP_EP", dp_comm_size=2 * moe_param_count
                                    ))
            # 添加工作项 moe_grad_norm2
            self.workload.append(Work_Item(name="moe_grad_norm2", forward_compute_time=default_compute_time,
                                    forward_comm="NONE", forward_comm_size=0,
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
                                    dp_compute_time=default_compute_time, dp_comm="REDUCESCATTER_DP_EP", dp_comm_size=4 * moe_param_count
                                    ))


                        
        for _ in range(self.ga_num):  # 遍历全局批次，ga_num 表示全局批次数
            for layer in layers:  # 遍历每个层，layers 是层的列表
                name = layer.layer_name  # 获取当前层的名称
                forward_comm = backward_comm = backward_comm_2 = "NONE"  # 初始化正向、反向通信方式为无
                forward_comm_size = tp_comm_size  # 正向通信大小设置为 tensor 并行的通信大小
                emb_comm_size = tp_comm_size  # 嵌入层的通信大小
                backward_comm_size = 0  # 反向通信大小设置为0
                dp_comm = "NONE"  # 数据并行的通信方式设置为无
                dp_comm_size = 0  # 数据并行通信大小设置为0
                if self.args.enable_sequence_parallel:  # 如果启用了序列并行
                    if "embedding" in name:  # 如果是嵌入层
                        if args.tensor_model_parallel_size == 1:  # 如果张量模型并行大小为1
                            forward_comm = "NONE"  # 正向通信为无
                            backward_comm = "NONE"  # 反向通信为无
                        else:  # 否则，使用 ALLREDUCE
                            forward_comm = "ALLREDUCE"  # 正向通信使用 ALLREDUCE
                            backward_comm = "NONE"  # 反向通信为无
                        emb_compute_time = _get_aiob_compute_time(  # 获取嵌入层的计算时间
                            self.compute_cache, "", "embedding"
                        )
                        self.workload.append(  # 将工作项添加到工作负载
                            Work_Item(
                                name=name,  # 任务名称为当前层名称
                                forward_compute_time=emb_compute_time,  # 正向计算时间为嵌入层的计算时间
                                forward_comm=forward_comm,  # 正向通信方式
                                forward_comm_size=emb_comm_size,  # 正向通信大小
                                backward_compute_time=default_compute_time,  # 反向计算时间为默认值
                                backward_comm=backward_comm,  # 反向通信方式
                                backward_comm_size=backward_comm_size,  # 反向通信大小
                                dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                dp_comm=dp_comm,  # 数据并行通信方式
                                dp_comm_size=dp_comm_size,  # 数据并行通信大小
                            )
                        )
                    if "row" in name:  # 如果是行并行层
                        forward_compute_time = _get_aiob_compute_time(  # 获取正向计算时间
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(  # 获取反向计算时间
                            self.compute_cache, "backward", name.split("_")[0]
                        )

                        if self.args.recompute_activations and 'attention' in name:  # 如果启用了激活重计算并且层包含 attention
                            forward_compute_time *= 2  # 正向计算时间乘以2
                        forward_compute_time = int(forward_compute_time / 2)  # 除以2并取整
                        backward_compute_time = int(backward_compute_time / 2)  # 除以2并取整
                        forward_comm_size_sp = tp_comm_size  # 设置正向通信大小

                        if args.tensor_model_parallel_size == 1:  # 如果张量模型并行大小为1
                            forward_comm = "NONE"  # 正向通信为无
                            backward_comm = "NONE"  # 反向通信为无
                        else:  # 否则，使用 REDUCESCATTER 和 ALLGATHER
                            forward_comm = "REDUCESCATTER"  # 正向通信为 REDUCESCATTER
                            backward_comm = "ALLGATHER"  # 反向通信为 ALLGATHER
                        self.workload.append(  # 将工作项添加到工作负载
                                Work_Item(
                                    name=name,  # 任务名称
                                    forward_compute_time=forward_compute_time,  # 正向计算时间
                                    forward_comm=forward_comm,  # 正向通信方式
                                    forward_comm_size=forward_comm_size,  # 正向通信大小
                                    backward_compute_time=backward_compute_time,  # 反向计算时间
                                    backward_comm=backward_comm,  # 反向通信方式
                                    backward_comm_size=forward_comm_size_sp,  # 反向通信大小
                                    dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size,  # 数据并行通信大小
                                )
                            )

                    elif "column" in name:  # 如果是列并行层
                        forward_compute_time = _get_aiob_compute_time(  # 获取正向计算时间
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(  # 获取反向计算时间
                            self.compute_cache, "backward", name.split("_")[0]
                        )

                        if self.args.recompute_activations and 'attention' in name:  # 如果启用了激活重计算并且层包含 attention
                            forward_compute_time *= 2  # 正向计算时间乘以2
                        forward_compute_time = int(forward_compute_time / 2)  # 除以2并取整
                        backward_compute_time = int(backward_compute_time / 2)  # 除以2并取整

                        if args.tensor_model_parallel_size == 1:  # 如果张量模型并行大小为1
                            forward_comm = "NONE"  # 正向通信为无
                            backward_comm = "NONE"  # 反向通信为无
                            backward_comm_2 = "NONE"  # 第二次反向通信为无
                        else:  # 否则，使用 ALLGATHER 和 REDUCESCATTER
                            forward_comm = "ALLGATHER"  # 正向通信为 ALLGATHER
                            backward_comm = "REDUCESCATTER"  # 反向通信为 REDUCESCATTER
                            backward_comm_2 = "ALLGATHER"  # 第二次反向通信为 ALLGATHER
                        self.workload.append(  # 将工作项添加到工作负载
                                Work_Item(
                                    name=name,  # 任务名称
                                    forward_compute_time=forward_compute_time,  # 正向计算时间
                                    forward_comm=forward_comm,  # 正向通信方式
                                    forward_comm_size=forward_comm_size,  # 正向通信大小
                                    backward_compute_time=backward_compute_time,  # 反向计算时间
                                    backward_comm=backward_comm,  # 反向通信方式
                                    backward_comm_size=backward_comm_size,  # 反向通信大小
                                    dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size,  # 数据并行通信大小
                                )
                            )
                    elif "moelayer" in name:  # 如果是 moe 层
                        forward_compute_time = _get_aiob_compute_time(  # 获取正向计算时间
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(  # 获取反向计算时间
                            self.compute_cache, "backward", name.split("_")[0]
                        )
                        if args.tensor_model_parallel_size == 1:  # 如果张量模型并行大小为1
                            forward_comm1 = "NONE"  # 正向通信1为无
                            forward_comm2 = "NONE"  # 正向通信2为无
                            forward_comm3 = "ALLTOALL_EP"  # 正向通信3为 ALLTOALL_EP
                            forward_comm4 = "NONE"  # 正向通信4为无
                            forward_comm5 = "NONE"  # 正向通信5为无
                            forward_comm6 = "ALLTOALL_EP"  # 正向通信6为 ALLTOALL_EP
                            forward_comm7 = "NONE"  # 正向通信7为无
                        else:  # 否则，使用不同的通信策略
                            forward_comm1 = "ALLGATHER"  # 正向通信1为 ALLGATHER
                            forward_comm2 = "ALLTOALL"  # 正向通信2为 ALLTOALL
                            forward_comm3 = "ALLTOALL_EP"  # 正向通信3为 ALLTOALL_EP
                            forward_comm4 = "ALLGATHER"  # 正向通信4为 ALLGATHER
                            forward_comm5 = "REDUCESCATTER"  # 正向通信5为 REDUCESCATTER
                            forward_comm6 = "ALLTOALL_EP"  # 正向通信6为 ALLTOALL_EP
                            forward_comm7 = "ALLTOALL"  # 正向通信7为 ALLTOALL
                        if args.expert_model_parallel_size != 1:  # 如果专家模型并行大小不为1
                            self.workload.append(Work_Item(  # 添加多个工作项
                                name=name,
                                forward_compute_time=forward_compute_time,
                                forward_comm=forward_comm1,
                                forward_comm_size=2*self.mbs*self.seq_len*self.num_experts,
                                backward_compute_time=backward_compute_time,
                                backward_comm=forward_comm1,
                                backward_comm_size=2*self.mbs*self.seq_len*self.num_experts,
                                dp_compute_time=default_compute_time,
                                dp_comm=dp_comm,
                                dp_comm_size=dp_comm_size,
                            ))
                            # 更多类似的 Work_Item 添加
                        else:
                            # 处理专家模型并行大小为1的情况
                            self.workload.append(Work_Item(  # 添加单个工作项
                                name=name,
                                forward_compute_time=forward_compute_time,
                                forward_comm=forward_comm1,
                                forward_comm_size=2*self.mbs*self.seq_len*self.num_experts,
                                backward_compute_time=backward_compute_time,
                                backward_comm=forward_comm1,
                                backward_comm_size=2*self.mbs*self.seq_len*self.num_experts,
                                dp_compute_time=default_compute_time,
                                dp_comm=dp_comm,
                                dp_comm_size=dp_comm_size,
                            ))
                            # 更多类似的 Work_Item 添加
                else:  # 如果没有启用序列并行
                    if args.tensor_model_parallel_size == 1:  # 如果张量模型并行大小为1
                        forward_comm = "NONE"  # 正向通信为无
                        backward_comm = "NONE"  # 反向通信为无
                    else:  # 否则，使用 ALLREDUCE
                        forward_comm = "ALLREDUCE"  # 正向通信为 ALLREDUCE
                        backward_comm = "NONE"  # 反向通信为无
                    if self.args.recompute_activations and 'attention' in name:  # 如果启用了激活重计算并且层包含 attention
                        forward_compute_time *= 2  # 正向计算时间乘以2
                    if "embedding" in name:  # 如果是嵌入层
                        emb_compute_time = _get_aiob_compute_time(  # 获取嵌入层计算时间
                            self.compute_cache, "", "embedding"
                        )
                        self.workload.append(  # 添加嵌入层的工作项
                            Work_Item(
                                name=name,
                                forward_compute_time=emb_compute_time,
                                forward_comm=forward_comm,
                                forward_comm_size=forward_comm_size,
                                backward_compute_time=default_compute_time,
                                backward_comm=backward_comm,
                                backward_comm_size=backward_comm_size,
                                dp_compute_time=backward_compute_time,
                                dp_comm=dp_comm,
                                dp_comm_size=dp_comm_size,
                            )
                        )
                    else:  # 其他层的处理
                        forward_compute_time = _get_aiob_compute_time(  # 获取正向计算时间
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(  # 获取反向计算时间
                            self.compute_cache, "backward", name.split("_")[0]
                        )
                        self.workload.append(  # 添加工作项
                            Work_Item(
                                name=name,
                                forward_compute_time=forward_compute_time,
                                forward_comm=forward_comm,
                                forward_comm_size=forward_comm_size,
                                backward_compute_time=backward_compute_time,
                                backward_comm=backward_comm,
                                backward_comm_size=backward_comm_size,
                                dp_compute_time=backward_compute_time,
                                dp_comm=dp_comm,
                                dp_comm_size=dp_comm_size,
                            )
                        )

            # compute_time = _get_aiob_compute_time(self.compute_cache, "forward", "embedding")
            # self.workload.append(Work_Item(name="embedding_norm", forward_compute_time=compute_time,
            #                         forward_comm = "ALLREDUCE", forward_comm_size= self.args.vocab_size*self.args.hidden_size*2,
            #                         backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
            #                         dp_compute_time=default_compute_time, dp_comm="NONE", dp_comm_size=0
            #                         ))
            

            
        for i in range(3):  # 循环3次，i从0到2
            self.workload.append(  # 向工作负载列表添加一个新的工作项
                Work_Item(  # 创建一个新的Work_Item实例
                    name="cross_entropy" + str(i + 1),  # 工作项的名称为"cross_entropy"后跟数字1、2、3
                    forward_compute_time=compute_time,  # 设置正向计算时间为预先定义的compute_time
                    forward_comm="ALLREDUCE",  # 设置正向通信方式为"ALLREDUCE"
                    forward_comm_size=self.args.seq_length * self.args.micro_batch * 4,  # 计算正向通信的大小，等于序列长度 * 微批次大小 * 4
                    backward_compute_time=compute_time,  # 设置反向计算时间与正向计算时间相同
                    backward_comm="NONE",  # 设置反向通信方式为"NONE"（不进行通信）
                    backward_comm_size=0,  # 反向通信大小为0
                    dp_compute_time=compute_time,  # 设置数据并行计算时间与正向计算时间相同
                    dp_comm="NONE",  # 设置数据并行通信方式为"NONE"（不进行通信）
                    dp_comm_size=0,  # 数据并行通信大小为0
                )
            )  # 结束Work_Item的构造，并将其添加到workload列表中

        for i in range(4):  # 循环4次，i从0到3
            self.workload.append(  # 向工作负载列表添加一个新的工作项
                Work_Item(  # 创建一个新的Work_Item实例
                    name="optimizer" + str(i + 1),  # 工作项的名称为"optimizer"后跟数字1、2、3、4
                    forward_compute_time=compute_time,  # 设置正向计算时间为预先定义的compute_time
                    forward_comm="ALLREDUCE",  # 设置正向通信方式为"ALLREDUCE"
                    forward_comm_size=4,  # 设置正向通信大小为4（假设这是一个固定的值）
                    backward_compute_time=compute_time,  # 设置反向计算时间与正向计算时间相同
                    backward_comm="NONE",  # 设置反向通信方式为"NONE"（不进行通信）
                    backward_comm_size=0,  # 反向通信大小为0
                    dp_compute_time=compute_time,  # 设置数据并行计算时间与正向计算时间相同
                    dp_comm="NONE",  # 设置数据并行通信方式为"NONE"（不进行通信）
                    dp_comm_size=0,  # 数据并行通信大小为0
                )
            )  # 结束Work_Item的构造，并将其添加到workload列表中



    def workload_generate(self):  # 定义生成工作负载的函数
        # args.world_size --> total gpus number  # 获取总的 GPU 数量 (通过 world_size 参数)
        self.ga_num = self.args.global_batch // (self.args.micro_batch * self.args.dp_num)  # 计算每个全局批次的工作单元数
        if self.ga_num < 1:  # 如果计算出的工作单元数小于1，则警告
            print(
                "[WARN]: ga num < 1, please confirm global_batch num and micro_batch num"
            )  # 打印警告信息，提示 global_batch 或 micro_batch 可能设置不当
        default_compute_time = 1  # 设置默认计算时间为1
        compute_time = 0  # 初始化计算时间为0
        tp_comm_size = (  # 计算张量并行通信的大小
            2 * self.args.micro_batch * self.args.seq_length * self.args.hidden_size
        )  # 张量并行通信大小 = 2 * 微批次大小 * 序列长度 * 隐藏层大小
        layers = self.get_model_details()  # 获取模型的详细层信息
        total_params, moe_param_count = self._get_total_params()  # 获取模型的总参数数和 MoE 模型的参数数
        # print(f"Total params is {total_params}, moe params is {moe_param_count}")  # 打印总参数数和 MoE 参数数（注释掉了）

        forward_compute_time = default_compute_time  # 将正向计算时间设置为默认值
        backward_compute_time = default_compute_time  # 将反向计算时间设置为默认值
        
        self.workload.append(  # 向工作负载列表中添加新的工作项
            Work_Item(  # 创建一个新的 Work_Item 实例
                name="grad_norm",  # 设置工作项的名称为 "grad_norm"
                forward_compute_time=forward_compute_time,  # 设置正向计算时间
                forward_comm="ALLGATHER",  # 设置正向通信方式为 "ALLGATHER"
                forward_comm_size=2 * total_params,  # 设置正向通信的大小为 2 * 总参数数
                backward_compute_time=backward_compute_time,  # 设置反向计算时间
                backward_comm="NONE",  # 设置反向通信方式为 "NONE"（不进行通信）
                backward_comm_size=0,  # 设置反向通信的大小为 0
                dp_compute_time=default_compute_time,  # 设置数据并行计算时间为默认值
                dp_comm="REDUCESCATTER",  # 设置数据并行通信方式为 "REDUCESCATTER"
                dp_comm_size=4 * total_params,  # 设置数据并行通信的大小为 4 * 总参数数
            )
        )  # 结束 Work_Item 的构造并将其添加到工作负载列表中

            
        if not self.args.enable_sequence_parallel:  # 如果没有启用序列并行
            self.workload.append(  # 向工作负载列表中添加一个新的工作项
                Work_Item(  # 创建一个新的 Work_Item 实例
                    name="layernorm",  # 设置工作项的名称为 "layernorm"
                    forward_compute_time=default_compute_time,  # 设置正向计算时间为默认值
                    forward_comm="NONE",  # 正向通信方式为 "NONE"（不进行通信）
                    forward_comm_size=0,  # 正向通信的大小为 0
                    backward_compute_time=default_compute_time,  # 设置反向计算时间为默认值
                    backward_comm="ALLREDUCE",  # 反向通信方式为 "ALLREDUCE"
                    backward_comm_size=2 * total_params,  # 反向通信的大小为 2 * 总参数数
                    dp_compute_time=default_compute_time,  # 设置数据并行计算时间为默认值
                    dp_comm="NONE",  # 数据并行通信方式为 "NONE"（不进行通信）
                    dp_comm_size=0,  # 数据并行通信的大小为 0
                )
            )  # 结束 Work_Item 的构造并将其添加到工作负载列表中

        if args.expert_model_parallel_size != args.dp_num:  # 如果专家模型并行大小与数据并行数不相等
            self.workload.append(  # 向工作负载列表中添加一个新的工作项
                Work_Item(  # 创建一个新的 Work_Item 实例
                    name="moe_grad_norm1",  # 设置工作项的名称为 "moe_grad_norm1"
                    forward_compute_time=default_compute_time,  # 设置正向计算时间为默认值
                    forward_comm="NONE",  # 正向通信方式为 "NONE"（不进行通信）
                    forward_comm_size=0,  # 正向通信的大小为 0
                    backward_compute_time=default_compute_time,  # 设置反向计算时间为默认值
                    backward_comm="NONE",  # 反向通信方式为 "NONE"（不进行通信）
                    backward_comm_size=0,  # 反向通信的大小为 0
                    dp_compute_time=default_compute_time,  # 设置数据并行计算时间为默认值
                    dp_comm="ALLGATHER_DP_EP",  # 数据并行通信方式为 "ALLGATHER_DP_EP"
                    dp_comm_size=2 * moe_param_count,  # 数据并行通信的大小为 2 * MoE 参数数
                )
            )  # 结束 Work_Item 的构造并将其添加到工作负载列表中

            self.workload.append(  # 向工作负载列表中添加另一个新的工作项
                Work_Item(  # 创建一个新的 Work_Item 实例
                    name="moe_grad_norm2",  # 设置工作项的名称为 "moe_grad_norm2"
                    forward_compute_time=default_compute_time,  # 设置正向计算时间为默认值
                    forward_comm="NONE",  # 正向通信方式为 "NONE"（不进行通信）
                    forward_comm_size=0,  # 正向通信的大小为 0
                    backward_compute_time=default_compute_time,  # 设置反向计算时间为默认值
                    backward_comm="NONE",  # 反向通信方式为 "NONE"（不进行通信）
                    backward_comm_size=0,  # 反向通信的大小为 0
                    dp_compute_time=default_compute_time,  # 设置数据并行计算时间为默认值
                    dp_comm="REDUCESCATTER_DP_EP",  # 数据并行通信方式为 "REDUCESCATTER_DP_EP"
                    dp_comm_size=4 * moe_param_count,  # 数据并行通信的大小为 4 * MoE 参数数
                )
            )  # 结束 Work_Item 的构造并将其添加到工作负载列表中

            
        for _ in range(self.ga_num):  # 循环 `ga_num` 次，`ga_num` 是全局批次除以（微批次大小 * 数据并行数）后的值
            for layer in layers:  # 遍历模型中的每一层
                name = layer.layer_name  # 获取当前层的名称
                forward_comm = backward_comm = backward_comm_2 = "NONE"  # 初始化正向和反向通信类型为 "NONE"
                forward_comm_size = tp_comm_size  # 设置正向通信大小为 `tp_comm_size`
                backward_comm_size = tp_comm_size  # 设置反向通信大小为 `tp_comm_size`
                dp_comm = "NONE"  # 初始化数据并行通信类型为 "NONE"
                dp_comm_size = 0  # 初始化数据并行通信大小为 0

                if self.args.enable_sequence_parallel:  # 如果启用了序列并行
                    if "embedding" in name:  # 如果当前层名称包含 "embedding"
                        self.workload.append(  # 向工作负载列表中添加新的工作项
                            Work_Item(  # 创建一个新的 Work_Item 实例
                                name=name,  # 设置工作项名称
                                forward_compute_time=default_compute_time,  # 设置正向计算时间为默认值
                                forward_comm=forward_comm,  # 设置正向通信类型
                                forward_comm_size=forward_comm_size,  # 设置正向通信大小
                                backward_compute_time=default_compute_time,  # 设置反向计算时间为默认值
                                backward_comm=backward_comm,  # 设置反向通信类型
                                backward_comm_size=backward_comm_size,  # 设置反向通信大小
                                dp_compute_time=backward_compute_time,  # 设置数据并行计算时间
                                dp_comm=dp_comm,  # 设置数据并行通信类型
                                dp_comm_size=dp_comm_size,  # 设置数据并行通信大小
                            )
                        )  # 结束 Work_Item 构造并将其添加到工作负载列表

                    if "row" in name:  # 如果当前层名称包含 "row"
                        if self.args.recompute_activations and 'attention' in name:  # 如果启用了重计算激活并且是注意力层
                            forward_comm_size *= 2  # 将正向通信大小乘以 2
                        forward_comm = "REDUCESCATTER"  # 设置正向通信方式为 "REDUCESCATTER"
                        backward_comm = "ALLGATHER"  # 设置反向通信方式为 "ALLGATHER"
                        self.workload.append(  # 向工作负载列表中添加新的工作项
                            Work_Item(  # 创建一个新的 Work_Item 实例
                                name=name,  # 设置工作项名称
                                forward_compute_time=default_compute_time,  # 设置正向计算时间为默认值
                                forward_comm=forward_comm,  # 设置正向通信类型
                                forward_comm_size=forward_comm_size,  # 设置正向通信大小
                                backward_compute_time=default_compute_time,  # 设置反向计算时间为默认值
                                backward_comm="NONE",  # 设置反向通信类型
                                backward_comm_size=tp_comm_size,  # 设置反向通信大小
                                dp_compute_time=default_compute_time,  # 设置数据并行计算时间为默认值
                                dp_comm=dp_comm,  # 设置数据并行通信类型
                                dp_comm_size=dp_comm_size,  # 设置数据并行通信大小
                            )
                        )  # 结束 Work_Item 构造并将其添加到工作负载列表

                    if "column" in name:  # 如果当前层名称包含 "column"
                        if self.args.recompute_activations and 'attention' in name:  # 如果启用了重计算激活并且是注意力层
                            forward_comm_size *= 2  # 将正向通信大小乘以 2
                        forward_comm = "ALLGATHER"  # 设置正向通信方式为 "ALLGATHER"
                        forward_comm2 = "NONE"  # 初始化第二种正向通信方式为 "NONE"
                        backward_comm = "REDUCESCATTER"  # 设置反向通信方式为 "REDUCESCATTER"
                        backward_comm_2 = "ALLGATHER"  # 设置第二种反向通信方式为 "ALLGATHER"
                        self.workload.append(  # 向工作负载列表中添加新的工作项
                            Work_Item(  # 创建一个新的 Work_Item 实例
                                name=name,  # 设置工作项名称
                                forward_compute_time=default_compute_time,  # 设置正向计算时间为默认值
                                forward_comm=forward_comm,  # 设置正向通信类型
                                forward_comm_size=forward_comm_size,  # 设置正向通信大小
                                backward_compute_time=default_compute_time,  # 设置反向计算时间为默认值
                                backward_comm="NONE",  # 设置反向通信类型
                                backward_comm_size=0,  # 设置反向通信大小为 0
                                dp_compute_time=default_compute_time,  # 设置数据并行计算时间为默认值
                                dp_comm=dp_comm,  # 设置数据并行通信类型
                                dp_comm_size=dp_comm_size,  # 设置数据并行通信大小
                            )
                        )  # 结束 Work_Item 构造并将其添加到工作负载列表

                    if "moelayer" in name:  # 如果当前层名称包含 "moelayer"
                        forward_comm1 = "ALLGATHER"  # 设置正向通信方式 1 为 "ALLGATHER"
                        forward_comm2 = "ALLTOALL"  # 设置正向通信方式 2 为 "ALLTOALL"
                        forward_comm3 = "ALLTOALL_EP"  # 设置正向通信方式 3 为 "ALLTOALL_EP"
                        forward_comm4 = "ALLGATHER"  # 设置正向通信方式 4 为 "ALLGATHER"
                        forward_comm5 = "REDUCESCATTER"  # 设置正向通信方式 5 为 "REDUCESCATTER"
                        forward_comm6 = "ALLTOALL_EP"  # 设置正向通信方式 6 为 "ALLTOALL_EP"
                        forward_comm7 = "ALLTOALL"  # 设置正向通信方式 7 为 "ALLTOALL"
                        self.workload.append(  # 向工作负载列表中添加新的工作项
                            Work_Item(  # 创建一个新的 Work_Item 实例
                                name=name,  # 设置工作项名称
                                forward_compute_time=default_compute_time,  # 设置正向计算时间为默认值
                                forward_comm=forward_comm1,  # 设置正向通信方式为 `forward_comm1`
                                forward_comm_size=2 * self.seq_len * self.num_experts,  # 设置正向通信大小
                                backward_compute_time=default_compute_time,  # 设置反向计算时间为默认值
                                backward_comm=forward_comm1,  # 设置反向通信方式为 `forward_comm1`
                                backward_comm_size=2 * self.seq_len * self.num_experts,  # 设置反向通信大小
                                dp_compute_time=default_compute_time,  # 设置数据并行计算时间为默认值
                                dp_comm=dp_comm,  # 设置数据并行通信类型
                                dp_comm_size=dp_comm_size,  # 设置数据并行通信大小
                            )
                        )  # 结束 Work_Item 构造并将其添加到工作负载列表
                        self.workload.append(  # 向工作负载列表中添加新的工作项
                            Work_Item(  # 创建一个新的 Work_Item 实例
                                name=name,  # 设置工作项名称
                                forward_compute_time=default_compute_time,  # 设置正向计算时间为默认值
                                forward_comm=forward_comm2,  # 设置正向通信方式为 `forward_comm2`
                                forward_comm_size=tp_comm_size // self.tp,  # 设置正向通信大小
                                backward_compute_time=default_compute_time,  # 设置反向计算时间为默认值
                                backward_comm=forward_comm2,  # 设置反向通信方式为 `forward_comm2`
                                backward_comm_size=tp_comm_size // self.tp,  # 设置反向通信大小
                                dp_compute_time=default_compute_time,  # 设置数据并行计算时间为默认值
                                dp_comm=dp_comm,  # 设置数据并行通信类型
                                dp_comm_size=dp_comm_size,  # 设置数据并行通信大小
                            )
                        )  # 结束 Work_Item 构造并将其添加到工作负载列表
                        # 后续类似的处理方式用于构造其他通信方式的工作项
                        
                else:  # 如果没有启用序列并行
                    forward_comm = "ALLREDUCE"  # 设置正向通信方式为 "ALLREDUCE"
                    backward_comm = "ALLREDUCE"  # 设置反向通信方式为 "ALLREDUCE"
                    if self.args.recompute_activations and 'attention' in name:  # 如果启用了重计算激活并且是注意力层
                        forward_comm_size *= 2  # 将正向通信大小乘以 2
                    if "embedding" in name:  # 如果当前层名称包含 "embedding"
                        self.workload.append(  # 向工作负载列表中添加新的工作项
                            Work_Item(  # 创建一个新的 Work_Item 实例
                                name=name,  # 设置工作项名称
                                forward_compute_time=default_compute_time,  # 设置正向计算时间为默认值
                                forward_comm=forward_comm,  # 设置正向通信类型
                                forward_comm_size=forward_comm_size,  # 设置正向通信大小
                                backward_compute_time=default_compute_time,  # 设置反向计算时间为默认值
                                backward_comm=backward_comm,  # 设置反向通信类型
                                backward_comm_size=backward_comm_size,  # 设置反向通信大小
                                dp_compute_time=backward_compute_time,  # 设置数据并行计算时间为默认值
                                dp_comm=dp_comm,  # 设置数据并行通信类型
                                dp_comm_size=dp_comm_size,  # 设置数据并行通信大小
                            )
                        )  # 结束 Work_Item 构造并将其添加到工作负载列表
                    else:  # 其他情况（非 embedding 层）
                        self.workload.append(  # 向工作负载列表中添加新的工作项
                            Work_Item(  # 创建一个新的 Work_Item 实例
                                name=name,  # 设置工作项名称
                                forward_compute_time=default_compute_time,  # 设置正向计算时间为默认值
                                forward_comm=forward_comm,  # 设置正向通信类型
                                forward_comm_size=forward_comm_size,  # 设置正向通信大小
                                backward_compute_time=default_compute_time,  # 设置反向计算时间为默认值
                                backward_comm=backward_comm,  # 设置反向通信类型
                                backward_comm_size=backward_comm_size,  # 设置反向通信大小
                                dp_compute_time=default_compute_time,  # 设置数据并行计算时间为默认值
                                dp_comm=dp_comm,  # 设置数据并行通信类型
                                dp_comm_size=dp_comm_size,  # 设置数据并行通信大小
                            )
                        )  # 结束 Work_Item 构造并将其添加到工作负载列表
            self.workload.append(  # 向工作负载列表中添加 "embedding_norm" 工作项
                Work_Item(  # 创建 "embedding_norm" 工作项
                    name="embedding_norm",  # 设置工作项名称为 "embedding_norm"
                    forward_compute_time=default_compute_time,  # 设置正向计算时间为默认值
                    forward_comm="ALLREDUCE",  # 设置正向通信方式为 "ALLREDUCE"
                    forward_comm_size=self.args.vocab_size * self.args.hidden_size * 2,  # 设置正向通信大小
                    backward_compute_time=default_compute_time,  # 设置反向计算时间为默认值
                    backward_comm="NONE",  # 设置反向通信方式为 "NONE"
                    backward_comm_size=0,  # 设置反向通信大小为 0
                    dp_compute_time=default_compute_time,  # 设置数据并行计算时间为默认值
                    dp_comm="NONE",  # 设置数据并行通信方式为 "NONE"
                    dp_comm_size=0,  # 设置数据并行通信大小为 0
                )
            )  # 结束 Work_Item 构造并将其添加到工作负载列表

            
                
        for i in range(3):  # 循环 3 次，用于创建 3 个交叉熵（cross_entropy）工作项
            self.workload.append(  # 向工作负载列表中添加新的工作项
                Work_Item(  # 创建一个新的 Work_Item 实例
                    name="cross_entropy" + str(i + 1),  # 设置工作项名称为 "cross_entropy" 后跟序号（1, 2, 3）
                    forward_compute_time=compute_time,  # 设置正向计算时间为 `compute_time`
                    forward_comm="ALLREDUCE",  # 设置正向通信方式为 "ALLREDUCE"
                    forward_comm_size=self.args.seq_length * self.args.micro_batch * 4,  # 设置正向通信大小为 `seq_length` * `micro_batch` * 4
                    backward_compute_time=compute_time,  # 设置反向计算时间为 `compute_time`
                    backward_comm="NONE",  # 设置反向通信方式为 "NONE"
                    backward_comm_size=0,  # 设置反向通信大小为 0
                    dp_compute_time=compute_time,  # 设置数据并行计算时间为 `compute_time`
                    dp_comm="NONE",  # 设置数据并行通信方式为 "NONE"
                    dp_comm_size=0,  # 设置数据并行通信大小为 0
                )
            )  # 结束 Work_Item 构造并将其添加到工作负载列表

        for i in range(4):  # 循环 4 次，用于创建 4 个优化器（optimizer）工作项
            self.workload.append(  # 向工作负载列表中添加新的工作项
                Work_Item(  # 创建一个新的 Work_Item 实例
                    name="optimizer" + str(i + 1),  # 设置工作项名称为 "optimizer" 后跟序号（1, 2, 3, 4）
                    forward_compute_time=compute_time,  # 设置正向计算时间为 `compute_time`
                    forward_comm="ALLREDUCE",  # 设置正向通信方式为 "ALLREDUCE"
                    forward_comm_size=4,  # 设置正向通信大小为 4
                    backward_compute_time=compute_time,  # 设置反向计算时间为 `compute_time`
                    backward_comm="NONE",  # 设置反向通信方式为 "NONE"
                    backward_comm_size=0,  # 设置反向通信大小为 0
                    dp_compute_time=compute_time,  # 设置数据并行计算时间为 `compute_time`
                    dp_comm="NONE",  # 设置数据并行通信方式为 "NONE"
                    dp_comm_size=0,  # 设置数据并行通信大小为 0
                )
            )  # 结束 Work_Item 构造并将其添加到工作负载列表

    def dump_file(self, filename):  # 定义一个 `dump_file` 函数，用于将工作负载信息保存到文件
        filename = filename + ".txt"  # 给文件名添加 ".txt" 后缀
        with open(filename, "w") as f:  # 打开文件以写入（"w" 模式表示写入）
            f.write((  # 写入模型配置信息
                f"HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: {self.args.tensor_model_parallel_size} "
                f"ep: {self.args.expert_model_parallel_size} "
                f"pp: {self.args.pipeline_model_parallel} "
                f"vpp: {self.args.num_layers} "
                f"ga: {self.ga_num} all_gpus: {self.args.world_size} "
                f"checkpoints: 0 checkpoint_initiates: 0"
            ) + "\n")  # 每个配置项之间以空格分隔，末尾加换行符

            f.write(str(len(self.workload)) + "\n")  # 写入工作负载的总数量（`self.workload` 列表的长度）
            for item in self.workload:  # 遍历每个工作项
                f.write(  # 写入每个工作项的属性
                    "\t".join([str(getattr(item, k)) for k in item.__dict__.keys()])  # 获取工作项每个属性的值，并将它们用制表符连接
                    + "\n"  # 每个工作项的属性写在一行
                )


class simAI_MicroTest:  # 定义一个类 `simAI_MicroTest`
    def __init__(self, args):  # 初始化函数，接受 `args` 参数
        self.args = args  # 将传入的 `args` 参数保存为实例变量
        self.workload = []  # 初始化一个空列表 `workload`，用于存储工作项

    def _simAI_microtest_convert(self, comm_type):  # 定义一个转换函数，根据通信类型转换为标准格式
        if comm_type == "all_reduce" or comm_type == "allreduce":  # 如果通信类型为 "all_reduce" 或 "allreduce"
            return "ALLREDUCE"  # 返回标准的 "ALLREDUCE"
        elif comm_type == "all_gather" or comm_type == "allgather":  # 如果通信类型为 "all_gather" 或 "allgather"
            return "ALLGATHER"  # 返回标准的 "ALLGATHER"
        elif comm_type == "reduce_scatter" or comm_type == "reducescatter":  # 如果通信类型为 "reduce_scatter" 或 "reducescatter"
            return "REDUCESCATTER"  # 返回标准的 "REDUCESCATTER"
        elif comm_type == "all_to_all" or comm_type == "alltoall":  # 如果通信类型为 "all_to_all" 或 "alltoall"
            return "ALLTOALL"  # 返回标准的 "ALLTOALL"
        else:  # 如果没有匹配的通信类型
            return  # 返回 None，表示没有匹配的类型

    def workload_generator(self):  # 定义一个生成工作负载的函数
        curr_size = self.args.begin_size  # 从 `args.begin_size` 开始设置当前大小
        default_compute_time = 1  # 设置默认计算时间为 1
        while curr_size <= self.args.end_size:  # 当当前大小小于等于 `args.end_size` 时，生成工作负载
            self.workload.append(  # 向 `workload` 列表中添加新的工作项
                Work_Item(  # 创建一个新的 `Work_Item` 实例
                    name="micro_test",  # 设置工作项名称为 "micro_test"
                    forward_compute_time=default_compute_time,  # 设置正向计算时间为默认计算时间（1）
                    forward_comm="NONE",  # 设置正向通信方式为 "NONE"
                    forward_comm_size=0,  # 设置正向通信大小为 0
                    backward_compute_time=default_compute_time,  # 设置反向计算时间为默认计算时间（1）
                    backward_comm="NONE",  # 设置反向通信方式为 "NONE"
                    backward_comm_size=0,  # 设置反向通信大小为 0
                    dp_compute_time=default_compute_time,  # 设置数据并行计算时间为默认计算时间（1）
                    dp_comm=self._simAI_microtest_convert(self.args.test_comm),  # 将通信类型转换为标准格式并设置
                    dp_comm_size=curr_size,  # 设置数据并行通信大小为当前大小
                    process_time=1,  # 设置处理时间为 1
                )
            )
            curr_size *= 2  # 将当前大小翻倍，生成不同规模的工作负载

    def dump_file(self, filename):  # 定义一个将工作负载写入文件的函数
        filename = filename + ".txt"  # 给文件名添加 ".txt" 后缀
        with open(filename, "w") as f:  # 打开文件以写入（"w" 模式表示写入）
            if not self.args.multi_all_reduce_enable:  # 如果 `multi_all_reduce_enable` 为 False
                f.write(f"MICRO" + "\n")  # 写入 "MICRO" 字符串并换行
                f.write(str(len(self.workload)) + "\n")  # 写入工作负载的总数量（`self.workload` 列表的长度）
                for item in self.workload:  # 遍历每个工作项
                    f.write(  # 写入每个工作项的属性
                        "\t".join([str(getattr(item, k)) for k in item.__dict__.keys()])  # 获取工作项每个属性的值，并将它们用制表符连接
                        + "\n"  # 每个工作项的属性写在一行
                    )
            else:  # 如果 `multi_all_reduce_enable` 为 True
                f.write(  # 写入模型配置信息
                    f"HYBRID_TRANSFORMER_FWD_IN_BCKWD	model_parallel_NPU_group: {self.args.tensor_model_parallel_size} \
                        expert_parallel_npu_group: {self.args.expert_model_parallel_size} pp: {self.args.pipeline_model_parallel} \
                        ga: {self.ga_num} all_gpus: {self.args.world_size} checkpoints: 0 checkpoint_initiates: 0"
                    + "\n"
                )
                f.write(str(len(self.workload)) + "\n")  # 写入工作负载的总数量
                for item in self.workload:  # 遍历每个工作项
                    f.write(  # 写入每个工作项的属性
                        "\t".join([str(getattr(item, k)) for k in item.__dict__.keys()])  # 获取工作项每个属性的值，并将它们用制表符连接
                        + "\n"  # 每个工作项的属性写在一行
                    )



if __name__ == "__main__":  # 如果当前脚本是主程序入口
    args = get_params()  # 获取命令行参数或配置文件中的参数
    print(args)  # 打印参数，用于调试或验证
    model = MegatronModel(args)  # 使用获取的参数创建 Megatron 模型
    result_dir = "results/workload/"  # 设定结果文件保存目录
    if not os.path.isdir(result_dir):  # 如果结果目录不存在
        os.makedirs(result_dir)  # 创建目录
    filename = f"{args.gpu_type}-{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-gbs{args.global_batch}-mbs{args.micro_batch}-seq{args.seq_length}-MOE-{args.moe_enable}-GEMM-{args.moe_grouped_gemm}-flash_attn-{args.use_flash_attn}"  # 根据参数生成文件名
    filepath = os.path.join(result_dir, filename)  # 将结果目录和文件名合并成完整路径
    params = model.parameters()  # 获取模型参数
    # work = SIMAI_workload(model, args, GPU_Tensor_core.A100, "gpt13B")  # 注释掉的部分：创建工作负载生成器对象，暂时不使用
    # name_layers = work.workload_generate()  # 注释掉的部分：生成工作负载，暂时不使用
    # work.dump_file("test")  # 注释掉的部分：保存工作负载，暂时不使用
    print(sum(p.numel() for p in params))  # 输出模型的参数总数（参数数量），即模型的大小
    
    if args.aiob_enable:  # 如果启用了 AIOB（异步I/O加速）
        params = model.parameters()  # 获取模型参数
        args.model_param = sum(p.numel() for p in params)  # 将模型的参数总数存储在 args 中
        if args.comp_filepath == None:  # 如果没有提供计算文件路径
            comp_filepath = get_comp_out(args)  # 获取计算输出路径
            compute_cache = extract_averages(comp_filepath, args)  # 从计算文件中提取平均值，得到计算缓存
        else:  # 如果提供了计算文件路径
            print("comp_filepath:", args.comp_filepath)  # 打印提供的计算文件路径
            comp_filepath = args.comp_filepath  # 使用提供的计算文件路径
            compute_cache = extract_averages(comp_filepath, args)  # 提取计算缓存

        print("compute_cache = {")  # 打印计算缓存的键值对
        for key, value in compute_cache.items():  # 遍历计算缓存字典
            print(f"    '{key}' : {value},")  # 打印每个键值对
        print("}")  # 结束计算缓存的打印
        work = SIMAI_workload(  # 创建工作负载生成器对象，并传入模型和参数
            model, args, compute_cache  # 传递计算缓存参数
        )
        name_layers = work.workload_generate_aiob()  # 使用 AIOB 模式生成工作负载
        work.dump_file(filepath)  # 保存生成的工作负载到文件
        print("workload save in :", filepath)  # 打印工作负载保存路径
    else:  # 如果没有启用 AIOB
        work = SIMAI_workload(model, args, None)  # 创建工作负载生成器对象，但不传递计算缓存
        name_layers = work.workload_generate()  # 生成工作负载
        work.dump_file(filepath)  # 保存工作负载到文件
        print(f"workload save in : {filepath}.txt")  # 打印工作负载保存路径，附加文件后缀 .txt

