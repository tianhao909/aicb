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
版权 (c) 2021，阿里巴巴集团；  
根据 Apache 许可证，版本 2.0（“许可证”）授权；  
除非遵守本许可证的规定，否则您不得使用此文件。  
您可以在以下网址获得许可证的副本：  
   http://www.apache.org/licenses/LICENSE-2.0  
除非适用法律要求或书面协议另有约定，软件按“原样”提供，不附任何类型的明示或暗示的担保或条件。  
请参阅许可证，以了解有关许可权限和限制的具体语言。
"""

from typing import List, Dict  # 导入 List 和 Dict 类型注解
import pandas as pd  # 导入pandas库，通常用于数据处理
import pickle  # 导入pickle库，用于序列化和反序列化
from enum import Enum  # 导入Enum类，定义枚举类型
import argparse  # 导入argparse模块，命令行解析
import sys  # 导入sys模块，系统相关操作
import time  # 导入time模块，时间相关操作
import os  # 导入os模块，操作系统相关功能
import json  # 导入json模块，用于JSON数据处理
from collections import defaultdict  # 导入defaultdict，字典的子类，支持默认值
import math  # 导入math模块，进行数学计算
import re  # 导入正则表达式模块，用于字符串处理

try:
    import torch  # 尝试导入PyTorch库
except ImportError as e:  # 如果导入失败
    torch = None  # 设置torch为None
    print("Failed to import 'torch'.")  # 输出错误信息

def generate_masked_orthogonal_rank_groups(  # 定义函数生成掩码正交排名组
    world_size: int,  # world_size：全球进程数
    parallel_size: List[int],  # parallel_size：并行尺寸列表，表示各个并行维度的大小
    mask: List[bool],  # mask：掩码列表，决定哪些并行方法被包含在生成的组中
) -> List[List[int]]:  # 返回一个包含多个列表的列表，每个子列表是一个排名组

    """Generate orthogonal parallel groups based on the parallel size and mask.
    
    该函数根据并行尺寸和掩码生成正交并行组

    Arguments:
        world_size (int): world size

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example, 
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then 
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the 
            generated group is the `pp` group.
            
    根据并行尺寸和掩码生成正交并行组

    **参数：**

    - `world_size` (int): 世界大小

    - `parallel_size` (List[int]):  
    每种正交并行类型的并行大小。例如，如果 tensor_parallel_size = 2，pipeline_model_parallel_group = 3，data_parallel_size = 4，
    并且并行映射顺序是 tp-pp-dp，那么 parallel_size = [2, 3, 4]。

    - `mask` (List[bool]):  
    掩码控制生成的组所表示的并行方法。如果 `mask[i]` 为 True，表示生成的组包含第 i 种并行方法。
    例如，如果 parallel_size = [tp_size, pp_size, dp_size]，且 mask = [True, False, True]，那么生成的组是 `tp-dp` 组；
    如果 mask = [False, True, False]，则生成的组是 `pp` 组。
    """

    def prefix_product(a: List[int], init=1) -> List[int]:  # 定义前缀积函数，计算列表的累积乘积
        r = [init]  # 初始化结果列表，包含一个初始值
        for v in a:  # 遍历列表a中的每个值
            init = init * v  # 累积乘积
            r.append(init)  # 将当前乘积值加入结果列表
        return r  # 返回前缀积列表

    def inner_product(a: List[int], b: List[int]) -> int:  # 定义内积函数，计算两个列表的内积
        return sum([x * y for x, y in zip(a, b)])  # 计算内积：将两个列表对应元素相乘并求和

    def decompose(index, shape, stride=None):  # 定义分解函数，解构给定索引为多个维度索引
        ''' 
        This function solve the math problem below:
            There is an equation: 
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        '''
        if stride is None:  # 如果没有给定步长，则使用前缀积
            stride = prefix_product(shape)  # 计算步长（前缀积）
        idx = [(index // d) % s for s, d in zip(shape, stride)]  # 使用整除和取余来解构索引
        # stride 是前缀积的结果。最后一个stride值没有使用。
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)  # 确保计算结果与输入一致
        return idx  # 返回解构后的索引列表

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]  # 通过掩码提取出需要的并行尺寸（masked部分）
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]  # 提取不需要的并行尺寸（unmasked部分）

    global_stride = prefix_product(parallel_size)  # 计算全局步长
    masked_stride = [d for d, m in zip(global_stride, mask) if m]  # 提取被掩码部分的步长
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]  # 提取未掩码部分的步长

    group_size = prefix_product(masked_shape)[-1]  # 计算一个组的大小，即masked部分的前缀积的最后值
    num_of_group = world_size // group_size  # 计算组的数量

    ranks = []  # 初始化排名列表
    for group_index in range(num_of_group):  # 遍历每个组
        # 获取未掩码部分的解构索引
        decomposed_group_idx = decompose(group_index, unmasked_shape)  
        rank = []  # 初始化当前组的排名列表
        for rank_in_group in range(group_size):  # 遍历组内的每个排名
            # 获取掩码部分的解构索引
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)  
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)  # 计算掩码部分的内积
                + inner_product(decomposed_group_idx, unmasked_stride)  # 计算未掩码部分的内积
            )
        ranks.append(rank)  # 将当前组的排名加入总排名列表
    return ranks  # 返回所有组的排名列表


class RankGenerator(object):  # 定义 RankGenerator 类，用于生成并行计算的排名组
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> None:  # 构造函数，初始化类的各项参数
        self.tp = tp  # 初始化tensor并行维度（tp）大小
        self.ep = ep  # 初始化专家并行维度（ep）大小
        self.dp = dp  # 初始化数据并行维度（dp）大小
        self.pp = pp  # 初始化流水线并行维度（pp）大小
        self.cp = cp  # 初始化计算并行维度（cp）大小
        self.world_size = tp * dp * pp * cp  # 计算全局进程数，即所有并行维度的乘积

        # 初始化一个字典，将并行维度名称映射到其对应的大小
        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.order = order  # 设置并行维度的顺序
        order = order.lower()  # 将顺序转为小写形式

        if 'ep' in order:  # 如果顺序中包含专家并行（ep）
            # 验证ep和dp必须是相邻的
            if 'ep-dp' not in order and 'dp-ep' not in order:
                raise RuntimeError(f"The ep and dp must be adjacent in order ({self.order}).")

        # 验证所有并行维度是否在顺序中出现，如果未出现且维度大小不为1，抛出异常
        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:  # 如果维度名称不在顺序中，则将其添加到顺序末尾
                order = order + '-' + name

        self.order_w_ep = order  # 存储包含ep的并行顺序
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])  # 存储不包含ep的并行顺序
        self.ordered_size_wo_ep = []  # 初始化不包含ep的并行维度的尺寸列表
        self.ordered_size_w_ep = []  # 初始化包含ep的并行维度的尺寸列表

        # 根据并行顺序填充尺寸列表
        for token in order.split('-'):
            if token == 'dp':  # 如果是数据并行（dp）
                self.ordered_size_w_ep.append(self.dp // self.ep)  # 如果包含ep，则数据并行的尺寸为 dp // ep
                self.ordered_size_wo_ep.append(self.dp)  # 不包含ep时，数据并行的尺寸为dp
            elif token == 'ep':  # 如果是专家并行（ep）
                self.ordered_size_w_ep.append(self.ep)  # 添加ep的尺寸
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])  # 否则，按照名称映射添加对应尺寸
                self.ordered_size_wo_ep.append(self.name_to_size[token])  # 不包含ep时，添加相同尺寸

    def get_mask(self, order: str, token: str):  # 获取掩码，用于控制并行方法的选择
        ordered_token = order.split('-')  # 将顺序字符串拆分成单个维度
        token = token.split('-')  # 将输入token字符串拆分成单个并行维度
        mask = [False] * len(ordered_token)  # 初始化一个全为False的掩码列表，大小等于并行顺序的长度
        for t in token:  # 遍历输入的token维度
            mask[ordered_token.index(t)] = True  # 将对应的维度位置标记为True，表示该维度参与
        return mask  # 返回掩码列表

    def get_ranks(self, token, independent_ep=False):  # 获取并行维度的排名组
        '''Get rank group by input token.

        Arguments:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.

            independent_ep (bool: True):
                This flag controls whether we treat EP and DP independently.
                EP shares ranks with DP, if we want to get ranks related to
                EP, we should set the flag. For example, get_ranks('dp', True)
                will get DP modulo EP group, and get_ranks('dp', False) will
                get full DP group.
                
        根据输入的 token 获取排名组

        **参数：**

        - `token` (str):  
        指定要获取的排名类型。如果我们想获取多个并行类型，可以使用连字符 `-` 来分隔它们。
        例如，如果我们想获取 TP_DP 组，token 应该是 `'tp-dp'`。

        - `independent_ep` (bool: True):  
        该标志控制是否将 EP 和 DP 独立处理。EP 与 DP 共享排名，如果我们想获取与 EP 相关的排名，应设置此标志。
        例如，`get_ranks('dp', True)` 会获取与 EP 相关的 DP 组，`get_ranks('dp', False)` 会获取完整的 DP 组。
        '''
        if independent_ep:  # 如果独立处理ep
            parallel_size = self.ordered_size_w_ep  # 使用包含ep的并行尺寸列表
            order = self.order_w_ep  # 使用包含ep的并行顺序
        else:  # 否则使用不包含ep的设置
            parallel_size = self.ordered_size_wo_ep  # 使用不包含ep的并行尺寸列表
            order = self.order_wo_ep  # 使用不包含ep的并行顺序
        mask = self.get_mask(order, token)  # 获取并行掩码
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, parallel_size, mask)  # 生成并行组的排名
        return ranks  # 返回生成的排名组

def gelu_impl(x):  # 定义gelu实现函数
    """OpenAI's gelu implementation."""  # OpenAI的gelu实现
    return (
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))  # GELU函数的数学公式
    )


def openai_gelu(x):  # 定义OpenAI版本的gelu函数
    return gelu_impl(x)  # 调用gelu_impl函数返回结果


def erf_gelu(x):  # 定义erf版本的gelu实现
    return (
        x
        * 0.5
        * (
            torch.erf(x / 1.41421).to(dtype=x.dtype)  # 计算x的误差函数并转换为x的dtype类型
            + torch.ones_like(x).to(dtype=x.dtype)  # 加上1，确保输出的类型和x相同
        )
    )


def Comp_with_aiob(workload, compute_cache):  # 定义与aiob的计算合并函数
    for item in workload.workload:  # 遍历工作负载中的每个项
        if item.comm_type == CommType.computation:  # 如果该项的通信类型是计算
            for key in compute_cache:  # 遍历计算缓存
                key_temp = key.split("_")[0]  # 获取缓存键的第一个部分
                if key_temp in item.stage:  # 如果缓存键的第一部分出现在当前项的stage中
                    item.msg_size = compute_cache[key]  # 设置当前项的消息大小为缓存中的对应值
                    break  # 跳出内层循环
    return workload  # 返回修改后的工作负载



def get_comp_out(args):  # 定义获取计算输出的函数，参数是args，包含配置和参数
    vocab_size = args.vocab_size  # 获取词汇表大小
    batch_size = args.micro_batch  # 获取微批次大小
    seq_len = args.seq_length  # 获取序列长度
    tp = args.tensor_model_parallel_size  # 获取tensor模型并行度（tp）
    vocab_size = args.padded_vocab_size  # 获取填充后的词汇表大小
    if "Megatron" in args.frame:  # 如果框架类型是Megatron
        device = torch.cuda.current_device()  # 获取当前设备，通常为GPU
        from workload_generator.mocked_model.AiobMegatron import MegatronModel  # 导入Megatron模型

        measure_model = MegatronModel(args)  # 创建Megatron模型实例
        measure_model.train()  # 设置模型为训练模式
        if args.dtype == "bfloat16":  # 如果数据类型是bfloat16
            dtype = torch.bfloat16  # 设置数据类型为bfloat16
        elif args.dtype == "float16":  # 如果数据类型是float16
            dtype = torch.float16  # 设置数据类型为float16
        else:  # 否则默认为float32
            dtype = torch.float32
        # total_input_1 = torch.rand(args.seq_len,
        #                                       args.batch_size,
        #                                       args.hidden_size,
        #                                       device=device).to(dtype)
        # 上面是一个注释掉的代码，原本是生成一个随机输入张量
        masked_input = torch.randint(  # 生成一个随机整数输入张量
            0,
            math.ceil(vocab_size / tp),  # 随机整数的上限是词汇表大小除以tp
            (batch_size, seq_len),  # 生成大小为(batch_size, seq_len)的张量
            device=device,  # 张量生成在当前设备（GPU）
            dtype=torch.int64,  # 设置张量数据类型为int64
        )
        filepath = measure_model(masked_input)  # 将生成的输入传递给模型并获取输出文件路径
        return filepath  # 返回文件路径

def extract_averages(file_path, args):  # 定义提取平均值的函数，参数是文件路径和配置
    attention_avg_sum = 0.0  # 初始化attention平均值总和
    mlp_avg_sum = 0.0  # 初始化mlp平均值总和
    other_avgs = {}  # 用于存储其他部分的平均值
    grad_forward = 0.0  # 初始化梯度前向传播时间
    grad_backward = 0.0  # 初始化梯度反向传播时间

    section_header_re = re.compile(r"^(\w+):")  # 正则表达式，匹配章节头部（如param_time:）
    time_gpu_avg_re = re.compile(r"time_gpu_avg:\s+(\d+(\.\d+)?)")  # 正则表达式，匹配time_gpu_avg字段
    time_gpu_min_re = re.compile(r"time_gpu_min:\s+(\d+(\.\d+)?)")  # 正则表达式，匹配time_gpu_min字段

    with open(file_path, "r") as file:  # 打开文件进行读取
        current_section = None  # 初始化当前章节变量

        for line in file:  # 遍历文件的每一行
            header_match = section_header_re.match(line)  # 匹配章节头部
            if header_match:  # 如果匹配到章节头部
                current_section = header_match.group(1).strip()  # 获取章节名称并去掉两端空格

            avg_match = time_gpu_avg_re.search(line)  # 匹配time_gpu_avg字段
            min_match = time_gpu_min_re.search(line)  # 匹配time_gpu_min字段
            if current_section == "param_time":  # 如果当前章节是param_time
                if min_match:  # 如果匹配到time_gpu_min
                    grad_forward = float(min_match.group(1)) * 1000  # 转换为微秒（us）
                if avg_match:  # 如果匹配到time_gpu_avg
                    grad_backward = float(avg_match.group(1)) * 1000  # 转换为微秒（us）
            elif avg_match and current_section:  # 如果匹配到平均值并且当前章节有效
                avg_value = float(avg_match.group(1)) * 1000  # 转换为微秒（us）
                if "atten" in current_section or current_section == "layernorm":  # 如果当前章节是attention或者layernorm
                    if args.recompute_activations and 'flash' in current_section:  # 如果需要重计算激活并且当前章节包含'flash'
                        attention_avg_sum += avg_value * 2  # 对attention的平均值进行加倍
                    else:
                        attention_avg_sum += avg_value  # 否则正常累加
                elif "mlp" in current_section or current_section == "layernorm2":  # 如果当前章节是mlp或者layernorm2
                    mlp_avg_sum += avg_value  # 累加mlp的平均值
                else:  # 其他章节
                    other_avgs[current_section] = avg_value  # 存储其他章节的平均值

    # 四舍五入并转换为整数
    attention_forward = round(attention_avg_sum)  # 对attention前向传播时间取整
    attention_backward = attention_forward  # 将attention反向传播时间设置为与前向传播相同
    mlp_forward = round(mlp_avg_sum)  # 对mlp前向传播时间取整
    mlp_backward = mlp_forward  # 将mlp反向传播时间设置为与前向传播相同
    grad_backward = round(grad_backward)  # 对梯度反向传播时间取整
    grad_forward = round(grad_forward)  # 对梯度前向传播时间取整
    other_avgs_int = {k: round(v) for k, v in other_avgs.items() if k != "param_time"}  # 四舍五入并转换为整数，排除param_time

    a100_compute_cache = {  # 构建a100计算缓存字典，存储计算结果
        "attention_forward": attention_forward,
        "attention_backward": attention_backward,
        "mlp_forward": mlp_forward,
        "mlp_backward": mlp_backward,
        "grad_forward": grad_forward,
        "grad_backward": grad_backward,
    }
    a100_compute_cache.update(other_avgs_int)  # 更新其他章节的计算缓存

    return a100_compute_cache  # 返回计算缓存字典

def process_all_keys(input_file):  # 定义处理所有键的函数，参数是输入文件路径
    with open(input_file, "r") as file:  # 打开文件进行读取
        first_line_str = file.readline().strip()  # 读取第一行并去除空格
        remaining_content = file.read().strip()  # 读取文件其余内容并去除空格
    # 尝试修复和构建合法的 JSON 字符串
    corrected_content = remaining_content.replace("}{", "},{").replace("]}{", "]},{")  # 修复文件中的JSON格式问题

    # 构建 JSON 数组
    json_str = f"[{corrected_content}]"  # 将修复后的内容包裹成一个JSON数组

    try:
        data = json.loads(json_str)  # 尝试加载修复后的JSON字符串

        processed_results = defaultdict(lambda: defaultdict(list))  # 使用defaultdict嵌套列表用于存储结果
        for entry in data:  # 遍历数据中的每一项
            for key, measurements in entry.items():  # 遍历每一项的key和对应的测量值
                for measure in measurements:  # 遍历测量值
                    measure_key, measure_value = next(iter(measure.items()))  # 获取测量的key和value
                    if "time_gpu" in measure_key:  # 如果key中包含time_gpu
                        processed_results[key]["time_gpu"].append(measure["time_gpu"])  # 存储time_gpu数据
                    else:  # 否则直接存储其他测量值
                        processed_results[key][measure_key] = measure_value

        for key, values in processed_results.items():  # 遍历处理后的结果
            if "time_gpu" in values:  # 如果包含time_gpu数据
                gpu_times = values["time_gpu"]  # 获取time_gpu的数据
                min_time_gpu = min(gpu_times)  # 获取最小的gpu时间
                gpu_times_filtered = [t for t in gpu_times if t <= 3 * min_time_gpu]  # 过滤掉超过3倍最小时间的数据
                values["time_gpu_max"] = max(gpu_times_filtered)  # 获取最大gpu时间
                values["time_gpu_min"] = min_time_gpu  # 获取最小gpu时间
                values["time_gpu_avg"] = sum(gpu_times_filtered) / len(gpu_times_filtered)  # 计算平均gpu时间
                del values["time_gpu"]  # 删除原始的time_gpu数据

        with open(input_file, "w") as outfile:  # 打开文件进行写入
            outfile.write(first_line_str + "\n")  # 写入第一行
            for key, values in processed_results.items():  # 遍历处理后的结果
                outfile.write(f"{key}:\n")  # 写入每个key
                for subkey, subvalue in values.items():  # 遍历每个subkey和subvalue
                    outfile.write(f"    {subkey}: {subvalue}\n")  # 写入subkey和subvalue
        print(f"Compute-results save in:{input_file}")  # 输出保存路径

    except json.JSONDecodeError as e:  # 捕获JSON解码错误
        print(f"Failed to decode JSON: {e}")  # 输出解码错误信息
        print("Invalid JSON content:\n", corrected_content)  # 输出修复前的无效JSON内容



def cuda_timing_decorator(func):  # 定义一个装饰器函数，接受一个函数作为参数
    def wrapper(*args, **kwargs):  # 装饰器内的包装函数，接受任意位置和关键字参数

        start_event = torch.cuda.Event(enable_timing=True)  # 创建一个CUDA事件，启用计时功能
        end_event = torch.cuda.Event(enable_timing=True)  # 创建另一个CUDA事件，启用计时功能

        start_event.record()  # 记录起始事件
        result = func(*args, **kwargs)  # 执行传入的函数，并返回结果
        end_event.record()  # 记录结束事件
        torch.cuda.synchronize()  # 等待CUDA操作完成，确保计时准确

        elapsed_time_ms = start_event.elapsed_time(end_event) * 1000  # 计算两个事件之间的时间差，单位为毫秒
        return result, elapsed_time_ms  # 返回函数执行结果和耗时

    return wrapper  # 返回包装后的函数

def get_aiob_path(args):  # 定义获取输出路径的函数，参数为配置对象args
    result_dir = "./results/aiob_outputs"  # 设置结果文件的目录路径
    if not os.path.isdir(result_dir):  # 如果目录不存在
        os.makedirs(result_dir)  # 创建该目录
    filename = f"{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-gbs{args.global_batch}-mbs{args.micro_batch}-seq{args.seq_length}-flash_attn-{args.use_flash_attn}.txt"  # 构建文件名，包含多个参数
    filepath = os.path.join(result_dir, filename)  # 将目录和文件名合并成完整的文件路径
    return filepath  # 返回生成的文件路径

def write_op(time_list, args):  # 定义写入操作的函数，参数为时间列表和配置对象
    filepath = get_aiob_path(args)  # 调用get_aiob_path函数获取文件路径
    with open(filepath, "w") as file:  # 打开文件进行写入
        file.write(f"train_iter:{args.epoch_num}\n")  # 写入训练迭代次数
        data_str = json.dumps(time_list, indent=4)  # 将时间列表转换为格式化的JSON字符串

        file.write(data_str)  # 将格式化后的时间数据写入文件
    return filepath  # 返回文件路径

class ReduceOp(Enum):  # 定义一个枚举类，用于表示常见的归约操作
    SUM = 0  # 求和操作
    PRODUCT = 1  # 求积操作
    MIN = 2  # 求最小值操作
    MAX = 3  # 求最大值操作
    BAND = 4  # 按位与操作
    BOR = 5  # 按位或操作
    BXOR = 6  # 按位异或操作
    AVG = 7  # 求平均值操作
    UNUSED = 8  # 未使用的操作

class CommType(str, Enum):  # 定义一个字符串类型的枚举类，表示通信类型
    """Enum class for possible comm types"""

    all_reduce = "all_reduce"  # 所有设备之间的归约操作
    isend = "isend"  # 异步发送操作
    irecv = "irecv"  # 异步接收操作
    broadcast = "broadcast"  # 广播操作
    all_gather = "all_gather"  # 所有设备的数据聚集
    reduce_scatter = "reduce_scatter"  # 归约并分散
    barrier = "barrier"  # 同步屏障
    reduce = "reduce"  # 归约操作
    reduce_scatter_tensor = "reduce_scatter_tensor"  # 归约分散张量
    all_gather_into_tensor = "all_gather_into_tensor"  # 聚集数据到张量
    computation = "computation"  # 计算操作
    epoch_end = "epoch_end"  # 训练周期结束
    all_to_all = "all_to_all"  # 全到全操作

    @classmethod  # 类方法，用于根据值返回对应的通信类型
    def get_comm_type(cls, value):
        for comm_type in cls:  # 遍历所有通信类型
            if comm_type.value == value:  # 如果找到匹配的通信类型
                return comm_type  # 返回对应的通信类型
        raise ValueError("Invailid communication type")  # 如果没有找到，抛出异常

class CommGroup(str, Enum):  # 定义一个字符串类型的枚举类，表示通信组
    """Enum class for possible comm groups"""

    dp_group = "dp_group"  # 数据并行组
    pp_group = "pp_group"  # 管道并行组
    tp_group = "tp_group"  # 张量并行组
    ep_group = "ep_group"  # 专家并行组
    ep_dp_group = "ep_dp_group"  # 专家数据并行组
    ep_tp_group = "ep_tp_group"  # 专家张量并行组
    embedding_group = "embedding_group"  # 嵌入组
    all = "all_nodes"  # 所有节点组

class WorkloadWriter:  # 定义一个工作负载写入器类
    @staticmethod  # 静态方法，用于写入工作负载数据
    def write_workload(workload: List[Dict], args, filename: str):  # 接受工作负载数据、配置和文件名作为参数
        df = pd.DataFrame.from_dict(workload)  # 将工作负载数据转换为DataFrame对象
        df = df.fillna(-1)  # 填充缺失值为-1
        df.to_csv(filename, index=False)  # 将DataFrame保存为CSV文件，去掉行索引

    @staticmethod  # 静态方法，用于加载工作负载数据
    def load_workload(filename: str) -> List[Dict]:  # 接受文件名作为参数
        filename = filename.split(".")  # 将文件名按点分割
        filename[-1] = "pkl"  # 将文件扩展名改为pkl
        filename = ".".join(filename)  # 重新组合文件名
        workload, args = pickle.load(open(filename, "rb"))  # 读取pickle文件，加载工作负载和配置
        return workload, args  # 返回工作负载和配置



def get_params():  # 定义获取参数的函数
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器

    # 定义--frame参数，通信框架选项，默认为Megatron
    parser.add_argument(
        "--frame",
        help="communication framework",
        choices=["Megatron", "DeepSpeed", "collective_test"],
        default="Megatron",
    )

    # 定义--gpu_type参数，指定GPU类型，默认为None
    parser.add_argument("--gpu_type", type=str, default=None),

    # 定义--world_size参数，指定GPU的数量，默认为1
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of GPUs")

    # 定义--tensor_model_parallel_size参数，指定张量模型并行度，默认为1
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1,
                        help='Degree of tensor model parallelism.')

    # 定义--pipeline_model_parallel参数，指定管道模型并行度，默认为1
    parser.add_argument("--pipeline_model_parallel", type=int, default=1,
                        help='Degree of pipeline model parallelism.')

    # 定义--context-parallel-size参数，指定上下文并行度，默认为1
    parser.add_argument('--context-parallel-size', type=int, default=1,
                       help='Degree of context parallelism.')

    # 定义--pp_rank参数，指定encoder和decoder的拆分位置，默认为-1
    parser.add_argument("--pp_rank", type=int, default=-1,
                        help='Rank where encoder and decoder should be split.')

    # 定义--global_batch参数，指定全局批量大小，默认为4
    parser.add_argument("--global_batch", type=int, default=4,
                        help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')

    # 定义--micro_batch参数，指定每个模型实例的批量大小，默认为1
    parser.add_argument("--micro_batch", type=int, default=1,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.'
                        )

    # 定义--epoch_num参数，指定训练迭代次数，默认为1
    parser.add_argument("--epoch_num", type=int, default=1,
                        help="Number of iterations")

    # 定义--computation_enable参数，是否启用计算，默认为False
    parser.add_argument("--computation_enable", action="store_true", help="Enable computation")

    # 定义--dtype参数，指定数据类型，默认为bfloat16
    parser.add_argument("--dtype", default="bfloat16")

    # 定义--ffn_hidden_size参数，指定Transformer前馈网络的隐藏层大小
    parser.add_argument(
        "--ffn_hidden_size",
        type=int,
        default=None,
        help="Transformer Feed-Forward Network hidden size. "
        "This is set to 4*hidden-size if not provided",
    )

    # 定义--enable_visual参数，是否启用可视化
    parser.add_argument(
        "--enable_visual",
        action="store_true",
        help="Enable visualization",
    )

    # 定义--workload_only参数，是否只生成工作负载
    parser.add_argument("--workload_only", action="store_true", help="Only generate workload")

    # 调用其他函数来获取额外的参数配置
    get_model_params(parser)
    get_ds_params(parser)
    get_megatron_params(parser)
    get_collective_test_params(parser)
    get_moe_params(parser)
    get_simAI_workload_params(parser)
    get_aiob_params(parser)

    # 解析命令行参数
    args = parser.parse_args()

    # 验证world_size能否被tensor_model_parallel_size和pipeline_model_parallel的乘积整除
    assert (
        args.world_size % (args.tensor_model_parallel_size * args.pipeline_model_parallel) == 0
    ), f"world size: {args.world_size}, tp: {args.tensor_model_parallel_size}, pp: {args.pipeline_model_parallel}"

    # 如果启用了moe，确保同时启用sequence_parallel
    if args.moe_enable:
        assert (
            args.moe_enable and args.enable_sequence_parallel
        ), f"moe must be enabled with sequence parallel"

    # 计算数据并行度
    args.dp_num = args.world_size // (args.tensor_model_parallel_size * args.pipeline_model_parallel)

    # 计算micro-batches的数量
    args.num_microbatches = args.global_batch // (args.dp_num * args.micro_batch)

    # 如果启用了aiob且未启用计算，则启用计算
    if args.aiob_enable and not args.computation_enable:
            args.computation_enable = True

    # 如果没有设置num_attention_heads，默认使用num_layers的值
    if args.num_attention_heads is None:
        args.num_attention_heads = args.num_layers

    # 获取填充后的词汇表大小
    args.padded_vocab_size = get_padded_vocab_size(args)

    # 如果未设置ffn_hidden_size，则根据hidden_size计算
    if args.ffn_hidden_size is None:
        if args.swiglu:
            # 如果启用了swiglu，减小MLP的维度，保持参数数量与4*h相当
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        else:
            args.ffn_hidden_size = 4 * args.hidden_size

    # 如果启用了swiglu，则设置gated_linear_unit为True，禁用bias和gelu融合
    if args.swiglu:
        args.gated_linear_unit = True
        args.bias_gelu_fusion = False

    # 检查专家并行度
    if args.expert_model_parallel_size > 1:
        assert args.num_experts is not None, "num_experts must be non None to use expert model parallelism"
        assert args.num_experts % args.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size."
        assert not args.dtype == "float16", \
            "Expert parallelism is not supported with fp16 training."

    # 如果启用了Grouped GEMM，确保dtype是bfloat16
    if args.moe_grouped_gemm:
        assert args.dtype == "bfloat16", 'Currently GroupedGEMM for MoE only supports bf16 dtype.'

    # 如果启用了管道并行度，调整层数
    if args.pipeline_model_parallel > 1 :
        args.num_layers = int(args.num_layers // args.pipeline_model_parallel)

    return args  # 返回解析后的参数对象


ARGS = None  # 初始化全局变量ARGS


def get_aiob_params(parser: argparse.ArgumentParser):  # 定义获取AIob参数的函数
    parser.add_argument(  # 添加--aiob_enable参数，启用AIob来获取操作的实际计算时间
        "--aiob_enable",
        action="store_true",  # 如果设置了该参数，将其值设置为True
        help="Enable aiob to get operation real compute time",  # 参数帮助信息
    )
    parser.add_argument("--comp_filepath", type=str, default=None,  # 添加--comp_filepath参数，指定AIob计算路径
                        help="Use aiob_lib to get operation real compute time",)  # 帮助信息
    parser.add_argument("--gated_linear_unit", default=False)  # 添加--gated_linear_unit参数，默认值为False
    parser.add_argument("--bias_gelu_fusion", action="store_true",  # 添加--bias_gelu_fusion参数，启用bias和gelu融合
                        help='Enable bias and gelu fusion.')  # 帮助信息
    parser.add_argument("--openai_gelu", action="store_true",  # 添加--openai_gelu参数，启用OpenAI的GeLU实现
                         help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility'
                       'reasons.')  # 帮助信息，警告此选项用于向后兼容
    parser.add_argument("--onnx_safe", action="store_true",  # 添加--onnx_safe参数，启用ONNX安全模式
                        help='Use workarounds for known problems with '
                       'Torch ONNX exporter')  # 帮助信息
    parser.add_argument("--squared_relu", action="store_true",  # 添加--squared_relu参数，启用平方ReLU激活
                        help='Use squared relu activation instead of default gelu')  # 帮助信息
    parser.add_argument('--recompute_activations', action='store_true',  # 添加--recompute_activations参数，重新计算激活
                       help='Recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')  # 帮助信息


def get_model_params(parser: argparse.ArgumentParser):  # 定义获取模型参数的函数
    parser.add_argument("--model_name", help="Model for training")  # 添加--model_name参数，指定训练使用的模型
    parser.add_argument(  # 添加--hidden_size参数，指定Transformer的隐藏层大小，默认为1024
        "--hidden_size", type=int, help='Transformer hidden size.', default=1024
    )
    parser.add_argument("--num_layers", type=int, help='Number of transformer layers.', default=24)  # 添加--num_layers参数，指定Transformer层数，默认为24
    parser.add_argument(  # 添加--seq_length参数，指定最大序列长度，默认为2048
        "--seq_length", type=int, help='Maximum sequence length to process.', default=2048
    )
    parser.add_argument("--num_attention_heads", help='Number of transformer attention heads.',type=int, default=None)  # 添加--num_attention_heads参数，指定注意力头数
    parser.add_argument("--vocab_size", type=int, help='Size of vocab before EOD or padding.', default=32000)  # 添加--vocab_size参数，指定词汇表大小，默认为32000
    parser.add_argument("--max_position_embeddings", type=int,help='Maximum number of position embeddings to use. '  # 添加--max_position_embeddings参数，指定最大位置编码数
                       'This is the size of position embedding.', default=4096)
    parser.add_argument("--add_bias_linear",help='Enable bias in the linear layers', action="store_true")  # 添加--add_bias_linear参数，启用线性层的偏置项
    parser.add_argument(  # 添加--use_flash_attn参数，启用FlashAttention实现
        "--use_flash_attn",
        action="store_true",
        help="Use FlashAttention implementation of attention.",
    )
    parser.add_argument(  # 添加--swiglu参数，启用SwigLU（门控线性单元）和SiLU激活
        "--swiglu",
        action="store_true",
        help="Use gated linear units and SiLU activation instead of default gelu",
    )


def get_ds_params(parser: argparse.ArgumentParser):  # 定义获取DeepSpeed参数的函数
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])  # 添加--stage参数，指定DeepSpeed阶段，默认为3
    parser.add_argument("--amp_enabled", action="store_true")  # 添加--amp_enabled参数，启用自动混合精度
    parser.add_argument("--reduce_bucket_size", type=int, default=int(5e8))  # 添加--reduce_bucket_size参数，指定减少桶的大小，默认为500MB

    # for stage1/2 only  # 仅适用于阶段1和2
    parser.add_argument("--allgather_bucket_size", type=int, default=int(5e8))  # 添加--allgather_bucket_size参数，指定所有聚集的桶大小
    parser.add_argument("--contiguous_gradients", action="store_true")  # 添加--contiguous_gradients参数，启用连续梯度

    # for stage 3 only  # 仅适用于阶段3
    parser.add_argument("--param_persistence_threshold", type=int, default=int(1e5))  # 添加--param_persistence_threshold参数，设置参数持久化阈值
    parser.add_argument(  # 添加--model_persistence_threshold参数，设置模型持久化阈值
        "--model_persistence_threshold", type=int, default=int(sys.maxsize)
    )
    parser.add_argument("--max_live_parameters", type=int, default=int(1e9))  # 添加--max_live_parameters参数，设置最大活跃参数数
    parser.add_argument("--prefetch_bucket_size", type=int, default=int(1e9))  # 添加--prefetch_bucket_size参数，设置预取桶大小


def get_megatron_params(parser: argparse.ArgumentParser):  # 定义获取Megatron参数的函数
    parser.add_argument("--enable_sequence_parallel",help='Enable sequence parallel optimization.',action="store_true")  # 添加--enable_sequence_parallel参数，启用序列并行优化
    parser.add_argument(  # 添加--use-distributed-optimizer参数，启用分布式优化器
        "--use-distributed-optimizer",
        action="store_true",
        help="Use distributed optimizer.",
    )
    parser.add_argument("--make_vocab_size_divisible_by", help='Pad the vocab size to be divisible by this value.'  # 添加--make_vocab_size_divisible_by参数，确保词汇表大小能被该值整除
                       'This is added for computational efficiency reasons.',type=int, default=128)
    parser.add_argument(  # 添加--overlap_grad_reduce参数，启用梯度重叠减少（尚未实现）
        "--overlap_grad_reduce",
        action="store_true",
        default=False,
        help="If set, overlap DDP grad reduce. (Not implement yet)",
    )


def get_collective_test_params(parser: argparse.ArgumentParser):  # 定义获取集体通信测试参数的函数
    parser.add_argument("--begin_size", type=int, default=1048576)  # 添加--begin_size参数，指定开始测试的大小，默认为1048576
    parser.add_argument("--end_size", type=int, default=1048576)  # 添加--end_size参数，指定结束测试的大小，默认为1048576
    parser.add_argument("--test_comm", type=str, default="all_reduce")  # 添加--test_comm参数，指定测试的通信类型，默认为all_reduce
    parser.add_argument("--iter_num", type=int, default=500)  # 添加--iter_num参数，指定迭代次数，默认为500
    parser.add_argument("--multi_all_reduce_enable", type=int, default=0)  # 添加--multi_all_reduce_enable参数，启用多次all-reduce操作


def get_simAI_workload_params(parser: argparse.ArgumentParser):  # 定义获取simAI工作负载参数的函数
    parser.add_argument("--overlap_version", action="store_true")  # 添加--overlap_version参数，启用重叠版本


def get_moe_params(parser: argparse.ArgumentParser):  # 定义获取MoE（专家模型）参数的函数
    parser.add_argument('--moe_enable', action="store_true")  # 添加--moe_enable参数，启用MoE模型
    parser.add_argument('--expert_model_parallel_size', type=int, default=1, help='Degree of expert model parallelism.')  # 添加--expert_model_parallel_size参数，指定专家模型并行度，默认为1
    parser.add_argument('--num_experts', type=int, default=1, help='Number of Experts in MoE (None means no MoE)')  # 添加--num_experts参数，指定专家数量
    parser.add_argument('--moe_router_topk', type=int, default=1, help='Number of experts to route to for each token. The default is 2.')  # 添加--moe_router_topk参数，指定每个token路由到的专家数量，默认为1
    parser.add_argument('--moe_grouped_gemm', action='store_true',  # 添加--moe_grouped_gemm参数，启用分组GEMM优化
                       help='When there are multiple experts per rank, compress multiple local (potentially small) gemms in a single kernel launch to improve the utilization and performance by leveraging the Grouped GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).')
    parser.add_argument('--activation_func', type=str, help='activation_func for mlp')  # 添加--activation_func参数，指定MLP的激活函数


def ensure_divisibility(numerator, denominator):  # 定义确保数字能被整除的函数
    """Ensure that numerator is divisible by the denominator."""  # 确保分子能被分母整除
    assert numerator % denominator == 0, "{} is not divisible by {}".format(  # 使用assert验证分子是否能被分母整除
        numerator, denominator
    )


def get_padded_vocab_size(args):  # 定义获取填充词汇表大小的函数
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""  # 填充词汇表大小，以便能被模型并行大小整除，并且保持GPU友好
    after = args.vocab_size  # 获取当前词汇表大小
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size  # 计算填充后的倍数
    while (after % multiple) != 0:  # 如果当前大小不能被倍数整除
        after += 1  # 增加词汇表大小，直到能被整除
    return after  # 返回填充后的词汇表大小


def divide(numerator, denominator):  # 定义除法函数
    """Ensure that numerator is divisible by the denominator and return
    the division value."""  # 确保分子能被分母整除并返回商
    ensure_divisibility(numerator, denominator)  # 调用ensure_divisibility函数确保整除
    return numerator // denominator  # 返回商

