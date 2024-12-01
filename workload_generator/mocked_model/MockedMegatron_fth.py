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

from utils.utils import divide, CommType, CommGroup  # 从utils.utils导入divide函数，CommType和CommGroup枚举类型
from workload_generator.mocked_model.MockedModel import MockedModel, Linear, MockedParam  # 从mocked_model导入MockedModel, Linear, MockedParam
from log_analyzer.log import Workload, LogItem  # 从log_analyzer模块导入Workload和LogItem


# mocked version of Megatron RowParallelLinear
class MegatronRowLinear(MockedModel):  # 定义MegatronRowLinear类，继承MockedModel类（表示模型的一种模拟）
    def __init__(  # 初始化方法，创建MegatronRowLinear的实例
        self,
        input_size,  # 输入的大小
        output_size,  # 输出的大小
        tp,  # tensor并行度（在数据并行中切分张量的数量）
        seq_len,  # 序列长度
        batch_size,  # 批量大小
        layer_id,  # 层的ID
        prefix_name,  # 前缀名称（用于命名）
        sequence_parallel_enabled=True,  # 是否启用序列并行
        computation_enable=False,  # 是否启用计算
        name=None,  # 模型名称
        add_bias_linear=False,  # 是否添加偏置项
    ):
        self.layer_id = layer_id  # 初始化层的ID
        self.name = prefix_name + "_row"  # 使用前缀名称加上"_row"来定义名称
        self.input_size, self.output_size = input_size, output_size  # 定义输入和输出的大小
        self.input_size_per_partition = divide(input_size, tp)  # 切分输入大小：每个分区的输入大小
        self.weight = MockedParam(  # 创建MockedParam对象，模拟权重参数
            (output_size, self.input_size_per_partition), name=name  # 权重的形状是 (output_size, input_size_per_partition)
        )
        if add_bias_linear:  # 如果启用了偏置项
            self.bias = MockedParam((output_size, 1), name=self.name + "_bias")  # 创建偏置参数
        self.sequence_parallel_enabled = sequence_parallel_enabled  # 启用序列并行
        self.computation_enable = computation_enable  # 启用计算
        self.tensor_model_parallel_size, self.seq_len, self.batch_size = tp, seq_len, batch_size  # 初始化tensor并行大小、序列长度、批量大小
        self.comm_size = 2 * seq_len * batch_size * output_size  # 计算通信大小

    def forward(self):  # 定义前向传播方法
        workloads = Workload()  # 创建工作负载实例
        # output_ = torch.matmul(total_input, weight.t()): (s, b, h)
        if self.computation_enable:  # 如果计算启用
            workloads.append(  # 将计算工作负载记录到workloads中
                LogItem(  # 创建日志条目
                    comm_type=CommType.computation,  # 通信类型为计算
                    msg_size=(  # 消息大小：两个矩阵相乘的尺寸
                        (self.seq_len, self.batch_size, self.input_size_per_partition),  # 输入张量的形状
                        (self.input_size_per_partition, self.output_size),  # 权重张量的形状
                    ),
                    stage="forward.MegatronRowLinear." + self.name,  # 阶段信息
                )
            )
        if self.tensor_model_parallel_size > 1:  # 如果tensor并行度大于1
            if self.sequence_parallel_enabled:  # 如果启用了序列并行
                # output_ = reduce_scatter_to_sequence_parallel_region(output_parallel): (s/tp, b, h)
                workloads.append(  # 记录reduce_scatter工作负载
                    LogItem(
                        comm_type=CommType.reduce_scatter,  # 通信类型为reduce_scatter
                        comm_group=CommGroup.tp_group,  # 所属通信组为tensor模型并行组
                        comm_group_size=self.tensor_model_parallel_size,  # 通信组大小
                        msg_size=self.comm_size,  # 消息大小
                        stage="forward.MegatronRowLinear",  # 阶段信息
                    )
                )
            else:  # 如果没有启用序列并行
                # output_ = reduce_from_tensor_model_parallel_region(output_parallel)
                workloads.append(  # 记录all_reduce工作负载
                    LogItem(
                        comm_type=CommType.all_reduce,  # 通信类型为all_reduce
                        comm_group=CommGroup.tp_group,  # 所属通信组为tensor模型并行组
                        comm_group_size=self.tensor_model_parallel_size,  # 通信组大小
                        msg_size=self.comm_size,  # 消息大小
                        stage="forward.MegatronRowLinear",  # 阶段信息
                    )
                )
        return workloads  # 返回生成的工作负载

    def backward(self):  # 定义反向传播方法
        workloads = Workload()  # 创建工作负载实例
        if self.tensor_model_parallel_size > 1:  # 如果tensor并行度大于1
            if self.sequence_parallel_enabled:  # 如果启用了序列并行
                # output_ = reduce_scatter_to_sequence_parallel_region(output_parallel): (s/tp, b, h)
                workloads.append(  # 记录all_gather工作负载
                    LogItem(
                        comm_type=CommType.all_gather,  # 通信类型为all_gather
                        comm_group=CommGroup.tp_group,  # 所属通信组为tensor模型并行组
                        comm_group_size=self.tensor_model_parallel_size,  # 通信组大小
                        msg_size=self.comm_size,  # 消息大小
                        stage="backward.MegatronRowLinear",  # 阶段信息
                    )
                )
        # grad_input = grad_output.matmul(weight): (s, b, h)*(h, h'/N)
        # grad_weight = grad_output.t().matmul(total_input): (h, s*b)*(s*b, h'/N)
        if self.computation_enable:  # 如果计算启用
            workloads.append(  # 记录计算工作负载
                LogItem(
                    comm_type=CommType.computation,  # 通信类型为计算
                    msg_size=(  # 消息大小：两个矩阵相乘的尺寸
                        (self.seq_len, self.batch_size, self.output_size),  # 输出张量的形状
                        self.weight.shape,  # 权重张量的形状
                    ),
                    stage="backward.MegatronRowLinear" + self.name,  # 阶段信息
                )
            )
            workloads.append(  # 记录计算工作负载
                LogItem(
                    comm_type=CommType.computation,  # 通信类型为计算
                    msg_size=(  # 消息大小：两个矩阵相乘的尺寸
                        (self.output_size, self.seq_len * self.batch_size),  # 输入张量的形状
                        (self.seq_len * self.batch_size, self.input_size_per_partition),  # 权重张量的形状
                    ),
                    stage="backward.MegatronRowLinear" + self.name,  # 阶段信息
                )
            )
        return workloads  # 返回生成的反向传播工作负载

class MegatronColumnLinear(MockedModel):  # 定义MegatronColumnLinear类，继承自MockedModel，表示一种模型的模拟实现
    def __init__(  # 初始化方法，定义MegatronColumnLinear对象时的参数
        self,
        input_size,  # 输入的大小
        output_size,  # 输出的大小
        tp,  # tensor并行度（在数据并行中切分张量的数量）
        seq_len,  # 序列长度
        batch_size,  # 批量大小
        layer_id,  # 层的ID
        prefix_name,  # 前缀名称（用于命名）
        sequence_parallel_enabled=True,  # 是否启用序列并行
        computation_enable=False,  # 是否启用计算
        name=None,  # 模型名称
        add_bias_linear=False,  # 是否添加偏置项
    ):
        self.layer_id = layer_id  # 初始化层的ID
        self.name = prefix_name + "_column"  # 使用前缀名称加上"_column"来定义该层的名称
        self.input_size, self.output_size = input_size, output_size  # 初始化输入和输出的大小
        self.output_size_per_partition = divide(output_size, tp)  # 将输出大小按照tensor并行度分割，每个分区的输出大小
        self.weight = MockedParam(  # 创建模拟参数对象，用于存储权重
            (input_size, self.output_size_per_partition), name=name  # 权重的形状为 (input_size, output_size_per_partition)
        )
        if add_bias_linear:  # 如果启用了偏置项
            self.bias = MockedParam(  # 创建模拟的偏置参数
                (self.output_size_per_partition, 1), name=self.name + "_bias"  # 偏置的形状为 (output_size_per_partition, 1)
            )
        self.sequence_parallel_enabled = sequence_parallel_enabled  # 是否启用序列并行
        self.computation_enable = computation_enable  # 是否启用计算
        self.tensor_model_parallel_size, self.seq_len, self.batch_size = tp, seq_len, batch_size  # 初始化tensor模型并行大小、序列长度和批量大小
        self.comm_size = 2 * seq_len * batch_size * input_size  # 计算通信消息的大小
        if self.tensor_model_parallel_size > 1 and self.sequence_parallel_enabled:  # 如果启用了tensor模型并行并且启用了序列并行
            self.seq_len *= self.tensor_model_parallel_size  # 序列长度按照tensor模型并行的大小进行扩展

    def forward(self):  # 定义前向传播方法
        workloads = Workload()  # 创建一个空的工作负载实例，用于存储工作负载日志
        if self.tensor_model_parallel_size > 1:  # 如果tensor模型并行大小大于1
            if self.sequence_parallel_enabled:  # 如果启用了序列并行
                workloads.append(  # 向工作负载中添加日志条目
                    LogItem(  # 创建日志条目，表示通信操作
                        comm_type=CommType.all_gather,  # 通信类型为all_gather，表示数据收集
                        comm_group=CommGroup.tp_group,  # 所属通信组为tensor模型并行组
                        comm_group_size=self.tensor_model_parallel_size,  # 通信组的大小
                        msg_size=self.comm_size,  # 消息大小
                        stage="forward.MegatronColumnLinear",  # 阶段信息
                    )
                )
        # output = torch.matmul(total_input, weight.t())  # 矩阵乘法，计算输出
        if self.computation_enable:  # 如果启用了计算
            workloads.append(  # 向工作负载中添加计算操作的日志条目
                LogItem(  # 创建日志条目，表示计算操作
                    comm_type=CommType.computation,  # 通信类型为计算
                    msg_size=(  # 消息大小：两个矩阵相乘的维度
                        (self.seq_len, self.batch_size, self.input_size),  # 输入张量的形状
                        (self.input_size, self.output_size_per_partition),  # 权重张量的形状
                    ),
                    stage="forward.MegatronColumnLinear." + self.name,  # 阶段信息
                )
            )
        return workloads  # 返回生成的工作负载

    def backward(self):  # 定义反向传播方法
        workloads = Workload()  # 创建一个空的工作负载实例
        if self.tensor_model_parallel_size > 1:  # 如果tensor模型并行大小大于1
            if self.sequence_parallel_enabled:  # 如果启用了序列并行
                workloads.append(  # 向工作负载中添加日志条目
                    LogItem(  # 创建日志条目，表示all_gather操作
                        comm_type=CommType.all_gather,  # 通信类型为all_gather
                        comm_group=CommGroup.tp_group,  # 所属通信组为tensor模型并行组
                        comm_group_size=self.tensor_model_parallel_size,  # 通信组的大小
                        msg_size=self.comm_size,  # 消息大小
                        stage="backward.MegatronColumnLinear",  # 阶段信息
                    )
                )
        # grad_input = grad_output.matmul(weight): (s, b, h'/N)*(h'/N, h)
        # grad_weight = grad_output.t().matmul(total_input): (h, s*b)*(s*b, h'/N)
        if self.computation_enable:  # 如果启用了计算
            workloads.append(  # 向工作负载中添加计算操作的日志条目
                LogItem(  # 创建日志条目，表示计算操作
                    comm_type=CommType.computation,  # 通信类型为计算
                    msg_size=(  # 消息大小：两个矩阵相乘的维度
                        (self.seq_len, self.batch_size, self.output_size_per_partition),  # 输出张量的形状
                        (self.output_size_per_partition, self.input_size),  # 权重张量的形状
                    ),
                    stage="backward.MegatronColumnLinear." + self.name,  # 阶段信息
                )
            )
        if self.tensor_model_parallel_size > 1:  # 如果tensor模型并行大小大于1
            if self.sequence_parallel_enabled:  # 如果启用了序列并行
                workloads.append(  # 向工作负载中添加日志条目
                    LogItem(  # 创建日志条目，表示reduce_scatter操作
                        comm_type=CommType.reduce_scatter,  # 通信类型为reduce_scatter
                        comm_group=CommGroup.tp_group,  # 所属通信组为tensor模型并行组
                        comm_group_size=self.tensor_model_parallel_size,  # 通信组的大小
                        msg_size=self.comm_size,  # 消息大小
                        stage="backward.MegatronColumnLinear",  # 阶段信息
                    )
                )
        if self.computation_enable:  # 如果启用了计算
            workloads.append(  # 向工作负载中添加计算操作的日志条目
                LogItem(  # 创建日志条目，表示计算操作
                    comm_type=CommType.computation,  # 通信类型为计算
                    msg_size=(  # 消息大小：两个矩阵相乘的维度
                        (
                            self.output_size_per_partition,
                            self.seq_len * self.batch_size,
                        ),
                        (self.seq_len * self.batch_size, self.input_size),  # 权重张量的形状
                    ),
                    stage="backward.MegatronColumnLinear." + self.name,  # 阶段信息
                )
            )
        if self.tensor_model_parallel_size > 1:  # 如果tensor模型并行大小大于1
            if not self.sequence_parallel_enabled:  # 如果没有启用序列并行
                workloads.append(  # 向工作负载中添加日志条目
                    LogItem(  # 创建日志条目，表示all_reduce操作
                        comm_type=CommType.all_reduce,  # 通信类型为all_reduce
                        comm_group=CommGroup.tp_group,  # 所属通信组为tensor模型并行组
                        comm_group_size=self.tensor_model_parallel_size,  # 通信组的大小
                        msg_size=self.comm_size,  # 消息大小
                        stage="backward.MegatronColumnLinear",  # 阶段信息
                    )
                )
        return workloads  # 返回生成的反向传播工作负载


class FusedLayernorm(MockedModel):  # 定义一个FusedLayernorm类，继承自MockedModel，表示一个归一化层
    def __init__(self, hidden_size):  # 初始化方法，接收hidden_size参数
        self.layer_id = 0  # 层的ID为0
        self.name = "fused"  # 层的名称为"fused"
        self.weight = MockedParam((hidden_size, 1))  # 创建一个模拟参数weight，形状为(hidden_size, 1)
        self.bias = MockedParam((hidden_size, 1))  # 创建一个模拟参数bias，形状为(hidden_size, 1)


class MegatronAttention(MockedModel):  # 定义MegatronAttention类，继承自MockedModel，表示一个注意力层
    def __init__(  # 初始化方法，接收多个参数来配置注意力层
        self,
        num_attention_heads,  # 注意力头的数量
        hidden_size,  # 隐藏层大小
        tp,  # tensor并行度（模型切分的数量）
        seq_len,  # 序列长度
        batch_size,  # 批量大小
        layer_id,  # 层ID
        sequence_parallel_enabled,  # 是否启用序列并行
        computation_enable,  # 是否启用计算
        add_bias_linear,  # 是否添加偏置
    ):
        self.name = "attention_layer"  # 层的名称为"attention_layer"
        self.layer_id = layer_id  # 设置该层的ID
        self.kv_channels = hidden_size // num_attention_heads  # 计算每个注意力头的键值（key-value）通道数
        self.kv_projection_size = self.kv_channels * num_attention_heads  # 键值投影的大小（所有注意力头的键值通道大小）
        self.query_projection_size = self.kv_channels * num_attention_heads  # 查询投影的大小（所有注意力头的查询通道大小）

        # 创建qkv（查询、键、值）投影层（使用MegatronColumnLinear实现），这是一个列并行的线性层
        self.qkv = MegatronColumnLinear(
            hidden_size,  # 输入大小为隐藏层大小
            self.query_projection_size + 2 * self.kv_projection_size,  # 输出大小为查询投影 + 键值投影大小的两倍
            tp,  # tensor并行度
            seq_len,  # 序列长度
            batch_size,  # 批量大小
            layer_id,  # 层ID
            "attention",  # 使用"attention"作为前缀
            sequence_parallel_enabled,  # 是否启用序列并行
            computation_enable,  # 是否启用计算
            name="attention_column",  # 该层的名称
            add_bias_linear=add_bias_linear,  # 是否添加偏置
        )

        # 创建注意力计算密集层（使用MegatronRowLinear实现），这是一个行并行的线性层
        self.attention_dense = MegatronRowLinear(
            self.query_projection_size,  # 输入大小为查询投影的大小
            hidden_size,  # 输出大小为隐藏层大小
            tp,  # tensor并行度
            seq_len,  # 序列长度
            batch_size,  # 批量大小
            layer_id,  # 层ID
            "attention",  # 使用"attention"作为前缀
            sequence_parallel_enabled,  # 是否启用序列并行
            computation_enable,  # 是否启用计算
            name="attention_row",  # 该层的名称
            add_bias_linear=add_bias_linear,  # 是否添加偏置
        )

    def forward(self):  # 定义前向传播方法
        workloads = Workload()  # 创建一个空的工作负载实例，用于存储工作负载日志
        workloads.extend(self.qkv.forward())  # 调用qkv的前向传播，将结果添加到工作负载中
        workloads.extend(self.attention_dense.forward())  # 调用attention_dense的前向传播，将结果添加到工作负载中
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])  # 检查工作负载中的每个条目是否都是LogItem实例
        return workloads  # 返回工作负载

    def backward(self):  # 定义反向传播方法
        workloads = Workload()  # 创建一个空的工作负载实例
        workloads.extend(self.qkv.backward())  # 调用qkv的反向传播，将结果添加到工作负载中
        workloads.extend(self.attention_dense.backward())  # 调用attention_dense的反向传播，将结果添加到工作负载中
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])  # 检查工作负载中的每个条目是否都是LogItem实例
        return workloads  # 返回工作负载

class MegatronMlp(MockedModel):  # 定义一个MegatronMlp类，继承自MockedModel，表示一个MLP（多层感知机）层
    def __init__(  # 初始化方法，接收多个参数来配置MLP层
        self,
        hidden_size,  # 输入的隐藏层大小
        ffn_hidden_size,  # Feed Forward Network（FFN）部分的隐藏层大小
        tp,  # tensor并行度（模型切分的数量）
        seq_len,  # 序列长度
        batch_size,  # 批量大小
        layer_id,  # 层ID
        sequence_parallel_enabled,  # 是否启用序列并行
        computation_enable,  # 是否启用计算
        add_bias_linear,  # 是否添加偏置
    ):
        self.name = "mlp_layer"  # 层的名称为"mlp_layer"
        self.layer_id = layer_id  # 设置该层的ID
        # 创建dense_h_to_4h层（通过MegatronColumnLinear实现），这是一个列并行的线性层
        self.dense_h_to_4h = MegatronColumnLinear(
            hidden_size,  # 输入大小为隐藏层大小
            ffn_hidden_size,  # 输出大小为FFN隐藏层大小
            tp,  # tensor并行度
            seq_len,  # 序列长度
            batch_size,  # 批量大小
            layer_id,  # 层ID
            "mlp",  # 使用"mlp"作为前缀
            sequence_parallel_enabled,  # 是否启用序列并行
            computation_enable,  # 是否启用计算
            name="mlp_column",  # 该层的名称
            add_bias_linear=add_bias_linear,  # 是否添加偏置
        )
        # 创建dense_4h_to_h层（通过MegatronRowLinear实现），这是一个行并行的线性层
        self.dense_4h_to_h = MegatronRowLinear(
            ffn_hidden_size,  # 输入大小为FFN隐藏层大小
            hidden_size,  # 输出大小为隐藏层大小
            tp,  # tensor并行度
            seq_len,  # 序列长度
            batch_size,  # 批量大小
            layer_id,  # 层ID
            "mlp",  # 使用"mlp"作为前缀
            sequence_parallel_enabled,  # 是否启用序列并行
            computation_enable,  # 是否启用计算
            name="mlp_row",  # 该层的名称
            add_bias_linear=add_bias_linear,  # 是否添加偏置
        )

    def forward(self):  # 定义前向传播方法
        workloads = Workload()  # 创建一个空的工作负载实例，用于存储工作负载日志
        workloads.extend(self.dense_h_to_4h.forward())  # 调用dense_h_to_4h的前向传播，将结果添加到工作负载中
        workloads.extend(self.dense_4h_to_h.forward())  # 调用dense_4h_to_h的前向传播，将结果添加到工作负载中
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])  # 检查工作负载中的每个条目是否都是LogItem实例
        return workloads  # 返回工作负载

    def backward(self):  # 定义反向传播方法
        workloads = Workload()  # 创建一个空的工作负载实例
        workloads.extend(self.dense_h_to_4h.backward())  # 调用dense_h_to_4h的反向传播，将结果添加到工作负载中
        workloads.extend(self.dense_4h_to_h.backward())  # 调用dense_4h_to_h的反向传播，将结果添加到工作负载中
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])  # 检查工作负载中的每个条目是否都是LogItem实例
        return workloads  # 返回工作负载
class GroupedMLP(MockedModel):  # 定义一个GroupedMLP类，继承自MockedModel，用于表示一个分组MLP模型
    def __init__(  # 初始化方法，接收多个参数来配置GroupedMLP模型
        self,
        batch_size,  # 批量大小
        hidden_size,  # 隐藏层大小
        tp,  # tensor并行度（模型的分割数量）
        expert_model_parallel_size,  # 专家模型并行度
        ffn_hidden_size,  # FFN（前馈神经网络）隐藏层大小
        seq_len,  # 序列长度
        topk,  # Top-K选择（可能与专家选择相关）
        num_experts,  # 专家数量
        id,  # 层的ID
    ):
        self.name = "mlp_moelayer"  # 设置该层的名称为"mlp_moelayer"
        self.layer_id = id  # 设置该层的ID
        # 计算每个本地专家的数量
        num_local_experts = num_experts // expert_model_parallel_size
        # 计算第一层全连接的输出大小
        fc1_output_size = ffn_hidden_size * num_local_experts
        # 计算每个partition的fc1输出大小
        fc1_output_size_per_parttition = divide(fc1_output_size, tp)
        # 计算第二层全连接的输入大小
        fc2_input_size = ffn_hidden_size * num_local_experts
        # 计算每个partition的fc2输入大小
        fc2_input_size_per_parttition = divide(fc2_input_size, tp)
        # 初始化权重1（第一层线性变换的权重）
        self.weight1 = MockedParam((hidden_size, fc1_output_size_per_parttition))
        # 初始化权重2（第二层线性变换的权重）
        self.weight2 = MockedParam((fc2_input_size_per_parttition, hidden_size))
        # 设置并行度（tensor并行度）
        self.tp_size = tp
        # 设置Top-K选择
        self.topk = topk
        # 设置序列长度
        self.seq_len = seq_len
        # 设置专家数量
        self.num_experts = num_experts
        # 设置批量大小
        self.batch_size = batch_size
        # 设置隐藏层大小
        self.hidden_size = hidden_size

    def permutation(self, stage):  # 定义permutation方法，用于进行专家并行中的数据交换
        workloads = Workload()  # 创建一个空的工作负载实例
        if self.tp_size > 1:  # 如果tensor并行度大于1
            # 添加all_to_all的通信工作负载（用于分发数据到不同的TP组）
            workloads.append(
                LogItem(
                    comm_type=CommType.all_to_all,  # 通信类型：all_to_all
                    comm_group=CommGroup.tp_group,  # 通信组：TP组
                    comm_group_size=self.tp_size,  # TP组的大小
                    msg_size=self.seq_len * self.hidden_size * self.batch_size // self.tp_size * 2,  # 计算消息大小
                    stage=f"{stage}.MoE",  # 当前阶段：MoE
                )
            )
        # 添加ep_group的all_to_all通信工作负载（用于专家并行的数据交换）
        workloads.append(
            LogItem(
                comm_type=CommType.all_to_all,  # 通信类型：all_to_all
                comm_group=CommGroup.ep_group,  # 通信组：专家组
                msg_size=self.seq_len * self.hidden_size * self.batch_size // self.tp_size * 2,  # 计算消息大小
                stage=f"{stage}.MoE",  # 当前阶段：MoE
            )
        )
        if self.tp_size > 1:  # 如果tensor并行度大于1
            # TODO: 假设令牌在所有专家间均匀分配，实际上并非如此
            workloads.append(
                LogItem(
                    comm_type=CommType.all_gather,  # 通信类型：all_gather
                    comm_group=CommGroup.tp_group,  # 通信组：TP组
                    msg_size=2 * self.hidden_size * self.topk * self.batch_size * self.seq_len,  # 计算消息大小
                    stage=f"{stage}.MoE.permutation",  # 当前阶段：MoE.permutation
                )
            )

        return workloads  # 返回工作负载

    def unpermutation(self, stage):  # 定义unpermutation方法，用于进行数据的反向交换
        workloads = Workload()  # 创建一个空的工作负载实例
        if self.tp_size > 1:  # 如果tensor并行度大于1
            # TODO: 假设令牌在所有专家间均匀分配，实际上并非如此
            workloads.append(
                LogItem(
                    comm_type=CommType.reduce_scatter,  # 通信类型：reduce_scatter
                    comm_group=CommGroup.tp_group,  # 通信组：TP组
                    msg_size=2 * self.hidden_size * self.batch_size * self.topk * self.seq_len,  # 计算消息大小
                    stage=f"{stage}.MoE.unpermutation",  # 当前阶段：MoE.unpermutation
                )
            )
        # 添加ep_group的all_to_all通信工作负载（用于专家并行的数据交换）
        workloads.append(
            LogItem(
                comm_type=CommType.all_to_all,  # 通信类型：all_to_all
                comm_group=CommGroup.ep_group,  # 通信组：专家组
                msg_size=self.seq_len * self.hidden_size * self.batch_size * self.topk // self.tp_size * 2,  # 计算消息大小
                stage=f"{stage}.MoE",  # 当前阶段：MoE
            )
        )
        if self.tp_size > 1:  # 如果tensor并行度大于1
            # TODO: 假设令牌在所有专家间均匀分配，实际上并非如此
            workloads.append(
                LogItem(
                    comm_type=CommType.all_to_all,  # 通信类型：all_to_all
                    comm_group=CommGroup.tp_group,  # 通信组：TP组
                    msg_size=2 * self.hidden_size * self.seq_len * self.batch_size // self.tp_size,  # 计算消息大小
                    stage=f"{stage}.MoE",  # 当前阶段：MoE
                )
            )

        return workloads  # 返回工作负载

    def forward(self):  # 定义前向传播方法
        workloads = Workload()  # 创建一个空的工作负载实例
        workloads.append(LogItem(  # 添加前向传播的初始通信工作负载
                    comm_type=CommType.all_gather,  # 通信类型：all_gather
                    comm_group=CommGroup.tp_group,  # 通信组：TP组
                    msg_size=2 * self.hidden_size * self.batch_size * self.seq_len,  # 计算消息大小
                    stage=f"forward.MoE.preprocess",  # 当前阶段：MoE.preprocess
                ))
        workloads.extend(self.permutation(stage="forward"))  # 添加permutation阶段的工作负载
        workloads.extend(self.unpermutation(stage="forward"))  # 添加unpermutation阶段的工作负载
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])  # 确保所有工作负载都是LogItem类型
        return workloads  # 返回前向传播的工作负载

    def backward(self):  # 定义反向传播方法
        workloads = Workload()  # 创建一个空的工作负载实例
        self.permutation(stage="backward")  # 添加permutation阶段的反向传播工作负载
        self.unpermutation(stage="backward")  # 添加unpermutation阶段的反向传播工作负载
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])  # 确保所有工作负载都是LogItem类型
        return workloads  # 返回反向传播的工作负载


class SequentialMLP(MockedModel):
    def __init__(self):
        print("Not implement yet!")
        pass


class MegatronTransformorLayer(MockedModel):
    def __init__(
        self,
        hidden_size,
        ffn_hidden_size,
        tp,
        seq_len,
        batch_size,
        num_attention_heads,
        layer_id,
        expert_model_parallel_size,
        moe_router_topk,
        num_experts,
        moe_grouped_gemm=True,
        sequence_parallel_enabled=True,
        computation_enable=False,
        add_bias_linear=False,
        moe_enable=False,
    ):
        self.attention = MegatronAttention(
            num_attention_heads,
            hidden_size,
            tp,
            seq_len,
            batch_size,
            layer_id,
            sequence_parallel_enabled,
            computation_enable,
            add_bias_linear,
        )
        self.pre_mlp_layernorm = FusedLayernorm(hidden_size)
        self.post_attention_layernorm_bias = MockedParam((hidden_size, 1))
        if moe_enable and moe_grouped_gemm:
            self.mlp = GroupedMLP(
                batch_size,
                hidden_size,
                tp,
                expert_model_parallel_size,
                ffn_hidden_size,
                seq_len,
                moe_router_topk,
                num_experts,
                layer_id,
            )
        else:
            self.mlp = MegatronMlp(
                hidden_size,
                ffn_hidden_size,
                tp,
                seq_len,
                batch_size,
                layer_id,
                sequence_parallel_enabled,
                computation_enable,
                add_bias_linear,
            )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.attention.forward())
        workloads.extend(self.mlp.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        workloads.extend(self.attention.backward())
        workloads.extend(self.mlp.backward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads


class MegatronEmbedding(MockedModel):
    def __init__(self, padded_vocab_size, hidden_size, tp, seq_len, batch_size):
        self.name = "embedding_layer"
        self.layer_id = 0
        num_embedding_per_partition = divide(padded_vocab_size, tp)
        self.word_embedding = MockedParam(
            (4 * num_embedding_per_partition, hidden_size), name=self.name
        )
        self.tensor_model_parallel_size = tp
        # TODO : position embedding shape is max_sequence_length not sequence_length
        self.position_embedding = MockedParam((seq_len, hidden_size))
        self.comm_size = 2 * batch_size * seq_len * hidden_size

    def forward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            workloads.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="forward.MegatronEmbedding",
                )
            )
        return workloads

    def backward(self):
        workloads = Workload()
        if self.tensor_model_parallel_size > 1:
            workloads.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.tensor_model_parallel_size,
                    msg_size=self.comm_size,
                    stage="backward.MegatronEmbedding",
                )
            )
        return workloads


class MegatronModel(MockedModel):
    def __init__(self, config):
        self.embedding = MegatronEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
        )
        self.layers = [
            MegatronTransformorLayer(
                config.hidden_size,
                config.ffn_hidden_size,
                config.tensor_model_parallel_size,
                config.seq_length,
                config.micro_batch,
                config.num_attention_heads,
                i,
                config.expert_model_parallel_size,
                config.moe_router_topk,
                config.num_experts,
                config.moe_grouped_gemm,
                config.enable_sequence_parallel,
                config.computation_enable,
                config.add_bias_linear,
                config.moe_enable,
            )
            for i in range(config.num_layers)
        ]
        self.final_norm = MegatronColumnLinear(
            config.hidden_size,
            config.padded_vocab_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
            1,
            "final",
            sequence_parallel_enabled=config.enable_sequence_parallel,
            computation_enable=config.computation_enable,
            add_bias_linear=config.add_bias_linear,
        )

    def forward(self):
        workloads = Workload()
        workloads.extend(self.embedding.forward())
        for layer in self.layers:
            workloads.extend(layer.forward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads

    def backward(self):
        workloads = Workload()
        for layer in self.layers[::-1]:
            workloads.extend(layer.backward())
        workloads.extend(self.embedding.backward())
        assert all([isinstance(workload, LogItem) for workload in workloads.workload])
        return workloads
