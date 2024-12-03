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

import os, math  # 导入os模块用于操作系统功能，math模块用于数学运算
import pickle  # 导入pickle模块用于对象序列化和反序列化
import csv  # 导入csv模块用于CSV文件读写
import dataclasses  # 导入dataclasses模块用于简化类的定义
import numpy as np  # 导入numpy模块用于数组和矩阵的高效操作
from typing import Union, Dict, List  # 导入Union, Dict, List用于类型注解
from utils.utils import CommType, CommGroup  # 从utils模块导入CommType和CommGroup，用于定义通信类型和通信组
from log_analyzer.utils import convert_size_to_msg, calc_bw_log  # 从log_analyzer.utils导入转换消息大小和计算带宽的函数


@dataclasses.dataclass  # 使用dataclass装饰器简化类定义，自动生成init等方法
class LogItem:
    comm_type: CommType = dataclasses.field(default=None)  # 通信类型，默认为None
    comm_group: CommGroup = dataclasses.field(default=None)  # 通信组，默认为None
    comm_group_size: int = dataclasses.field(default=None)  # 通信组大小，默认为None
    msg_size: float = dataclasses.field(default=0)  # 消息大小，默认为0

    stage: str = dataclasses.field(default="")  # 阶段，默认为空字符串
    dst: int = dataclasses.field(default=None)  # 目标，默认为None
    src: int = dataclasses.field(default=None)  # 源，默认为None
    additional: str = dataclasses.field(default="")  # 附加信息，默认为空字符串

    _elapsed_time: float = dataclasses.field(default=None)  # 内部用于存储已用时间，默认为None
    algbw: float = dataclasses.field(default=None)  # 算法带宽，默认为None
    busbw: float = dataclasses.field(default=None)  # 总线带宽，默认为None
    count: float = dataclasses.field(default=1)  # 计数，默认为1

    @property  # 装饰器将此方法转化为属性
    def elapsed_time(self) -> float:  # 获取已用时间
        return self._elapsed_time

    @elapsed_time.setter  # 装饰器表示此方法为已用时间的设置器
    def elapsed_time(self, elapsed_time):
        self._elapsed_time = elapsed_time  # 设置已用时间
        self.algbw, self.busbw = calc_bw_log(  # 使用calc_bw_log计算算法带宽和总线带宽
            self.comm_type, self.msg_size, elapsed_time, self.comm_group_size
        )

    def is_epoch_end(self):  # 判断是否为epoch结束
        return self.comm_type == CommType.epoch_end  # 如果通信类型是epoch_end，返回True

    def is_workload(self):  # 判断是否为工作负载
        return self.elapsed_time is None  # 如果已用时间为空，则为工作负载

    def view_as_ds_log(self):  # 格式化日志信息为数据流格式
        log_str = f"[RANK 0] comm op: {self.comm_type} | comm group: {self.comm_group}"  # 格式化通信操作和通信组
        log_str += " | time (ms): {:.2f}".format(self.elapsed_time)  # 添加已用时间
        if self.comm_type == CommType.computation or self.additional == 'overlap':  # 如果是计算或重叠阶段
            log_str += " | msg size: " + '0'  # 消息大小为0
            log_str += " | algbw (GB): " + '0'  # 算法带宽为0
            log_str += " | busbw (GB): " + '0'  # 总线带宽为0
        else:
            log_str += " | msg size: " + convert_size_to_msg(self.msg_size)  # 将消息大小转换为可读格式
            log_str += " | algbw (GB): {:.2f} ".format(self.algbw)  # 格式化算法带宽为GB
            log_str += " | busbw (GB): {:.2f} ".format(self.busbw)  # 格式化总线带宽为GB
        return log_str  # 返回格式化后的日志字符串

    def csv_header(self):  # 获取日志条目的CSV头部
        return ",".join([k for k in self.__dict__.keys()])  # 返回类的所有字段名作为CSV头部

    def view_as_csv_line(self):  # 将日志条目转换为CSV格式的行
        return ",".join([str(getattr(self, k)) for k in self.__dict__.keys()])  # 获取类中每个字段的值，并连接成CSV格式

    def __str__(self):  # 定义日志条目的字符串表示
        if self.is_workload():  # 如果是工作负载，则返回"None"
            return "None"
        return "None"  # 默认返回"None"


def _print_stage_log(stage_name: str, stage_count: int, comm_type_info: Dict, primary_key: List[str], agg_key: List[str], performance_key: List[str], busbw_key: List[str]):
    header = f"{'Comm_Type':<15} {'Comm_Group':<12} {'Message_Size':<12} {'Count':<12} {'Avg_Elapsed_Time ± Std ':<24} {'Avg_BusBw ± Std':<24}\n"  # 定义日志头
    separator = "-" * len(header) + "\n"  # 生成分隔符
    log_str = separator + header + separator  # 将分隔符和头部信息拼接在一起

    for pkey in sorted(comm_type_info.keys()):  # 按照comm_type_info字典的键排序
        row_str = ""  # 初始化每一行的日志字符串
        values = {}  # 存储每一行的字段值
        for i, pkey_name in enumerate(primary_key):  # 遍历主键
            value = pkey[i] if pkey_name != "msg_size" else convert_size_to_msg(pkey[i])  # 如果是消息大小，进行格式转换
            values[pkey_name] = value  # 将主键的值存储在values字典中
        for key in agg_key:  # 遍历聚合键
            value = comm_type_info[pkey][key]  # 获取聚合值
            value = convert_size_to_msg(value) if key == "msg_size" else f"{value:.2f}"  # 如果是消息大小，转换为可读格式，否则保留两位小数
            values[key] = value  # 将聚合值存储在values字典中
        for key in performance_key:  # 遍历性能相关的键
            performance_value_list = sorted(comm_type_info[pkey][key])  # 对性能数据进行排序
            values[f'avg_{key}'] = f"{np.mean(performance_value_list):.2f}±{np.std(performance_value_list):.2f}"  # 计算平均值和标准差
            values[f'min_{key}'] = f"{performance_value_list[0]:.2f}"  # 最小值
            values[f'max_{key}'] = f"{performance_value_list[-1]:.2f}"  # 最大值
        
        for key in busbw_key:  # 遍历总线带宽的键
            busbw_value_list = sorted(comm_type_info[pkey][key])  # 对总线带宽数据进行排序
            values[f'avg_{key}'] = f"{np.mean(busbw_value_list):.2f}±{np.std(busbw_value_list):.2f}"  # 计算平均值和标准差

        row_str += f"{values['comm_type']:<15} {values['comm_group']:<12} {values['msg_size']:<12} {values['count']:<16} {values['avg__elapsed_time']:<24} {values['avg_busbw']:<18}\n"  # 拼接每一行数据
        log_str += row_str  # 将这一行数据添加到日志字符串中

    return log_str  # 返回最终拼接好的日志字符串


def _analyze_stage_log(comm_log: List[Dict], stage: str, comm_info: Dict[str, Dict]):
    def __update_info(
        info_dict,
        log,
        primary_key: List[str],
        agg_key: List[str],
        performance_key: List[str],
        busbw_key: List[str],
    ):
        primary_key = tuple(log[key] for key in primary_key)  # 将主键转换为元组形式，用于在info_dict中查找
        if primary_key not in info_dict:  # 如果info_dict中没有这个主键，初始化该主键的相关信息
            info_dict[primary_key] = dict((key, 0) for key in agg_key)  # 初始化聚合键的值为0
            info_dict[primary_key].update(dict((key, []) for key in performance_key))  # 初始化性能键的值为空列表
            info_dict[primary_key].update(dict((key, []) for key in busbw_key))  # 初始化总线带宽键的值为空列表
        for key in agg_key:  # 遍历聚合键，更新相应的值
            info_dict[primary_key][key] += log[key]  # 累加聚合值
        for key in performance_key:  # 遍历性能键，更新相应的值
            info_dict[primary_key][key].append(log[key])  # 将性能数据追加到列表中
        for key in busbw_key:  # 遍历总线带宽键，更新相应的值
            info_dict[primary_key][key].append(log[key])  # 将总线带宽数据追加到列表中

    if stage not in comm_info:  # 如果阶段(stage)不在comm_info中，初始化该阶段的信息
        comm_info[stage] = {
            "count": 0,  # 记录阶段的计数
            "comm_type_info": {},  # 记录每种通信类型的信息
            "detailed_comm_type_info": {},  # 记录详细的通信类型信息
        }
    comm_info[stage]["count"] += 1  # 增加该阶段的计数
    # comm_type_info用于记录按通信类型分组的统计信息 (key: comm_type, value: count, time_ms)
    comm_type_info = comm_info[stage]["comm_type_info"]
    # detailed_comm_type_info用于记录按通信类型和消息大小分组的统计信息 (key: comm_type, msg_size, value: count, time_ms)
    detailed_comm_type_info = comm_info[stage]["detailed_comm_type_info"]
    
    for log in comm_log:  # 遍历每条通信日志
        if log.comm_type != CommType.computation:  # 如果不是计算通信类型，更新信息
            __update_info(
                comm_type_info,  # 更新comm_type_info字典
                log.__dict__,  # 提取日志的字典形式
                ["comm_type", "comm_group"],  # 主键为通信类型和通信组
                ["count", "msg_size"],  # 聚合键为count和msg_size
                ["_elapsed_time"],  # 性能键为_elapsed_time
                ["busbw"],  # 总线带宽键为busbw
            )
            __update_info(
                detailed_comm_type_info,  # 更新detailed_comm_type_info字典
                log.__dict__,  # 提取日志的字典形式
                ["comm_type", "comm_group", "msg_size"],  # 主键为通信类型、通信组和消息大小
                ["count"],  # 聚合键为count
                ["_elapsed_time"],  # 性能键为_elapsed_time
                ["busbw"],  # 总线带宽键为busbw
            )


class Log:
    def __init__(self) -> None:  # Log类的初始化方法
        self.comm_logs = []  # 初始化通信日志列表
        self.comm_log_each_epoch = [[]]  # 每个epoch的通信日志，初始化为空的列表
        self.epoch_times = []  # 初始化记录每个epoch的时间

    def add_comm_log(self, comm_log: LogItem):  # 添加通信日志
        if (
            comm_log.is_epoch_end()  # 如果是epoch结束的标志
            and len(self.comm_logs) > 0  # 且当前日志列表不为空
            and not self.comm_logs[-1].is_epoch_end()  # 且最后一条日志不是epoch结束
        ):
            self.comm_logs.append(comm_log)  # 添加该epoch结束的日志
            self.comm_log_each_epoch.append([])  # 为新的epoch添加空的日志列表
            self.epoch_times.append(comm_log.elapsed_time)  # 记录该epoch的时间
            return
        self.comm_logs.append(comm_log)  # 否则直接添加日志
        self.comm_log_each_epoch[-1].append(comm_log)  # 将日志添加到当前epoch的日志列表中

    def analyze(self, print_fn=print):  # 分析通信日志的方法
        comm_info: Dict[str, Dict] = {}  # 初始化通信信息字典
        _analyze_stage_log(self.comm_log_each_epoch[0], "init", comm_info)  # 分析初始化阶段的通信日志
        for e_log in self.comm_log_each_epoch[1:]:  # 遍历训练阶段的每个epoch日志
            _analyze_stage_log(e_log, "train", comm_info)  # 分析训练阶段的通信日志
        for stage in comm_info.keys():  # 遍历所有阶段
            if stage != "init":  # 如果不是初始化阶段
                stage_count = comm_info[stage]["count"]  # 获取该阶段的计数
                comm_type_info = comm_info[stage]["comm_type_info"]  # 获取该阶段的通信类型信息
                detailed_comm_type_info = comm_info[stage]["detailed_comm_type_info"]  # 获取该阶段的详细通信类型信息

                log_str = _print_stage_log(stage, stage_count, detailed_comm_type_info, ["comm_type", "comm_group", "msg_size"], ["count"], ["_elapsed_time"], ["busbw"])  # 打印该阶段的详细日志
                print_fn(f"\n\tDetailed comm info for AICB {stage} stage\n{log_str}")  # 输出日志
        return comm_info  # 返回分析结果

    def dump(self, filename):  # 将通信日志保存到文件
        default_comm_folder_path = "results/comm_logs/"  # 定义默认的通信日志文件夹路径
        if not os.path.exists(default_comm_folder_path):  # 如果文件夹不存在
            os.makedirs(default_comm_folder_path, exist_ok=True)  # 创建文件夹
        if "." in filename:  # 如果文件名中有扩展名
            filename = filename.split(".")[0]  # 去掉扩展名
        filename = os.path.join("results/comm_logs/", filename)  # 将文件名与文件夹路径合并
        csv_filename = filename + "_log.csv"  # 为日志文件命名
        with open(csv_filename, "w") as f:  # 打开文件进行写入
            f.write(self.comm_logs[0].csv_header() + "\n")  # 写入CSV头部
            for log_item in self.comm_logs:  # 遍历所有通信日志
                f.write(log_item.view_as_csv_line() + "\n")  # 写入日志的CSV格式行
        return csv_filename  # 返回保存的CSV文件名

    @staticmethod
    def load(filename):  # 静态方法用于加载通信日志
        filename = filename.split(".")  # 分割文件名
        filename[-1] = "pkl"  # 将文件扩展名改为.pkl
        filename = ".".join(filename)  # 将文件名重新合并
        return pickle.load(open(filename, "rb"))  # 加载pickle文件并返回

    def _get_elapsed_time(self):  # 获取已用时间
        return self.epoch_times  # 返回每个epoch的时间

    def analyze_time(self, print_fn=print):  # 分析每个epoch的时间
        self.epoch_times.pop(0)  # 移除初始化阶段的时间
        max_val = max(self.epoch_times)  # 获取最大值
        min_val = min(self.epoch_times)  # 获取最小值
        mean_val = sum(self.epoch_times) / len(self.epoch_times)  # 计算平均值

        variance = sum((x - mean_val) ** 2 for x in self.epoch_times) / len(self.epoch_times)  # 计算方差
        variance = math.sqrt(variance)  # 计算标准差

        sorted_list = sorted(self.epoch_times)  # 排序时间列表
        p90_val = sorted_list[int(len(sorted_list) * 0.9)]  # 获取90分位数
        p99_val = sorted_list[int(len(sorted_list) * 0.99)]  # 获取99分位数
        header = f"{'Init time':<18} {'Max iteration time':<20} {'Min iteration time':<20} {'Avg iteration time':<20} {'P90 iteration time ':<20} {'Iteration time Std ':<20}\n"  # 打印表头
        separator = "-" * len(header) + "\n"  # 分隔符
        log_str = separator + header + separator  # 合并日志字符串
        iteration_result = f"{self.epoch_times[0]:<18.2f} {max_val:<20.2f} {min_val:<20.2f} {mean_val:<20.2f} {p90_val:<20.2f} {variance:<20.2f}\n"  # 打印每个epoch的时间结果
        log_str += iteration_result  # 将结果添加到日志字符串中
        print_fn(f"\n\tDetailed info for AICB iteration time\n{log_str}")  # 输出时间分析结果


class Workload:  # 定义一个工作负载类
    def __init__(self) -> None:  # 构造函数，初始化工作负载为空列表
        self.workload = []  # 工作负载列表

    def append(self, log_item: Union[LogItem, Dict]):  # 向工作负载中添加一个log_item，可以是LogItem对象或字典
        if isinstance(log_item, LogItem):  # 如果log_item是LogItem类型
            self.workload.append(log_item)  # 直接将LogItem对象添加到工作负载中
            return  # 返回，结束方法
        if "stage" not in log_item:  # 如果log_item字典中没有"stage"键
            log_item["stage"] = log_item["operation"] if "operation" in log_item else ""  # 将"stage"键设置为"operation"的值，如果没有"operation"键则为空字符串
        if "comm_group" not in log_item:  # 如果log_item字典中没有"comm_group"键
            assert (  # 进行断言，确保"comm_type"为计算类型时才可以不设置"comm_group"
                log_item["comm_type"] == CommType.computation
            ), "comm_group is required for non-computation comm_type"  # 如果comm_type不是计算类型，则抛出异常
            log_item["comm_group"] = CommGroup.all  # 如果是计算类型，设置默认的"comm_group"
        self.workload.append(  # 将log_item转换为LogItem对象，并添加到工作负载中
            LogItem(
                comm_type=log_item["comm_type"],  # 设置通信类型
                comm_group=log_item["comm_group"],  # 设置通信组
                comm_group_size=log_item["comm_group_size"],  # 设置通信组大小
                msg_size=log_item["msg_size"],  # 设置消息大小
                stage=log_item["stage"],  # 设置阶段
                src=log_item.get("src", None),  # 获取源地址，如果没有则为None
                dst=log_item.get("dst", None),  # 获取目标地址，如果没有则为None
                additional=log_item.get("additional", None),  # 获取附加信息，如果没有则为None
            )
        )

    def extend(self, new_workload):  # 扩展工作负载，将另一个Workload的日志追加到当前工作负载中
        self.workload.extend(new_workload.workload)  # 使用extend方法将new_workload中的日志项添加到当前工作负载列表中

    def dump(self, filename):  # 将工作负载保存到文件中
        folder_path = os.path.dirname(filename)  # 获取文件夹路径
        if folder_path and not os.path.exists(folder_path):  # 如果文件夹路径存在且文件夹不存在
            os.makedirs(folder_path)  # 创建文件夹
        default_folder_path = "results/mocked_workload/"  # 默认的文件夹路径
        if not os.path.exists(default_folder_path):  # 如果默认文件夹不存在
            os.makedirs(default_folder_path, exist_ok=True)  # 创建默认文件夹
        if "." in filename:  # 如果文件名包含扩展名
            filename = os.path.basename(filename).split(".")[0]  # 取文件名（不包含扩展名）
        filename = os.path.join("results/mocked_workload/", filename)  # 将默认路径与文件名结合
        csv_filename = filename + "_workload.csv"  # 给生成的文件命名为工作负载CSV文件
        with open(csv_filename, "w") as f:  # 打开文件进行写入
            f.write(self.workload[0].csv_header() + "\n")  # 写入工作负载的CSV头部
            for log_item in self.workload:  # 遍历所有工作负载中的日志项
                f.write(log_item.view_as_csv_line() + "\n")  # 将每个日志项按CSV格式写入文件
        print(f"Workload file generated:{csv_filename}")  # 输出文件生成的消息

    @staticmethod
    def load(filename):  # 静态方法加载工作负载文件
        filename = filename.split(".")  # 按照"."分割文件名
        filename[-1] = "pkl"  # 将文件扩展名改为pkl
        filename = ".".join(filename)  # 将文件名重新组合
        workload, args = pickle.load(open(filename, "rb"))  # 使用pickle加载文件并返回工作负载和其他参数
        return workload, args  # 返回工作负载和参数

