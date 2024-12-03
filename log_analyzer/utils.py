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

import math  # 导入数学模块，用于数学计算
from utils.utils import CommGroup, CommType, get_args  # 从utils.utils模块导入CommGroup、CommType和get_args

def convert_size_to_msg(size_bytes):  # 定义一个将字节大小转换为更易读的字符串格式的函数
    if size_bytes == 0:  # 如果输入的字节数为0
        return "0 B"  # 返回“0 B”字符串
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")  # 定义大小单位名称
    i = int(math.floor(math.log(size_bytes, 1024)))  # 计算字节数在这些单位中的位置，使用对数来确定大小级别
    p = math.pow(1024, i)  # 根据单位位置计算1024的幂
    s = round(size_bytes / p, 2)  # 根据单位位置计算大小并保留两位小数
    return "%s %s" % (s, size_name[i])  # 返回格式化的大小字符串，例如“1.5 MB”

def convert_msg_to_size(msg):  # 定义一个将格式化的大小字符串转换为字节数的函数
    if msg == "0B":  # 如果输入是“0B”
        return 0  # 返回0字节
    try:
        num, name = msg.split(" ")  # 将字符串按照空格分割，得到数值和单位
    except:  # 如果分割失败
        print(f"cannot convert msg into int")  # 打印错误信息
        return 0  # 返回0
    num, name = float(num), name.strip()  # 将数值转换为浮点数，去除单位的空格
    size_name = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]  # 定义单位名称列表
    if name not in size_name:  # 如果单位不在列表中
        return None  # 返回None，表示无法识别的单位
    p = math.pow(1024, size_name.index(name))  # 计算该单位对应的1024的幂
    return num * p  # 返回数值乘以对应的幂值，得到字节数

def calc_bw_log(comm_type: CommType, size, duration, group_size):  # 定义一个函数计算带宽和总线带宽
    n = group_size if group_size else 1  # 如果group_size有值则使用它，否则默认使用1
    duration /= 1000  # 将持续时间转换为秒（原来单位是毫秒）
    
    if comm_type in [CommType.all_gather, CommType.reduce_scatter]:  # 如果通信类型是all_gather或reduce_scatter
        # size *= n  # 原本的代码是将大小乘以group_size，但这里已注释掉
        tput = size / duration  # 计算吞吐量：总大小除以持续时间
        busbw = (size / duration) * ((n - 1) / n)  # 计算总线带宽，根据group_size调整
    elif comm_type == CommType.all_reduce:  # 如果通信类型是all_reduce
        tput = size / duration  # 计算吞吐量：总大小除以持续时间
        busbw = (size / duration) * (2 * (n - 1) / n)  # 计算总线带宽，乘以一个系数
    elif comm_type in [CommType.isend, CommType.irecv, CommType.barrier, CommType.computation]:  # 如果是异步发送、接收、屏障或计算类型
        return 0, 0  # 返回0，表示不需要带宽计算
    else:  # 其他通信类型，如broadcast、reduce、gather、scatter
        tput = size / duration  # 计算吞吐量：总大小除以持续时间
        busbw = tput  # 对于这些通信类型，吞吐量即为总线带宽

    tput /= 1024*1024*1024  # 将吞吐量从字节转换为GB
    busbw /= 1024*1024*1024  # 将总线带宽从字节转换为GB
    tput = round(tput, 2)  # 将吞吐量保留两位小数
    busbw = round(busbw, 2)  # 将总线带宽保留两位小数
    return tput, busbw  # 返回计算结果，分别是吞吐量和总线带宽
