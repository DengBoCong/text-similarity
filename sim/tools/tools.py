#! -*- coding: utf-8 -*-
""" Coding Tools
"""

# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

import re
import sys


class ProgressBar(object):
    """ 进度条工具 """

    EXECUTE = "%(current)d/%(total)d %(bar)s (%(percent)3d%%) %(metrics)s"
    DONE = "%(current)d/%(total)d %(bar)s - %(time).4fs/step %(metrics)s"

    def __init__(self, total: int = 100, num: int = 1, width: int = 30, fmt: str = EXECUTE,
                 symbol: str = "=", remain: str = ".", output=sys.stderr):
        """
        :param total: 执行总的次数
        :param num: 每执行一次任务数量级
        :param width: 进度条符号数量
        :param fmt: 进度条格式
        :param symbol: 进度条完成符号
        :param remain: 进度条未完成符号
        :param output: 错误输出
        """
        assert len(symbol) == 1
        self.args = {}
        self.metrics = ""
        self.total = total
        self.num = num
        self.width = width
        self.symbol = symbol
        self.remain = remain
        self.output = output
        self.fmt = re.sub(r"(?P<name>%\(.+?\))d", r"\g<name>%dd" % len(str(total)), fmt)

    def __call__(self, current: int, metrics: str):
        """
        :param current: 已执行次数
        :param metrics: 附加在进度条后的指标字符串
        """
        self.metrics = metrics
        percent = current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + ">" + self.remain * (self.width - size - 1) + "]"

        self.args = {
            "total": self.total * self.num,
            "bar": bar,
            "current": current * self.num,
            "percent": percent * 100,
            "metrics": metrics
        }
        print("\r" + self.fmt % self.args, file=self.output, end="")

    def reset(self, total: int, num: int, width: int = 30, fmt: str = EXECUTE,
              symbol: str = "=", remain: str = ".", output=sys.stderr):
        """重置内部属性
        :param total: 执行总的次数
        :param num: 每执行一次任务数量级
        :param width: 进度条符号数量
        :param fmt: 进度条格式
        :param symbol: 进度条完成符号
        :param remain: 进度条未完成符号
        :param output: 错误输出
        """
        self.__init__(total=total, num=num, width=width, fmt=fmt,
                      symbol=symbol, remain=remain, output=output)

    def done(self, step_time: float, fmt=DONE):
        """
        :param step_time: 该时间步执行完所用时间
        :param fmt: 执行完成之后进度条格式
        """
        self.args["bar"] = "[" + self.symbol * self.width + "]"
        self.args["time"] = step_time
        print("\r" + fmt % self.args + "\n", file=self.output, end="")


def get_dict_string(data: dict, prefix: str = "- ", precision: str = ": {:.4f} "):
    """将字典数据转换成key——value字符串
    :param data: 字典数据
    :param prefix: 组合前缀
    :param precision: key——value打印精度
    :return: 字符串
    """
    result = ""
    for key, value in data.items():
        result += (prefix + key + precision).format(value)

    return result
