import numpy as np


class Meter(object):    # 度量
    def __init__(self, name, val, avg):
        self.name = name
        self.val = val
        self.avg = avg

    def __repr__(self):
        return "{name}: {val:.6f} ({avg:.6f})".format(
            name=self.name, val=self.val, avg=self.avg
        )

    def __format__(self, *tuples, **kwargs):
        return self.__repr__()


class AverageMeter(object):  # 计算和保存平均值
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset() # 初始化的时候reset

    def reset(self): # 重新设置
        self.val = {}
        self.sum = {} # sum用来存总的数据
        self.count = {}

    def update(self, batch=1, **kwargs): # update数据 update(key = 1)
        val = {} # 新的字典
        for k in kwargs: # 对于传入的参数
            val[k] = kwargs[k] / float(batch) # val中的数据先进行batch级别的平均 # val['key'] = 1/1 = 1
        self.val.update(val) # 把val添加到self.val里面
        for k in kwargs:
            if k not in self.sum:  #如果是新的建
                self.sum[k] = 0  # 初始化到sum里
                self.count[k] = 0
            self.sum[k] += kwargs[k] # 再加
            self.count[k] += batch

    def __repr__(self): # 给程序员的显示接口， 可以直接打AverageMeter的变量来显示
        s = ''
        for k in self.sum:
            s += self.format_str(k)
        return s

    def format_str(self, attr):
        return "{name}: {val:.6f} ({avg:.6f}) ".format(
                    name=attr,
                    val=float(self.val[attr]),
                    avg=float(self.sum[attr]) / self.count[attr])

    def __getattr__(self, attr): # 如果访问的属性不再范围内，则调用这个方法 
        if attr in self.__dict__: # 如果这个attr在dict中的话
            return super(AverageMeter, self).__getattr__(attr) # 用他父类的getattr方法
        if attr not in self.sum: # 如果不在sum里面的话
            # logger.warn("invalid key '{}'".format(attr))
            # print("invalid key '{}'".format(attr)) # 报错（提示）
            return Meter(attr, 0, 0) # 返回 0 0
        return Meter(attr, self.val[attr], self.avg(attr)) # 返回正确的meter

    def avg(self, attr): # 计算avg
        return float(self.sum[attr]) / self.count[attr]


if __name__ == '__main__':
    # 用法
    avg = AverageMeter()                    # 初始化
    avg.update(time=1.1, accuracy=.99)      # 传入需要avg的参数
    avg.update(time=1.0, accuracy=.90)      # 多次传入来求平均

    print(avg)           # 将所有的平均值进行打印

    print(avg.time)      # 打印特定的值(str)
    print(avg.time.avg)  # 打印特定的平均(float)
    print(avg.time.val)  # 打印特定的值(float)
    print(avg.SS)        # 如果出现了没有的默认为0
    

