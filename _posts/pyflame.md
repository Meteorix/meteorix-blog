---
title: pyflame原理解析

date: 2019-5-24 20:12:22

tags: [Python,C++]

---

[pyflame](https://github.com/uber/pyflame)是uber开源的一个python性能profile工具。

> 没有profile的优化就是耍流氓

说起python，被吐槽最多的就是慢。根据上面的名言，第一步我们需要一个性能profile工具。

# profiler的两种原理

profile工具基本分成两种思路：

1. 插桩
    这个理解起来非常直白，在每个函数的开始和结束时计时，然后统计每个函数的执行时间。python自带的`Profile`和`cProfile`就是这个原理。但是插桩会有两个问题，一是插桩计时本身带来的性能消耗`overhead`极大，二是对一个正在运行的程序没法插桩。

2. 采样
    采样是高频地去探测正在运行的函数调用栈，然后根据大数定理，探测到次数越多的函数运行时间越长。pyflame正是基于采样的原理，统计结果用火焰图[flamegraph](http://www.brendangregg.com/flamegraphs.html)展示，可以让我们完整了解cpu运行状态。


# 使用pyflame和火焰图

## flame-server

# pyflame原理解析

ptrace

tstate

# 如果我们想看c/c++函数呢

libunwind
