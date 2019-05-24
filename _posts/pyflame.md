---
title: Pyflame原理解析

date: 2019-5-24 20:12:22

tags: [Python,C++]

---

[Pyflame](https://github.com/uber/pyflame)是uber开源的一个python性能profiler。

> 没有profile的优化就是耍流氓

说起python，被吐槽最多的就是慢。根据上面的名言，首先我们需要一个性能profiler。

# profiler的两种原理

profile工具基本分成两种思路：

1. 插桩

    这个理解起来非常直白，在每个函数的开始和结束时计时，然后统计每个函数的执行时间。python自带的`Profile`和`cProfile`就是这个原理。但是插桩会有两个问题，一是插桩计时本身带来的性能消耗`overhead`极大，二是对一个正在运行的程序没法插桩。

2. 采样

    采样是注入正在运行的进程，高频地去探测函数调用栈。根据大数定理，探测到次数越多的函数运行时间越长。pyflame正是基于采样的原理。统计结果用另一个神器——火焰图[flamegraph](http://www.brendangregg.com/flamegraphs.html)展示，我们就可以完整地了解cpu运行状态。


# 使用pyflame和火焰图

## flame-server

## 即时维护的分支

Uber写pyflame的哥们离职了，还没人接手这个项目。于是我自己维护了一个分支，做了几点微小的工作：
1. 修复py2.7编译脚本
1. 修复py3.7兼容性问题，感谢pr
1. 修复anaconda的兼容性问题，感谢另一个pr
1. 增加dockerfile，enable所有abi，目前同时支持py2.6/2.7/3.4-3.7
1. 试图增加c/c++ profile

# How magic happens

Uber官方博客给了一篇[由浅入深的讲解](https://eng.uber.com/pyflame/)，这里简单提几个关键点。

## ptrace

linux操作系统提供了一个强大的函数`ptrace`，让你可以注入任意进程(有sudo权限就是可以为所欲为)查看当前的寄存器、运行栈和内存，甚至可以任意修改别人的进程。

著名的`gdb`也是利用ptrace实现。比如打断点，其实就是保存了当前的运行状态，修改寄存器执行断点调试函数，之后恢复保存的运行状态继续运行。

## tstate

有了ptrace之后，我们可以通过python的符号找到`tstate`，也就是python虚拟机保存线程调用栈的地方。然后通过`tstate`，拿到虚拟机中的py调用栈，反解出python的函数名、所在文件行号等信息。后面就是高频采样和输出统计结果了。

这部分如果想深入了解，可以看[python虚拟机](https://github.com/Meteorix/pysourcenote/blob/master/vm.md)这篇介绍。

# 如果我们想profile c/c++呢

libunwind
