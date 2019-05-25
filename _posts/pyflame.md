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

    这个理解起来非常直白，在每个函数的开始和结束时计时，然后统计每个函数的执行时间。python自带的`Profile`和`cProfile`模块就是这个原理。但是插桩会有两个问题，一是插桩计时本身带来的性能消耗`overhead`极大，二是对一个正在运行的程序没法插桩。

2. 采样

    采样是注入正在运行的进程，高频地去探测函数调用栈。根据大数定理，探测到次数越多的函数运行时间越长。pyflame正是基于采样的原理。统计结果用另一个神器——火焰图[flamegraph](http://www.brendangregg.com/flamegraphs.html)展示，我们就可以完整地了解cpu运行状态。


# 使用pyflame和火焰图

官方提供了pyflame源码，需要自己编译对应python版本，然后调用flamegraph生成火焰图svg。具体可以看[官方文档](https://pyflame.readthedocs.io/en/latest/)。为了方便自己人使用，我写了个pyflame-server。

## pyflame-server

[https://github.com/Meteorix/pyflame-server](https://github.com/Meteorix/pyflame-server)

pyflame-server使用步骤很简单：
1. 下载编译好的pyflame二进制文件，支持py2.6/2.7/3.4/3.5/3.6/3.7
2. pyflame启动python进程，或者attach到正在运行的进程，得到profile.txt
3. 上传profile.txt，查看火焰图页面

pyflame-server是基于flask简单的web项目，欢迎参与开发。

## 如何读图

横向是采样次数的占比，越长的span表示调用次数越多，即时间消耗越多

纵向是函数调用栈，从底向上，最下层是可执行文件的入口

每个span里面显示了文件路径、函数名、行号、采样次数等信息，可以自己缩放svg图看看。

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

linux操作系统提供了一个强大的系统调用`ptrace`，让你可以注入任意进程(有sudo权限就是可以为所欲为)查看当前的寄存器、运行栈和内存，甚至可以任意修改别人的进程。

著名的`gdb`也是利用ptrace实现。比如打断点，其实就是保存了当前的运行状态，修改寄存器执行断点调试函数，之后恢复保存的运行状态继续运行。

## PyThreadState

有了ptrace之后，我们可以通过python的符号找到`PyThreadState`，也就是python虚拟机保存线程状态（包括线程调用栈）的地方。然后通过`PyThreadState`，拿到虚拟机中的py调用栈，遍历栈帧反解出python的函数名、所在文件行号等信息。后面就是高频采样和输出统计结果了。

![PyThreadState](/images/pyflame/Python-Thread-State.png)

这部分如果想深入了解，可以看[python虚拟机](https://github.com/Meteorix/pysourcenote/blob/master/vm.md)这篇介绍。

# 如果我们想profile c/c++呢

目前深度学习程序多半是c++和python混合开发。有时候我们看到python里面的一行代码，其实底层调用了几十行c++代码。这时候为了搞清楚性能消耗到底在哪，我们需要同时profile python和c++。这里提供基于pyflame和libunwind的实现思路。

## libunwind

`libunwind`是另一个开源的c++库，同样利用`ptrace`实现了远程注入和解c栈的接口。于是我们可以在一次ptrace断点时，同时解出c栈和py栈，然后用一个巧妙的办法将两个栈merge到一起。再修改一下`flamegraph`的配色，可以得到c/c++栈和py栈混合profile的效果。

![c/py混合profile](profile_c.svg)

通过上面的火焰图，我们能清楚的看到每个python调用，实际上调用的底层c++函数。

另一个好处是，python没有占用GIL的时候，我们可以看到c++的调用栈。

to be continued...
