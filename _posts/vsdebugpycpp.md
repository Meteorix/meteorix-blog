---
title: vs2017混合调试py/c++

date: 2019-2-13 23:51:32

tags: Python

---

# vs2017混合调试py/c++


## python卡死了怎么办?

一般的python程序卡死，可以用pycharm debug。但是有时候是python和c/c++库混合开发，比如pyqt或者boost python程序卡死，就非常难查。以前都是二分法注释代码查找问题，异常低效。

于是我尝试了vs2017的新功能：python & c++ 混合调试 [Debug Python and C++ together](https://docs.microsoft.com/en-us/visualstudio/python/debugging-mixed-mode-c-cpp-python-in-visual-studio?view=vs-2017)

## vs2017 python

vs一直被称为宇宙最强IDE，貌似python支持是vs2017新加的功能。相比pycharm之类的python编辑器，vs2017最大的优势就是在python和native c/c++代码的混合调试。包括：

*   python / c++ 调用栈合并显示
*   python / c++ 都可以断点
*   python / c++ 代码之间step
*   python / c++ 对象都可以watch

由于python的性能瓶颈，很多时候混合开发需要同时调试，这正是我们需要的！

## 环境搭建

下面以`Sunshine`(一个PyQt5程序)为例

1. 安装vs2017 python支持
2. 导入Sunshine项目，会在主目录生成Sunshine.sln，以后都可以双击打开了

    ```New Project->Python->From Existing Python Code```
    ![image.png](/images/vsdebugpycpp/5c3d9b205e60273aadf4650714DcRPJX.png)
3. 按照官网文档配置[Enable mixed-mode debugging in a Python project](https://docs.microsoft.com/en-us/visualstudio/python/debugging-mixed-mode-c-cpp-python-in-visual-studio?view=vs-2017#enable-mixed-mode-debugging-in-a-python-project)
4. 安装python的`pdb`文件，3.6直接用python installer安装，参考[这里](https://docs.microsoft.com/en-us/visualstudio/python/debugging-symbols-for-mixed-mode-c-cpp-python?view=vs-2017#download-symbols)
5. 配置`pdb`文件，主要是qt5的。qt5.9之后提供了pdb文件下载，[5.11.2的在这里](https://download.qt.io/archive/qt/5.11/5.11.2/)（PyQt最新版5.11.3，用的是Qt5.11.2，别问我为什么知道）
6. 选择`Sunshine.py` **F5**启动！

启动会比较慢，从输出窗口可以看出vs再加载各个py文件的symbol。。。（额，毕竟是试验性的功能）

## 让我们试试吧

### 卡死时候暂停

假设我们现在的Sunshine卡死了，这时候点击vs上的暂停按钮，就能立马停住。

![image.png](/images/vsdebugpycpp/5c3d9dbcaa49f15c3726191dzKYsU5Qw.png)

回想起Pycharm停不住的恐惧！毕竟vs是native的暂停，宇宙第一！

### 强大的混合调试

这时候就能使用强大的**混合调试**功能了，包括但不限于：
*   同时查看python/c++ callstack
*   callstack双击跳转到源码
*   切换线程
*   查看locals

![image.png](/images/vsdebugpycpp/5c3d9e54a7f2529830bb770bjqP16HQy.png)

### 混合的单步调试

以前python调到Qt里面去之后，就不知道发生了什么，现在可以从python step到c++中。

![image.png](/images/vsdebugpycpp/5c3da03a96dee435e6604c4aVmHaq5uC.png)

双击callstack中的c++ frame，vs会提示你打开。当然，这需要你本地下载了qt的c++代码。

然后你就可以在c++中打断点，做各种基本操作了

## 换成Attach模式

vs Debug启动python项目真的超级慢（感觉是有bug），而且我们经常是使用过程中进程卡死，这时候就需要Attach到卡死的python进程。

Debug->Attach to Process

![image.png](/images/vsdebugpycpp/5c3da1cb7f9d2a99198674256wRjQZ2J.png)

连上之后所有功能一样使用
