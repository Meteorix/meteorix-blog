---
title: gdb调试cpython

date: 2019-2-13 23:11:32

tags: Python

---

# gdb调试cpython

主要参考这篇文章：

*	https://www.podoliaka.org/2016/04/10/debugging-cpython-gdb/
*	https://blog.alswl.com/2013/11/python-gdb/


## 同时debug c栈和py栈

在调试脚本卡死的时候特别有用，如下图

![image](/images/gdbpython/bt.png)

## 安装python-dbg

```
sudo apt-get install gdb python-dbg
```

<!--more-->


``python-dbg``包含symbol和py-bt

会自动把libpython.py装到gdb的auto-load目录，并且保证后面的子目录跟python的目录地址一样

```
➜  ~ which python2.7
/usr/bin/python2.7
➜  ~ ls /usr/share/gdb/auto-load/usr/bin/          
python2.7-dbg-gdb.py  python2.7-gdb.py
```

### 其他版本python安装gdb dbg

由于自己在服务器上安装了多个版本的python，比如自己用源码编译的py3.7

```
➜  ~ which python3.7
/usr/local/bin/python3.7
```

然后可以在python的源码里面找到``Toos/gdb/libpython.py``
> https://github.com/python/cpython/tree/3.7/Tools/gdb

按照上面的目录规则cp到gdb的auto-load，保证调试python3.7进程的时候能找到

```
➜  ~ ls /usr/share/gdb/auto-load/usr/local/bin/          
python3.7-dbg-gdb.py  python3.7-gdb.py
```

### attach到python进程

```
ps -x | grep python
gdb -p <pid>
```

### 常用指令

```
bt    # 当前C调用栈
py-bt  # 当前Py调用栈
py-list  # 当前py代码位置
info thread   # 线程信息
thread <id>   # 切换到某个线程
thread apply all py-list  # 查看所有线程的py代码位置
ctrl-c  # 中断
```

py-bt如果遇到中文编码问题
export LC_CTYPE=C.UTF-8


### 配合gdb dashboard，更方便一点

https://github.com/cyrus-and/gdb-dashboard
