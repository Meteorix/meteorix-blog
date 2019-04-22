---
title: Airtest刷刷抖音

date: 2019-2-5 22:41:32

tags: [Python,Airtest]

---

# Airtest刷刷抖音

用[Airtest](https://github.com/AirtestProject/Airtest)做点有意思的事情，先来刷个抖音？

[github仓库 airtest-douyin](https://github.com/Meteorix/airtest-douyin)

![ide01](/images/airtest-douyin/ide01.png)

## Get Started

### 环境准备

手边没有android手机，iOS又懒得搭[ios-tagent](https://github.com/AirtestProject/iOS-Tagent)的环境，于是采用最偷懒的方式：

*   [夜神模拟器](https://www.yeshen.com/)（可用安卓机代替）
*   [AirtestIDE](http://airtest.netease.com/)


夜神模拟器装上抖音，用起来跟手机上一样舒服。看了下模拟器占内存200m和CPU 12%左右，还不错。夜神自带了一个多开器，后面分布式刷抖音再玩玩

![nox](/images/airtest-douyin/nox.png)


### 录制第一版代码

打开AirtestIDE，按照[文档](http://airtest.netease.com/docs/cn/2_device_connection/3_emulator_connection.html#id2)连接好模拟器

![ide01](/images/airtest-douyin/ide01.png)

为了每次能用代码自动打开抖音，先用右上角的安卓助手查看一下抖音的package id

![assistant](/images/airtest-douyin/assistant.png)

手动加上代码

```python
APP = "com.ss.android.ugc.aweme"

stop_app(APP)
start_app(APP)
```

然后将AirtestIDE调到安卓App的录制模式，进行一些操作，对应的代码就录制下来了

![ide02](/images/airtest-douyin/ide02.png)


### 稍微调整代码

自动录制的代码不太好，稍微调整一下

```python
poco(boundsInParent="[0.03194444444444444, 0.02734375]").click()
```

直接改成通过`text`来识别按钮

```python
poco(text="我").click()
```

后面的上划操作，改成上划屏幕的``60%``

```python
poco("com.ss.android.ugc.aweme:id/ak2").swipe([0, -0.6])
```

然后按`F5`运行一遍，一切正常


### 一直刷下去

简单地修改下最后一行代码，就能一直刷下去了

```python
for i in range(10):
    poco("com.ss.android.ugc.aweme:id/ak2").swipe([0, -0.6])
    sleep(1)
```

### 好人点个赞

继续用IDE的录制功能，进行点赞操作，生成下面的代码

```python
poco("com.ss.android.ugc.aweme:id/al8").click()
```

原来抖音需要登录之后才能点赞，先手动登录吧，代码里面留个`TODO`

```python
if poco(text="输入手机号码").exists():
    # TODO: 自动登录
    print("先手动登录一下吧~")
    break
```

![ide03](/images/airtest-douyin/ide03.png)


然后我们截个图留念

```
snapshot()
```

再运行一下，效果非常好

![snapshot](/images/airtest-douyin/snapshot.png)


> tips: 点击IDE工具栏的`log`按钮，你还能看到每步操作的报告。



### 提交代码

这个脚本里面没有用到图像识别，单个py文件就够了。于是我们从``douyin.air``里面取出代码文件。这样可以用你喜欢的编辑器打开修改，用python直接运行了。

最终代码在[code/douyin.py](https://github.com/Meteorix/airtest-douyin/blob/master/code/douyin.py)，直接python运行。

```shell
python douyin.py
```

### To be continued

*   录屏替代截图
*   多开&分布式
*   图像识别小姐姐点赞
