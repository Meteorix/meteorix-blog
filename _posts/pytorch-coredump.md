---
title: 记一次pytorch的coredump调试

date: 2019-4-30 19:47:32

tags: [Python,机器学习]

---


1. 两个算法小哥写了个flask的web服务器，跑ocr的服务。单个请求是好的，多请求就崩了, log就显示：
    ``Segmentation fault (core dumped)``

    >   “wtf is core dump?”

    just google it, 简单说就是程序崩溃之后dump出来的一些信息

2. 调试到了凌晨2点，加了一堆log，还是没有找到原因，看起来很随机。第二天找到我：

    > python coredump了怎么办？

3. 为了debug这个coredump，先打开系统设置
    ```
    ulimit -c unlimited
    ```

4. 再次制造coredump，在当前目录下就产生了一个4G的core文件
    ```
    -rw------- 1 liuxin root 4.2G 4月  29 14:21 core
    ```

5. core文件就是保存了crash时候的所有信息，调用栈、线程、内存等。然后运用我们的linux c++开发经验，可以知道用gdb来调试coredump文件

    ```
    gdb <executable> <coredump file>
    ```

6. 等等，这里我们是python程序崩了，为什么没有traceback？因为在调用到pytorch的c++代码时，直接segmentfault，并没能等到python的退出机制打印出traceback，直接崩了。但是操作系统能产生coredump文件，这是我们的救命稻草。

<!-- more -->

7. 直接用gdb调试python的coredump文件

    ```
    gdb python ./core
    ```
    显示一大堆毫无意义的信息，因为你的python还没有装上“符号文件”，操作系统的栈里面的地址需要对上符号信息，才能解析出c/c++代码

8. 找到你的python版本对应的符号文件，装上
    ```
    sudo apt install python3.6-dbg
    ```

9. 然后就能够看到c栈了
    ```
    (gdb) bt
    #0  0x00007fffd721ec4b in ?? () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
    #1  0x00007fffd734ed06 in ?? () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
    #2  0x00007fffd7138a46 in ?? () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
    #3  0x00007fffd7138c73 in ?? () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
    #4  0x00007fffd7290f40 in cuLaunchKernel () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
    #5  0x00007fffa72f2dcb in cudart::cudaApiLaunchCommon(void const*, bool) ()
       from /home/liuxin/venv/lib/python3.6/site-packages/torch/lib/libcaffe2_gpu.so
    #6  0x00007fffa73105a8 in cudaLaunch () from /home/liuxin/venv/lib/python3.6/site-packages/torch/lib/libcaffe2_gpu.so
    #7  0x00007fffa73c70f1 in cublasStatus_t cublasGemv<float, float, float, 128, 4, 16, 4, 4>(cublasContext*, cublasOperation_t, int, int, float const*, float const*, int, float const*, int, float const*, float*, int) ()
       from /home/liuxin/venv/lib/python3.6/site-packages/torch/lib/libcaffe2_gpu.so
    #8  0x00007fffa73e0bae in cublasSgemmRecursiveEntry(cublasContext*, int, int, int, int, int, float const*, float const*, int, float const*, int, float const*, float*, int) () from /home/liuxin/venv/lib/python3.6/site-packages/torch/lib/libcaffe2_gpu.so
    #9  0x00007fffa744f8dd in cublasSgemmEx_internal(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, float const*, void const*, cudaDataType_t, int, void const*, cudaDataType_t, int, float const*, void*, cudaDataType_t, int, bool, bool) ()
       from /home/liuxin/venv/lib/python3.6/site-packages/torch/lib/libcaffe2_gpu.so
    #10 0x00007fffa7459ad2 in cublasSgemmExRecursiveEntry(cublasContext*, int, int, int, int, int, float const*, void const*, cudaDataType_t, int, void const*, cudaDataType_t, int, float const*, void*, cudaDataType_t, int, cublasGemmAlgo_t, int, int) ()
       from /home/liuxin/venv/lib/python3.6/site-packages/torch/lib/libcaffe2_gpu.so
      
    ...下面还有200多行...
    ```

    通过上面的c栈，大概能看出是崩在了pytoch的rnn模块。但是，我们的python代码哪里出问题了呢？

10. 这时候需要找的是python栈在哪里呢？很幸运的是，python是c写的，通过c栈和python的符合信息，我们能反解出python调用栈！而且gdb-python直接帮我们做好了这件事。

    在python对应版本的源码中找``Toos/gdb/libpython.py``
    > https://github.com/python/cpython/tree/3.6/Tools/gdb
    
    下载下来，然后在gdb中使用
    ```
    (gdb) source libpython.py
    (gdb) py-bt
    Traceback (most recent call first):
      <built-in method _cudnn_rnn of type object at remote 0x7fffd51dd640>
      File "/home/liuxin/venv/lib/python3.6/site-packages/torch/nn/_functions/rnn.py", line 288, in forward
        dropout_ts)
      File "/home/liuxin/venv/lib/python3.6/site-packages/torch/nn/_functions/rnn.py", line 324, in forward
        return func(input, *fargs, **fkwargs)
      File "/home/liuxin/venv/lib/python3.6/site-packages/torch/nn/modules/rnn.py", line 192, in forward
        output, hidden = func(input, self.all_weights, hx, batch_sizes)
      File "/home/liuxin/venv/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
        result = self.forward(*input, **kwargs)
      File "/home/liuxin/shannon_vision/shannon_vision_models/shannon_vision_nanjing_bank/crnn/master/models/crnn.py", line 69, in forward
        recurrent, _ = self.rnn(x)
      File "/home/liuxin/venv/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
        result = self.forward(*input, **kwargs)
      File "/home/liuxin/venv/lib/python3.6/site-packages/torch/nn/modules/container.py", line 91, in forward
        input = module(input)
      File "/home/liuxin/venv/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
        result = self.forward(*input, **kwargs)
      File "/home/liuxin/shannon_vision/shannon_vision_models/shannon_vision_nanjing_bank/crnn/master/models/crnn.py", line 57, in forward
        out = self.rnn(conv)
      File "/home/liuxin/venv/lib/python3.6/site-packages/torch/nn/modules/module.py", line 477, in __call__
        result = self.forward(*input, **kwargs)
      File "/home/liuxin/shannon_vision/shannon_vision_models/shannon_vision_nanjing_bank/crnn/master/models/recognizer.py", line 65, in recognize
        pred_matrix = self.model(charline_image)
      File "xiang_server.py", line 93, in run_crnn_server
        ocr_pred, matrix = recognizer.recognize(img)
      File "/home/liuxin/venv/lib/python3.6/site-packages/flask/app.py", line 1799, in dispatch_request
     
    ...下面还有三十多行...
    ```

11. 找到问题了，原来是这个``recognize``函数，让我们来fix it。通过"经验"，我们猜到这是多线程调用gpu代码导致的问题，先试试加锁

    ```
    lock = threading.Lock()
    ...
    with lock:
        ocr_pred, matrix = recognizer.recognize(img)
    ...
    ```
    果然加了锁就再也没有诡异的coredump了，于是我们猜想：
    > gpu代码调用可能不是thread-safe的，加个线程锁比较安全。
    
12. 到底pytorch的函数是不是线程安全的呢？我们的算法小哥哥构造了一段代码验证:
    ```
    # -*- coding: utf-8 -*-
    from torch import nn
    import torch
    from threading import Thread
    
    
    class BidirectionalLSTM(nn.Module):
        """构建一个简单的网络"""
        def __init__(self, n_in, n_hidden, n_out):
            super(BidirectionalLSTM, self).__init__()
            self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True)
            self.embedding = nn.Linear(n_hidden * 2, n_out)
    
        def forward(self, x):
            recurrent, _ = self.rnn(x)
            t, b, h = recurrent.size()
            t_rec = recurrent.view(t * b, h)
            out = self.embedding(t_rec)  # [t * b, n_out]
            out = out.view(t, b, -1)
    
            return out
    
    
    lstm = BidirectionalLSTM(256, 256, 256)
    x = torch.randn(12,100,256)
    device = torch.device('cuda:0')
    lstm = lstm.to(device)
    x = x.to(device)
    
    
    def print_time(threadName):
        count = 0
        while count < 1000:
            lstm(x)
            count += 1
            print(threadName, ":", count)
    
    
    def run_rnn_multi_threaded():
        for i in range(5):
            name = "Thread-%d" % i
            t = Thread(target=print_time, args=(name,), name=name)
            t.start()
    
    
    if __name__ == '__main__':
        run_rnn_multi_threaded()
    ```
    运行没多久果然coredump了
    ```
    ...
    Thread-0 : 79
    Thread-4 : 74
    Thread-3 : 74
    Thread-2 : 74
    Thread-1 : 76
    Thread-0 : 80
    Thread-4 : 75
    Thread-3 : 75
    Thread-2 : 75
    Thread-1 : 77
    Thread-0 : 81
    Segmentation fault (core dumped)
    ```

13. 查了下``pytorch thread safety``，很多答案说默认是线程安全的。于是我提了个issue给官方 https://github.com/pytorch/pytorch/issues/19913 静待回复

14. 虽然只用两行代码就临时解决了问题，通过这次调试经历，我们可以获得知识点：core dump/gdb/debug symbol/thread-safety，以及python调试的终极武器：[gdb-python](https://github.com/Meteorix/meteorix-blog/blob/master/_posts/gdbpython.md)

    > 在windows上我们更有宇宙最强的visual studio，提供了等价的[python/c混合调试](https://github.com/Meteorix/meteorix-blog/blob/master/_posts/vsdebugpycpp.md)

15. 两天后，官方回复issue：pytorch新版本修复了很多线程安全问题。于是我升级pytorch 1.1，用上面的代码验证，果然没问题。为了提升web服务器的性能，我们准备升级pytorch版本，就可以去掉那个令人不爽的锁。

16. 关于这个问题，还有很多未解之谜 to be continued...
