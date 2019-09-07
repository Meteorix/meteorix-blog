---
title: BERT Runtime

date: 2019-9-7 15:03:33

tags: [cuda,深度学习]

---

# BERT Runtime

最近继续怼[BERT](https://arxiv.org/abs/1810.04805)，项目大部分模型都上了BERT，真香啊。

本来一直在使用`PyTorch JIT`来解决加速和部署的问题，顺手还写了个[service-streamer](https://github.com/ShannonAI/service-streamer)来做web和模型的中间件。
正好上个月NVIDIA开源了基于`TensorRT`的[BERT代码](https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT)，官方[blog](https://devblogs.nvidia.com/nlu-with-tensorrt-bert/)号称单次`inference`只用2.2ms，比cpu快20倍。但是正确的问法是：这东西能比TF/PyTorch快多少呢？

于是从[TensorRT](https://developer.nvidia.com/tensorrt)开始，认真学习了一波NVIDIA的BERT实现。并做了性能Benchmark对比TensorFlow和PyTorch，结论是gpu时间能快**15%-30%**。主要归因于对BERT的计算图优化，自己实现了4个cuda kernel，另外避免了TensorFlow和PyTorch等框架带来的overhead。

## Prerequisite

比较有用的几个背景知识：

1. 当然是BERT的[Paper](https://arxiv.org/abs/1810.04805)，[Tensorflow实现](https://github.com/google-research/bert)，[PyTorch实现](https://github.com/huggingface/pytorch-transformers)
1. Harvard写的著名解读[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
1. GPU和Cuda基础知识，很简单可以参考我的[cuda101](https://github.com/Meteorix/meteorix-blog/blob/master/_posts/cuda101.md)

## TensorRT

**TensorRT**是NVIDIA官方推出的inference引擎，建立在CUDA之上。可以对``TensorFlow/PyTorch``等框架训练出来的模型进行CUDA优化，达到更高的inference性能。同时支持低精度参数、跨平台部署等，总之就是对自己家的GPU使用的最好。

<!--more-->

跟[TensorRT](https://github.com/NVIDIA/TensorRT)的编译斗争了一两天，整体还是比较顺畅，照着``README``：

1. 准备环境，常规c++/py编译环境和cuda环境，我是`Titan XP + cuda-10.0 + cuDNN-7.4`
1. 下载TensorRT的binary release。TensorRT本身并没有开源，而是提供了编译好的lib。开源的周边代码包括：
    * `include`头文件
    * `plugin`实现一些cuda扩展
    * `parser`实现不同格式模型文件的解析
1. Docker build编译用的镜像。
1. 在Docker容器里面编译TensorRT的lib和开源代码。

## TensorRT BERT

TensorRT的BERT实现代码在[demo/BERT](https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT)目录下，主要提供了：
1. 针对BERT进行了4个计算图优化，用cuda实现了几个fusion的kernel，封装成TensorRT的plugin
1. TensorFlow模型文件转TensorRT模型文件的脚本
1. C++和python版API和完整的BERT inference代码。

还是看``README``，以`SQuAD(QA)`模型为例提供了完整的使用步骤：
1. 下载BERT在SQuAD上finetune的TF模型文件，或者你也可以用自己finetune的模型文件
1. 使用转换脚本将TF模型文件转换成TensorRT模型文件
1. 使用另一个脚本将模型、参数、输入问题转换为Tensor形式的输入输出
1. 编译C++可执行文件，即可测试加速后的模型和输入输出，并保存为`bert.engine`

这个`bert.engine`文件，就可以单独使用了。既可以用C++ API或Python API加载后使用，也可以使用TensorRT Serving的docker直接加载做service。

### Python API

NVIDIA也提供了Python API来完成上面的几个步骤，需要多编译一些python binding。不过既然我都编好了C++版本，就只用Python API做inference。后面测试结果可以看出，Python API在模型inference的性能上与C++版本比几乎没有损耗。

Python API的使用依赖[pycuda](https://developer.nvidia.com/pycuda)，这是另一个官方库，用来做Python与CUDA之间的直接交互。这里包括分配显存、内存与显存之间copy tensor等。读取`bert.engine`执行inference则是使用TensorRT发布的whl包。


### 复现NVIDIA提供的性能数据

NVIDIA官方数据是在`batchsize=1，seqlen=128`时测试的。在我们的Titan XP上分别使用C++和Python API，GPU时间都在`2.6ms`左右，基本复现了官方数据。

![gpucpu.png](/images/bert-runtime/gpucpu.png)

比较有意思的是，明明与pytorch和tensorflow等框架比更能说明bert优化的效果，可能是为了diss cpu好卖gpu卡吧 :P

下面我们就来正经做一下Benchmark

## Benchmark

对于BERT的inference，很大一部分时间消耗在预处理上，即将输入的文字``tokenize``为`index`，执行`padding`和`masking`，再组装成`tensor`。而我们这里的benchmark只关心GPU执行inference的性能。所以我们的计时代码只包含GPU时间，也就是tensor输入到输出的时间，排除掉前后处理时间，另外包含tensor在CPU和GPU之间copy的时间。

### 环境

**GPU版本**
* GPU Titan XP
* Cuda 10.0
* Cudnn 7.5

**Python3.6版本**
* Torch==1.2.0
* TensorFlow==1.14.0
* tensorrt==5.1.5.0

**BERT实现**
* tensorrt基于 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT
* TensorFlow基于 https://github.com/google-research/bert
* PyTorch基于 https://github.com/huggingface/pytorch-transformers

**模型**
* bert-base 12层，SQuQA finetuned
* 相同的模型参数，分别转换为tensorrt/tf/pytorch模型文件

### SQuAD任务

使用SQuAD(QA)任务进行测试
```
# 输入文章和问题
Passage: TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps.

Question: What is TensorRT?

# 输出答案
Answer: 'a high performance deep learning inference platform'
```

使用上面的QA任务样例，输入`padding`到`Sequence Length=328`，`Batch Size`分别使用`1`和`32`。测量100次取平均单句时间，单位是`ms`

### 结论

|bs * seqlen|tensorrt c++|tensorrt py|tensorflow|pytorch|pytorch jit|
|-|-|-|-|-|-|
|1 * 328|9.9|9.9|17|16.3|14.8|
|32 * 328|7.3| |11.6|9.9|8.6|

注：
1. TensorFlow接口封装不太熟悉，仅供参考，目测与PyTorch无jit版本性能差不多
2. TensorRT py接口暂时没实现多batch的inference，目测与c++版本性能差不多
3. 所有测试GPU利用率都接近`100%`，说明没有什么GPU之外的阻塞代码

结论：
1. TensorRT比PyTorch快39%-26% 
2. TensorRT比PyTorch jit快33%-16%

## 计算图优化和kernel优化

那么我们来看看TensorRT实现的BERT，到底做了哪些优化。

![bert.png](/images/bert-runtime/bert.png)

上面的计算图给了一个BERT `Transformer Encoder`的总览。对``Transformer``还不熟悉的话，可以回头看看Harvard写的著名解读[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)。总共有4点计算图优化，3点在`Transformer`中：
1. `gelu`激活函数的kernel实现
2. `skip`和`layernorm`函数的fusion
3. `Q/K/V`三个矩阵的合并乘法和转置

上面的前3个优化在12层``Transformer``中都会用到，所以性价比很高。第4点优化在最底层`BERT Embedding`层：

4. `embedding`和`layernorm`的fusion

下面分别看看4处优化是如何实现的，我也是趁此机会了解计算图优化和cuda kernel函数的编写。

### Gelu

按照`gelu`的公式，如果每步分开计算，每步kernel调用都会进行一次global显存的读写。

![gelu.png](/images/bert-runtime/gelu.png)

> 由于gpu的硬件特性，`global memory`的访问速度非常慢（相对计算而言），这里可以参考前一篇笔记中的[gpu设计和内存结构](https://github.com/Meteorix/meteorix-blog/blob/master/_posts/cuda101.md#gpu%E8%AE%BE%E8%AE%A1)。

于是TensorRT就写一个gelu的kernel，一次kernel函数调用解决问题，只用一次显存读写。

https://github.com/NVIDIA/TensorRT/blob/release/5.1/demo/BERT/plugins/geluPlugin.cu

```cpp
// constants for approximating the normal cdf
constexpr float A = 0.5;

constexpr float B = 0.7978845608028654; // sqrt(2.0/M_PI)

constexpr float C = 0.035677408136300125; // 0.044715 * sqrt(2.0/M_PI)

template <typename T, unsigned TPB>
__global__ void geluKernel(const T a, const T b, const T c, int n, const T* input, T* output)
{

    const int idx = blockIdx.x * TPB + threadIdx.x;

    if (idx < n)
    {
        const T in = input[idx];
        const T cdf = a + a * tanh(in * (c * in * in + b));
        output[idx] = in * cdf;
    }
}
```

对比PyTorch JIT

```
@torch.jit.script
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(0.797884 * (x + 0.044715 * torch.pow(x, 3))))

print(gelu.graph)
```

![gelujit.png](/images/bert-runtime/gelujit.png)

从计算图上看确实每一步是单独计算，除了`tanh`这种内置的函数，其他都要一层层函数调用。

不过，在PyTorch 1.2的最新代码中，我发现`gelu`也是用了内置的cuda实现，两者几乎等价。

### Skip and Layer-Normalization

LayerNorm层的PyTorch实现
```
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```

忽略掉几个不重要的参数，主要是计算`mean`和`std`，各需要遍历一次所有输入参数。

加上`LayerNorm`之前的`Skip`层，一共需要遍历三次所有输入参数。

```python
x = LayerNorm(x + Sublayer(x))
```

根据上面说的GPU硬件和显存特性，启动三次kernel函数、遍历三次，都是消耗较大的。
所以优化为：
1. 算`Skip`层的同时计算`x`和`x^2`的平均值
2. 再算`LayerNorm`层时直接用`x`和`x^2`的平均值得到`mean`和`std`

    ```python
    std = sqrt(mean(x^2) - mean(x)^2)
    ```

看代码的时候没明白，跟yuxian手推了一波这个公式（逃

![std.jpg](/images/bert-runtime/std.jpg)


这样将三次遍历fusion成一次，省去了读写global显存的时间

cuda代码实现：

https://github.com/NVIDIA/TensorRT/blob/release/5.1/demo/BERT/plugins/skipLayerNormPlugin.cu

类似的还有`Embeding+LN`的fusion，理论上所有`LN`前面有一次遍历的都可以先算出来`x`和`x^2`的均值，省去两次遍历：

https://github.com/NVIDIA/TensorRT/blob/release/5.1/demo/BERT/plugins/embLayerNormPlugin.cu



### QKV 优化

有了上面的基础，这里的两个优化比较容易理解，直接看图和代码

![qkv.png](/images/bert-runtime/qkv.png)

1）``QKV``本来是分别成三个矩阵然后转置，现在变成成一个三倍大的矩阵转置，再slice

https://github.com/NVIDIA/TensorRT/blob/release/5.1/demo/BERT/plugins/qkvToContextPlugin.cu


2）``Scale+Softmax``，在scale那一次遍历同时求得`exp(x)`，减少一次遍历

https://github.com/NVIDIA/TensorRT/blob/e47febadb256d94f65efe0f1eac54c7caedd65d4/demo/BERT/plugins/pluginUtil.h#L220


### 异步执行

TensorRT的blog特别提了一下异步执行。由于CPU和GPU是异构的，在CPU和GPU之间copy tensor、GPU runtime执行计算都是异步完成的。不强制同步可以增加整个流程的吞吐量`througput`。Profile的时候需要特别注意这个异步的时间。这点在TensorRT的python代码中也能看到，实现的非常仔细。

![async.png](/images/bert-runtime/async.png)

PyTorch实际上也是异步的，所以这点TensorRT没什么优势

## 如何使用

分析完TensorRT的BERT优化，我们看看能怎么用起来。

这30%左右的inference速度提升还是很香的，可能的用法有：

1. 使用Python API，替换tf/pytorch的BERT实现，前后处理代码不用动
1. 使用C++ API，封装前后处理C++代码，编译成二进制发布
1. 直接使用[tensorrt-inference-server](https://github.com/NVIDIA/tensorrt-inference-server)，server只处理tensor，前后处理需要另外实现


这三种用法都需要将tf/pytorch训练(finetune)好的模型文件，转化为tensorrt的`.engine`文件：
1. 转换模型参数，每个任务的模型BERT最上层会稍有不同
1. 确定输入输出、batch_size等参数，生成tensor文件
1. 用前两部的结果生成`.engine`文件

### So, what's next?

根据项目的发展的阶段，考虑采用三种用法，主要先理顺``模型迭代--业务开发--部署``的流程。

> 再次感叹BERT真香，NLP领域幸好有BERT，才能搞这些优化。
