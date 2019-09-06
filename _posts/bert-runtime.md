---
title: BERT Runtime

date: 2019-8-2 11:58:11

tags: [cuda,深度学习]

---

# BERT Runtime

最近继续怼[BERT](https://arxiv.org/abs/1810.04805)，项目大部分模型都上了BERT，真香啊。

本来一直在使用`PyTorch JIT`来解决加速和部署的问题，顺手还写了个[service-streamer](https://github.com/ShannonAI/service-streamer)来做web和模型的中间件。
正好上个月NVIDIA开源了基于`tensorrt`的BERT代码，[blog](https://devblogs.nvidia.com/nlu-with-tensorrt-bert/)号称单次`inference`只用2.2ms，比cpu快20倍。但是正确的问法是：这东西能比`tf/pytorch`快多少呢？

于是从TensorRT开始，认真学习了一波NVIDIA的BERT实现。并做了下性能Benchmark对比tf和pytorch，结论是gpu时间能快15%-30%。主要归因于对BERT的计算图优化，自己实现了4个cuda kernel，另外避免了tf和pytorch等框架带来的overhead。

## Prerequisite

比较有用的几个背景知识：

1. 当然是BERT的[Paper](https://arxiv.org/abs/1810.04805)，[Tensorflow实现](https://github.com/google-research/bert)，[PyTorch实现](https://github.com/huggingface/pytorch-transformers)
1. Harvard写的著名解读[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
1. GPU和Cuda基础知识，很简单可以参考我的[cuda101](https://github.com/Meteorix/meteorix-blog/blob/master/_posts/cuda101.md)

## TensorRT

跟[TensorRT](https://github.com/NVIDIA/TensorRT)的编译斗争了一两天，整体还是比较顺畅，照着README：

1. 准备环境，常规c++/py编译环境和cuda环境，我是`Titan XP + cuda-10.0 + cuDNN-7.4`
1. 下载TensorRT的binary release。TensorRT本身并没有开源，而是提供了编译好的lib。开源的周边部分包括：
    * `include`头文件
    * `plugin`实现一些cuda扩展
    * `parser`实现不同格式模型文件的解析
1. Docker build编译用的镜像。
1. 在Docker容器里面编译TensorRT的lib和开源代码。

## TensorRT BERT

## Benchmark

|bs * seqlen|tensorrt c++|tensorrt py|tensorflow|pytorch|pytorch jit|
|-|-|-|-|-|-|
|1 * 328|9.9|9.9|17|16.3|14.8|
|32 * 328|7.3| |11.6|9.9|8.6|


## 计算图优化和Kernel优化


## Next Steps



