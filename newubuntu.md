新ubuntu环境搭建
===============

### apt

1. apt源替换
    *   源列表：http://wiki.ubuntu.org.cn/
    *   清华源：https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/
2. 安装zsh/vim/git

```shell
sudo apt-get update
sudo apt-get install zsh vim git
```

### git

```
git config --global core.editor "vim"
```


### zsh

配置**oh-my-zsh** https://github.com/robbyrussell/oh-my-zsh

如果替换默认sh，先设置密码再替换
```shell
passwd
...
chsh zsh
```

解决zsh 有残留的问题，在 ``~/.vimrc`` 添加

```
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

> 参考：https://github.com/sindresorhus/pure/issues/300

### vimrc

配置**vimrc** https://github.com/amix/vimrc

还要在``~/.vimrc``加上
```
set nu  # 我喜欢加上行号
set fencs=utf-8,gbk   # 这一行的作用是告诉vim，打开一个文件时，尝试utf8,gbk两种编码
```

