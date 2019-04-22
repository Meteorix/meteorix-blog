新ubuntu环境搭建
===============

### apt

替换apt源
*   源列表：http://wiki.ubuntu.org.cn/
*   清华源：https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/

安装``zsh/vim/git``

```shell
sudo apt-get update
sudo apt-get install zsh vim git
```

### git

```
git config --global core.editor "vim"
git config --global --edit  # 设置name和email
```


### zsh

配置**oh-my-zsh** https://github.com/robbyrussell/oh-my-zsh

```shell
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
```

如果替换默认shell失败，先设置密码再替换
```shell
passwd  # 设置密码
chsh zsh
```

解决zsh 有残留的问题，在 ``~/.zshrc`` 添加

```
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

> 参考：https://github.com/sindresorhus/pure/issues/300

解决tabname闪烁的问题，在``~/.zshrc``disable掉autotitle
```
DISABLE_AUTO_TITLE="true"
```


### vimrc

配置**vimrc** https://github.com/amix/vimrc
```shell
git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
sh ~/.vim_runtime/install_basic_vimrc.sh
```

还要在``~/.vimrc``加上
```
set nu  # 我喜欢加上行号
set fencs=utf-8,gbk   # 这一行的作用是告诉vim，打开一个文件时，尝试utf8,gbk两种编码
```
