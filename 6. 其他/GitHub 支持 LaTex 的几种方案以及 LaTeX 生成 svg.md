### GitHub 支持 LaTex 的几种方案
1. [codecogs](http://www.codecogs.com/latex/eqneditor.php) 生成外部链接

优点：方便快捷，可以生成 gif、svg、png 等格式的图片，支持下载和直接外部链接引用。

缺点：收费，具体见 [名为开源，实际上每年都要收钱，这种策略好吗？](https://www.oschina.net/question/589241_133621)，[CODECOGS在线LaTeX公式编辑器收费陷阱！](https://blog.csdn.net/hdg34jk/article/details/78858067)。服务不可靠，如果提供服务的公司突然倒闭，那么以前写的博客都没用了。而且只能联网访问。

类似的还有 Google Chart，但是会面临被墙的风险。

2. [MathJax](https://www.mathjax.org/)

GitHub 因为安全问题不支持引入 js、css 文件，也不支持 `<script>、<style>` 等标签，所以不可行。

3. [GitHub App Texify](https://github.com/apps/texify)

优点: 不需要太多配置，直接像 LaTeX 一样使用，直接保存图片，可以本地预览。

缺点: 生成 `*.tex.md` 并保存 svg。图片文件，然后直接提交到整个仓库，非常愚蠢的一种做法，而且并不支持 GitHub Issue。

4. [GitHub with MathJax](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima?utm_source=chrome-app-launcher-info-dialog)

通过本地渲染 GitHub 界面支持 LaTeX，自欺欺人的一种方法，不推荐...

5. 本地生成图片并上传到图床或者 GitHub

不是太方便，但是可以解决很多已知问题。

<h2 id="LaTeX-生成-svg">LaTeX 生成 svg</h2>

参考该回答: [converting a latex code to mathml or svg code in python](https://stackoverflow.com/questions/9588261/converting-a-latex-code-to-mathml-or-svg-code-in-python#answer-16893390)。

通过 `latex` 命令生成 DVI 文件，然后使用 [dvisvgm](https://dvisvgm.de/) 生成 svg 图片。

通过模板转换:
```latex
\documentclass[paper=a5,fontsize=12pt]{scrbook}
\usepackage[pdftex,active,tightpage]{preview}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{tikz}
\begin{document}
\begin{preview}
\begin{tikzpicture}[inner sep=0pt, outer sep=0pt]
\node at (0, 0) {texCode}; % <--Put your tex-code here
\end{tikzpicture}
\end{preview}
\end{document}
```

缺点是会产生很多中间文件，需要安装 `LaTex` 环境，大到 3~4 G。

### 如何选择

目前我认为最好的方法是通过 [codecogs](http://www.codecogs.com/latex/eqneditor.php) 下载 gif 图片，然后上传到图床或者本地的仓库，可以解决服务不稳定的问题，至少可以保证自己的服务器或域名是可以访问的。

另外一种方法，自己搭建一个服务器，随便采用什么框架，然后通过上面介绍的 <a href="#LaTeX-生成-svg">LaTeX 生成 svg</a> 提供一个类似于 codecogs 的服务，返回 svg 图片。而且目前也有类似功能的，例如 [texoid](https://github.com/DMOJ/texoid)。

实现起来很简单，但我嫌麻烦，可以考虑上一种方法，我认为是在 GitHub 不支持 LaTeX 时的最优解。

### LaTex Fomula to SVG
将 LaTex 公式转 SVG 图片，有几种方法，转其他格式的图片可以使用其他的 API:
* codecogs API

访问有限制，需要额外的错误控制。请求、生成一张图片总共花费大概 ~2s，效果如下。

* tex2svg

来源于 [How to convert LaTeX equations to SVG?](https://askubuntu.com/questions/33196/how-to-convert-latex-equations-to-svg)。

安装命令行工具:

```shell
npm install --global https://github.com/mathjax/mathjax-node-cli.git
```

转换一个公式:

```shell
tex2svg '\sin^2{\theta} + \cos^2{\theta} = 1' > mathjax.svg
```
花费 ~2s，比 codecogs 略长。

* latex 和 dvisvgm

首先 ubuntu 需要安装 LaTex:

```shell
sudo apt-get install texlive-full
```

```latex
\documentclass[paper=a5,fontsize=15px]{scrbook}
\usepackage[pdftex,active,tightpage]{preview}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{tikz}
\begin{document}
\begin{preview}
\begin{tikzpicture}[inner sep=0pt, outer sep=0pt]
\node at (0, 0) {\begin{math}\sin^2{\theta} + \cos^2{\theta} = 1\end{math}}; % <--Put your tex-code here
\end{tikzpicture}
\end{preview}
\end{document}
```

```shell
➜  tmp time latex test.tex
...
latex test.tex  0.20s user 0.03s system 87% cpu 0.264 total
```
```shell
time dvisvgm --no-fonts test.dvi test.svg
...
dvisvgm --no-fonts test.dvi test.svg  0.19s user 0.01s system 93% cpu 0.211 total
```
这种方法最长也就 ~0.2s，比上面两种短的多得多。

* pdflatex 和 inkscape

未测试，和上一种方法时间上差不多，但是看生成的 pdf 比较好看，没有导出 svg。

* svg to png

[CairoSVG](https://cairosvg.org/) 将 svg 转 png

ubuntu:
```
sudo apt-get install libcairo2-dev
sudo apt-get install libffi-dev
pip3 install cairosvg
```

**综上，其实最后两种方法速度最快且效果很好**

```shell
ssh://ubuntu@hzzone.io:22/home/ubuntu/miniconda3/bin/python3 -u /home/ubuntu/online-latex-mathmatical-fomula/benchmark.py
  1%|▎                                    | 695/68883 [02:45<4:30:15,  4.21it/s]
```

### 实现

我最后还是实现了一个类似于 codecogs 的网站，因为不擅长前端，做的比较粗糙，具体原理和功能见 [online-latex-mathmatical-fomula](https://github.com/Hzzone/online-latex-mathmatical-fomula)

在线 [demo](https://hzzone.io/api/latex)

![](https://raw.githubusercontent.com/Hzzone/online-latex-mathmatical-fomula/master/demo.gif)