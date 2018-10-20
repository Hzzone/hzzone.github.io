## GitHub 支持 LaTex 的几种方案
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

## 如何选择

目前我认为最好的方法是通过 [codecogs](http://www.codecogs.com/latex/eqneditor.php) 下载 gif 图片，然后上传到图床或者本地的仓库，可以解决服务不稳定的问题，至少可以保证自己的服务器或域名是可以访问的。

另外一种方法，自己搭建一个服务器，随便采用什么框架，然后通过上面介绍的 <a href="#LaTeX-生成-svg">LaTeX 生成 svg</a> 提供一个类似于 codecogs 的服务，返回 svg 图片。而且目前也有类似功能的，例如 [texoid](https://github.com/DMOJ/texoid)。

实现起来很简单，但我嫌麻烦，可以考虑上一种方法，我认为是在 GitHub 不支持 LaTeX 时的最优解。

## 实现

我最后还是实现了一个类似于 codecogs 的网站，因为不擅长前端，做的比较粗糙，具体原理和功能见 [online-latex-mathmatical-fomula](https://github.com/Hzzone/online-latex-mathmatical-fomula)

在线 [demo](https://hzzone.io/api/latex)

![](https://raw.githubusercontent.com/Hzzone/online-latex-mathmatical-fomula/master/demo.gif)