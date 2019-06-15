知乎编辑器支持直接导入 markdown 文本，但是不支持直接渲染使用 LaTeX 写的数学公式，必须使用编辑器输入，对于公式比较多的内容很不方便。现在的知乎编辑器插件也不要想了，不支持公式。

我最开始考虑直接写 js 修改输入文本框内的内容，因为知乎编辑器公式渲染都是通过以下格式:
```
<span class="Formula isEditable" style="background-image: url(&quot;//www.zhihu.com/equation?tex=x_1&quot;);"><img class="Formula-image" data-eeimg="true" src="//www.zhihu.com/equation?tex=x_1" alt="x_1" width="19" height="16"><span class="Formula-placeholder" data-paste-ignore="true"><span data-offset-key="ackhh-1-0"><span data-text="true"> </span></span></span></span>
```

修改成功之后虽然文本编辑器渲染公式成功了，提交修改后依然为原来的内容。

再后来通过观察之后，知乎的其实是通过图片存在的：
```
<img src="https://www.zhihu.com/equation?tex=p" alt="p" eeimg="1">
```
当我在 markdown 文件中直接这样写公式时，再导入编辑器，成功渲染并修改。

所以最后，我通过正则表达式匹配 LaTeX 公式，并修改成以上格式，达到了直接导入 markdown 的功能。因此也不需要一个一个去转换公式。

可以写 python 实现，几行代码而已，但对我来说既然追求简便，那就最简单吧。我稍微写了点界面直接转换 markdown 文件，提交文件后点击 submit，文本不会上传，全部本地完成，而且只有一个 html 文件（PS: 我不太熟悉 js）

在线地址: [zhihu_markdown_latex_math_conversion](https://hzzone-1252747889.cos.ap-guangzhou.myqcloud.com/zhihu_markdown_latex_math_conversion.html)

测试代码:
```markdown
### 测试公式
* 行内公式

哈哈哈哈哈 $x_1+x_2$ 公式你好！

* 行间公式

哈哈哈哈哈

$$x_1+x_2$$

公式你好！
```

导出:

```markdown
### 测试公式
* 行内公式

哈哈哈哈哈 <img src="https://www.zhihu.com/equation?tex=x_1+x_2" alt="x_1+x_2" eeimg="1"> 公式你好！

* 行间公式

哈哈哈哈哈


<img src="https://www.zhihu.com/equation?tex=x_1+x_2" alt="x_1+x_2" eeimg="1">


公式你好！
```
效果图:
![](http://ww3.sinaimg.cn/large/006tNc79gy1g3u6tkltabj315s0qatas.jpg)

（知乎的 markdown 转 html 和 vscode 一样）