很少有图床支持 https，而阿里云，腾讯云的对象存储 https太麻烦，刚好我自己有服务器，并且用 Let’s Encrypt 上了免费的 https。放着也是放着，就用服务器来的那个图床用。

所以如何快速上传图片到服务器并且导出链接：

```shell
alias up='upload() { scp "$1" ubuntu@hzzone.io:/usr/share/nginx/html/images; echo "https://hzzone.io/images/$(basename "$1")" }; upload'
```

在 `zshrc` 中定义一个 alias 就很方便了。

效果如下：

```
 ~ up /Users/hzzone/Desktop/test.png
test.png                                      100%  232KB 921.1KB/s   00:00
https://hzzone.io/images/test.png
```

可以用这个非常方便的将文件上传到服务器，提供下载链接，不局限于图片。