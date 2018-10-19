### 参考
* [Marked.js Documentation](https://marked.js.org/#/USING_ADVANCED.md#highlight)
* [highlight.js Library API](https://highlightjs.readthedocs.io/en/latest/api.html)
* [highlight.js cdnjs](https://cdnjs.com/libraries/highlight.js)

```html
<!DOCTYPE html>
<html>
<head>
    <title></title>
</head>
<body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/0.5.1/marked.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.12.0/styles/xcode.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.12.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.12.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
<div id="code">
\`\`\`C++
for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
    if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
        loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
        if (legacy_version) {
            loss += std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0));
        } else {
            Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq_.cpu_data()[i]),
              Dtype(0.0));
            loss += dist*dist;
        }
    }
}
\`\`\`
</div>
<div id="content" >

</div>

<script type="text/javascript">
$("#content").html(marked($("#code").html(), {gfm: true, highlight: function(code, lang, callback) {
    return hljs.highlight(lang, code).value;
}}));
</script>
</body>
</html>
```

代码高亮风格只需要修改 css 文件，传入 `marked` 的各项参数（markdown 代买需删除反斜杠）。
