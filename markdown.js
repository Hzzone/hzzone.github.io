// node.js, "classic" way:
var MarkdownIt = require('markdown-it');
var fs = require("fs");
var hljs = require('highlight.js');

// 同步读取
// var data = fs.readFileSync(process.argv[2]).toString();
// var data = new Buffer(process.argv[2], 'base64').toString();
var data = Buffer.from(process.argv[2], 'base64').toString('utf-8');
md = new MarkdownIt({
    html: true,
    // breaks: true,
    linkify: true,
    typographer: true,
    highlight: function (str, lang) {
    if (lang && hljs.getLanguage(lang) && hljs.getLanguage(lang)!=hljs.getLanguage('md')) {
      try {
        return '<pre class="hljs"><code>' +
               hljs.highlight(lang, str, true).value +
               '</code></pre>';
      } catch (__) {}
    }

    // return '<pre class="hljs"><code>' + md.utils.escapeHtml(str) + '</code></pre>';
    return '<pre class="hljs"><code>' + hljs.highlightAuto(str).value + '</code></pre>';
  }
});

// 排除有中文的公式
function has_chinese(text) {
    return /.\*[/u4e00\-/u9fa5]\+.\*$/g.test(text);
}

function replace(content, regx) {
    content = content.replace(regx,
        function(expression) {
            var max_formula_length = 1000;
            if (expression.length > max_formula_length || has_chinese(expression)) return expression;
            expression = expression.replace(/\\/gm, '\\\\')
            expression = expression.replace(/_/g, '\_');
            return expression;
        });
    return content;
}

// https://meta.stackexchange.com/questions/34383/how-do-i-escape-backslashes-in-markdown
// 反斜杠转义
data = replace(data, /\$\$([\s\S]*?)\$\$/gm);
data = replace(data, /\$([\s\S]*?)\$/gm);

var result = md.render(data);
console.log(result);
