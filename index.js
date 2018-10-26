var nunjucks =require('nunjucks');
var marked = require('marked');
var fs = require("fs");
var path = require("path");
var util = require('util');
var child_process = require('child_process');
var entities = require("entities");

var resource = 'resource';

var root = path.resolve();
var resource_dir = path.join(root, resource);
var build_dir = path.join(root, 'build');

nunjucks.configure('templates', { autoescape: true });

function copyDir(src, dist) {
    child_process.spawn('cp', ['-r', src, dist]);
}

// 排除有中文的公式
function has_chinese(text) {
    return /.\*[/u4e00\-/u9fa5]\+.\*$/g.test(text);
}

// Set options
// `highlight` example uses `highlight.js`
const renderer = new marked.Renderer();
renderer.heading = (text, level, raw) => {
    const headder_id = raw.toLowerCase().replace(/[^a-zA-Z0-9\u4e00-\u9fa5]+/g, '-');
    // console.log(headder_id);
    var result = util.format('\<h%s id=%s\>%s\</h%s\>\n', level, headder_id, text, level);
    return result;
};
marked.setOptions({
    renderer: renderer,
    pedantic: false,
    gfm: true,
    tables: true,
    breaks: false,
    sanitize: false,
    smartLists: true,
    smartypants: false,
    xhtml: false,
    langPrefix: ''
});


function removeDir(dir) {
    var files = fs.readdirSync(dir);
    for(var i=0;i<files.length;i++){
        let newPath = path.join(dir,files[i]);
        let stat = fs.statSync(newPath);
        if(stat.isDirectory()){
            //如果是文件夹就递归下去
            removeDir(newPath);
        }else {
            //删除文件
            fs.unlinkSync(newPath);
        }
    }
    fs.rmdirSync(dir);//如果文件夹是空的，就将自己删除掉
}

if(fs.existsSync(build_dir)) {
    removeDir(build_dir);
}
fs.mkdirSync(build_dir);

copyDir(path.join(root, 'templates', 'static'), build_dir);

function replace(content, regx) {
    content = content.replace(regx,
        function(expression) {
            var max_formula_length = 1000;
            if (expression.length > max_formula_length || has_chinese(expression)) return expression;
            expression = expression.replace(/\\\\/g, '\\\\\\\\');
            expression = expression.replace(/\\ /g, '\\\\ ');
            expression = expression.replace(/\\%/g, '\\\\%');
            expression = expression.replace(/\\{/g, '\\\\{');
            expression = expression.replace(/\\}/g, '\\\\}');
            expression = expression.replace(/\\#/g, '\\\\#');
            expression = expression.replace(/\\~/g, '\\\\~');
            expression = expression.replace(/\\_/g, '\\\\_');
            expression = expression.replace(/\\&/g, '\\\\&');
            expression = expression.replace(/\\\$/g, '\\\\$');
            expression = expression.replace(/\\\^/g, '\\\\^');
            expression = expression.replace(/\\\|/g, '\\\\|');
            expression = expression.replace(/\_/g, '\\_');
            return expression;
        });
    return content;
}

function generate(raw_data, title) {
    var data = replace(raw_data, /\$\$([\s\S]*?)\$\$/gm);
    data = replace(data, /\$([\s\S]*?)\$/gm);
    // var data = raw_data;
    data = marked(data);
    var html_data = nunjucks.render('markdown.html', {title: title, markdownbody: data});
    html_data = entities.decodeHTML(html_data);
    // mjpage(html_data, {format: ["TeX"]}, {html: true, css: true}, function (output) {
    //     fs.writeFileSync(build_file_path, output, {encoding: 'utf8'});
    // });
    return html_data
}

// console.log(generate('$$\\mathcal{L}_C=\\frac{1}{2} \\sum_{i=1}^{m} \\| x_i-c_{y_i} \\|_2^2$$', 'test'));

var path_dict = {};

var dirs = fs.readdirSync(resource_dir);
dirs.forEach(function (dir_name) {
    var dir_path = path.join(resource_dir, dir_name);
    let stat = fs.statSync(dir_path);
    console.log(dir_name);
    if (stat.isFile() || dir_name==".git") return;
    var build_class_dir = path.join(build_dir, dir_name);
    fs.mkdirSync(build_class_dir);
    path_dict[dir_name] = new Array();
    var files = fs.readdirSync(dir_path);
    files.forEach(function (filename) {
        var file_postfix = path.extname(filename);
        var file_path = path.join(dir_path, filename);
        if (file_postfix == '.md') {
            var title = filename.slice(0, filename.length - 3);
            path_dict[dir_name].push(title);
            var build_file_path = path.join(build_class_dir, title + '.html');
            var raw_data = fs.readFileSync(file_path, {encoding: 'utf8'});

            var html_data = generate(raw_data);

            fs.writeFileSync(build_file_path, html_data, { encoding: 'utf8' });
        }
    });
});

var readme_data = fs.readFileSync(path.join(resource_dir, 'README.md'), { encoding: 'utf8' });
var data = marked(readme_data);
var title = "黄智忠的个人笔记";
for (var class_name in path_dict) {
    if (path_dict[class_name].length==0) continue;
    data += util.format("<h2>%s</h2>\n<ul>", class_name);
    path_dict[class_name].forEach(function (article_title) {
        data += util.format('<li><a class="link" href="%s.html">%s</a></li>\n', util.format("%s/%s", class_name, article_title), article_title);
    });
    data += "</ul>\n";
}
var html_data = entities.decodeHTML(nunjucks.render('markdown.html', {title: title, markdownbody: data}));
fs.writeFileSync(path.join(build_dir, "index.html"), html_data, {encoding: 'utf8'});



