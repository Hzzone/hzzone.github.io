import os
import time
import os.path as osp
from jinja2 import Template
from config import cfg
from tqdm import tqdm
from easydict import EasyDict as edict
import re
import requests
import base64
import urllib
import pysnooper
import nbformat
from nbconvert import MarkdownExporter
import sys


cfg.github.headers = {'Authorization': 'token {}'.format(sys.argv[1])}

# 删除生成的文件，除 .git 以外
for f in os.listdir(cfg.local.generate):
    if f == '.git':
        continue
    os.popen('rm -rf {}'.format(osp.join(cfg.local.generate, f.replace(' ', '\ '))))
# 拷贝静态文件
os.popen('cp -r {} {}'.format(osp.join('template', 'static'),
                              cfg.local.generate))

# 加载模板
base_template_path = 'template/layout.html'
post_template_path = 'template/post.html'
index_template_path = 'template/index.html'
base_template = Template(open(base_template_path).read())
post_template = Template(open(post_template_path).read())
index_template = Template(open(index_template_path).read())

# 将 notebook 转换成 md, 并将比如 matplotlib 的图片转换成 base64 编码
def convert_ipynb2md(content):
    jake_notebook = nbformat.reads(content, as_version=4)
    exporter = MarkdownExporter()
    body, resource = exporter.from_notebook_node(jake_notebook)
    for name, value in resource['outputs'].items():
        postfix = osp.splitext(name)[1]
        encoded_data = base64.b64encode(value).decode()
        # base64 编码，后缀 .png
        body = body.replace(name, 'data:image/{};base64,{}'.format(postfix[1:], encoded_data))
    return body

convert_functions = {
    '.md': lambda x: x,
    '.ipynb': convert_ipynb2md
}

def read_from_github():
    # 获取博客分支的 commit 内容
    branch_sha = requests.get(cfg.github.source_branch_url, headers=cfg.github.headers).json()['commit']['sha']
    tree_sha = requests.get('{}/{}'.format(cfg.github.commits_url, branch_sha), headers=cfg.github.headers).json()['tree']['sha']

    tmp_tags = requests.get('{}/{}'.format(cfg.github.tree_url, tree_sha), headers=cfg.github.headers).json()['tree']
    tmp_tags = sorted(tmp_tags, key=lambda x: x['path'])
    tmp_tags = filter(lambda x: x['type'] == 'tree' and x['path'][0].isdigit(), tmp_tags)
    data = edict()
    data.tags = []
    for tmp_tag in tqdm(tmp_tags):
        tag = edict()
        tag.org_name = tmp_tag['path']
        tag.name = re.sub('\d{1}. ', '', tmp_tag['path'])
        tag.posts = []
        tag_posts = requests.get(tmp_tag['url'], headers=cfg.github.headers).json()['tree']
        tag_posts = filter(lambda x: x['type'] == 'blob', tag_posts)
        tag_posts = filter(lambda x: osp.splitext(x['path'])[1] in cfg.conversion.postfix, tag_posts)
        for tmp_post in tqdm(tag_posts):
            post = edict()
            post.name, post.postfix = osp.splitext(tmp_post['path'])
            content = requests.get(tmp_post['url'], headers=cfg.github.headers).json()['content']

            content = convert_functions[post.postfix](content)

            post.content = base64.b64decode(content).decode('utf-8')
            payload = {
                'sha': branch_sha,
                'path': '{}/{}'.format(tmp_tag['path'], tmp_post['path'], 'utf-8')
            }
            params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
            commits = requests.get(cfg.github.file_commit_url, params=params, headers=cfg.github.headers).json()
            commits_time = [x['commit']['committer']['date'] for x in commits]
            commits_time = [time.strptime(x, '%Y-%m-%dT%H:%M:%SZ') for x in commits_time]
            post.year = commits_time[-1].tm_year
            post.formated_ctime = time.strftime("%Y-%m-%d %H:%M:%S", commits_time[-1])
            if len(commits) > 1:
                post.formated_mtime = time.strftime("%Y-%m-%d %H:%M:%S", commits_time[0])
            else:
                post.formated_mtime = None
            post.tag = tag.name
            post.org_tag = tag.org_name
            tag.posts.append(post)
        data.tags.append(tag)
    return data

def read_from_local():
    # 获取博客分支的 commit 内容
    branch_sha = requests.get(cfg.github.source_branch_url, headers=cfg.github.headers).json()['commit']['sha']

    tmp_tags = list(filter(lambda x: x[0].isdigit() and os.path.isdir(osp.join(cfg.local.source, x)), sorted(os.listdir(cfg.local.source))))
    data = edict()
    data.tags = []
    for tmp_tag in tqdm(tmp_tags):
        tag = edict()
        tag.org_name = tmp_tag
        tag.name = re.sub('\d{1}. ', '', tmp_tag)
        tag.posts = []
        tag_source = osp.join(cfg.local.source, tmp_tag)
        tag_posts = os.listdir(tag_source)
        tag_posts = list(filter(lambda x: not os.path.isdir(osp.join(tag_source, x)), tag_posts))
        tag_posts = list(filter(lambda x: osp.splitext(x)[1] in cfg.conversion.postfix, tag_posts))
        for tmp_post in tqdm(tag_posts):
            post = edict()
            post.name, post.postfix = osp.splitext(tmp_post)
            post_path = osp.join(tag_source, tmp_post)
            with open(post_path) as f:
                post.content = convert_functions[post.postfix](f.read())
            try:
                payload = {
                    'sha': branch_sha,
                    'path': '{}/{}'.format(tmp_tag, tmp_post, 'utf-8')
                }
                params = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
                commits = requests.get(cfg.github.file_commit_url, params=params, headers=cfg.github.headers).json()
                commits_time = [x['commit']['committer']['date'] for x in commits]
                commits_time = [time.strptime(x, '%Y-%m-%dT%H:%M:%SZ') for x in commits_time]
                post.year = commits_time[-1].tm_year
                post.formated_ctime = time.strftime("%Y-%m-%d %H:%M:%S", commits_time[-1])
                if len(commits) > 1:
                    post.formated_mtime = time.strftime("%Y-%m-%d %H:%M:%S", commits_time[0])
                else:
                    post.formated_mtime = None
            except:
                mtime = time.localtime(os.path.getmtime(post_path))
                post.year = mtime.tm_year
                post.formated_ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getctime(post_path)))
                post.formated_mtime = time.strftime("%Y-%m-%d %H:%M:%S", mtime)
            post.tag = tag.name
            post.org_tag = tag.org_name
            tag.posts.append(post)
        data.tags.append(tag)
    return data

data = read_from_local()
# data = read_from_github()

# 按时间排序
for tag in data.tags:
    tag.posts = sorted(tag.posts, key=lambda x: x.formated_ctime)

# index
with open(osp.join(cfg.local.source, 'README.md')) as f:
    content = f.read()
README = os.popen("node markdown.js {}".format(base64.b64encode(content.encode()).decode())).read()
index_content = index_template.render(
    tags=data.tags,
    README=README)
main_content = base_template.render(main_content=index_content, avatar=cfg.self.avatar,
                                    short_introduction=cfg.self.short_introduction,
                                    name=cfg.self.name,
                                    title='主页')

with open(osp.join(cfg.local.generate, 'index.html'), 'w') as f:
    f.write(main_content)

# 生成所有博客界面
for tag in tqdm(data.tags):
    for post in tag.posts:
        content = os.popen("node markdown.js {}".format(base64.b64encode(post.content.encode()).decode())).read()
        post_content = post_template.render(post_content=content,
                                            post=post,
                                            cfg=cfg)
        main_content = base_template.render(main_content=post_content, avatar=cfg.self.avatar,
                                            short_introduction=cfg.self.short_introduction,
                                            name=cfg.self.name,
                                            title=post.name)
        tag_dir = osp.join(cfg.local.generate, tag.org_name)
        if not os.path.exists(tag_dir):
            os.mkdir(tag_dir)
        with open(osp.join(tag_dir, post.name + '.html'), 'w') as f:
            f.write(main_content)

