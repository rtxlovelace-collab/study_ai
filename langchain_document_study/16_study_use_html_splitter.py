'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 16_study_use_html_splitter.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from langchain_text_splitters import HTMLHeaderTextSplitter

html_string = """ 
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>标题1</h1>
        <p>关于标题1的一些介绍文本。</p>
        <div>
            <h2>子标题1</h2>
            <p>关于子标题1的一些介绍文本。</p>
            <h3>子子标题1</h3>
            <p>关于子子标题1的一些文本。</p>
            <h3>子子标题2</h3>
            <p>关于子子标题2的一些文本。</p>
        </div>
        <div>
            <h3>子标题2</h2>
            <p>关于子标题2的一些文本。</p>
        </div>
        <br>
        <p>关于标题1的一些结束文本。</p>
    </div>
</body>
</html>
"""
header_to_split_on = [
    ("h1","一级标题"),
    ("h2","二级标题"),
    ("h3","三级标题")
]
html_splitter = HTMLHeaderTextSplitter(header_to_split_on)
chunk = html_splitter.split_text(html_string)
for one in chunk:
    print(one)

