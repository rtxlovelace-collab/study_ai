

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
    ("h1", "标题1级"),
    ("h2", "标题2级"),
    ("h3", "标题3级")
]
from langchain_text_splitters import HTMLHeaderTextSplitter
html_splitter = HTMLHeaderTextSplitter(header_to_split_on)
chunks = html_splitter.split_text(html_string)
print(chunks)
for one in chunks:
    print(one)