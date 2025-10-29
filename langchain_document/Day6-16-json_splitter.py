import requests
from langchain_text_splitters import RecursiveJsonSplitter

#获取并加载json文件
url = 'https://api.smith.langchain.com/openapi.json'
json_data = requests.get(url).json()
#归递json文件分割
text_splitter = RecursiveJsonSplitter(max_chunk_size=300)
#分割json数据并创建文档
json_chunks = text_splitter.split_json(json_data=json_data)
chunks = text_splitter.create_documents(json_chunks)
for one in chunks:
    print(one)
print(len(chunks))
