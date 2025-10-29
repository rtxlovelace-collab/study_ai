#一次性加载文件夹下所有文件,包括文件夹内的子文件夹
from langchain_community.document_loaders import FileSystemBlobLoader

loader = FileSystemBlobLoader(".",show_progress=True)
for doc in loader.yield_blobs():
    print(doc)