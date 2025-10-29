from langchain_community.document_loaders.generic import GenericLoader

#带进度条的通用文档加载器，显示当前加载的文件名和文件大小
loader = GenericLoader.from_filesystem(".", glob="*.txt", show_progress=True)
for idx, doc in enumerate(loader.lazy_load()):
    print(f"当前加载第{idx + 1}个文件，文件信息:{doc.metadata},文件正文：{doc.page_content}")