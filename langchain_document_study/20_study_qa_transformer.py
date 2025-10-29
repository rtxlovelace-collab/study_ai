'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 20_study_qa_transformer.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_community.document_transformers import DoctranQATransformer
from langchain_core.documents import Document

dotenv.load_dotenv()

# 1.构建文档列表
page_content = """..."""
documents = [Document(page_content=page_content)]

# 2.构建问答转换器并转换
qa_transformer = DoctranQATransformer(openai_api_model="gpt-4o-mini")
transformer_documents = qa_transformer.transform_documents(documents)

# 3.输出内容
for qa in transformer_documents[0].metadata.get("questions_and_answers"):
    print(qa)