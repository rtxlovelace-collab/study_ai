'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 12_check_langurage.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.JAVA)
print(separators)