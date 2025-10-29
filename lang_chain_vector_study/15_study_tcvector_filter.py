'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 15_study_tcvector_filter.py
* @Time: 2025/9/4
* @All Rights Reserve By Brtc
'''

import os
import dotenv
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import META_FIELD_TYPE_STRING, MetaField, ConnectionParams, \
    META_FIELD_TYPE_UINT64

dotenv.load_dotenv()
db = TencentVectorDB(
    embedding=None,
    connection_params=ConnectionParams(
        url = os.environ.get("TC_VECTOR_DB_URL"),
        username = os.environ.get("TC_VECTOR_DB_USERNAME"),
        key = os.environ.get("TC_VECTOR_DB_KEY"),
        timeout=int(os.environ.get("TC_VECTOR_DB_TIMEOUT"))
    ),
    database_name=os.environ.get("TC_VECTOR_DB_DATABASE"),
    collection_name="inner_embd_index",
    meta_fields=[
        MetaField(name="text", data_type = META_FIELD_TYPE_STRING, index=False,description="this the meta text"),
        MetaField(name="page", data_type=META_FIELD_TYPE_UINT64, index=True, description="this the meta page")
    ]
)
# 检索
ids  = db.similarity_search_with_score("我养了一只猫名字叫笨笨", expr="page>5")
for one in ids:
    print(one)