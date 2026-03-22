import os

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore

db_config = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "database": os.getenv("POSTGRES_VECTOR_DB"),
    "embed_dim": 1536,
    "hnsw_kwargs": {
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 40,
        "hnsw_dist_method": "vector_cosine_ops",
    },
    "create_engine_kwargs": {"pool_size": 5},
}

catalog_store = PGVectorStore.from_params(**db_config, table_name="techmart_catalog")
faq_store = PGVectorStore.from_params(**db_config, table_name="techmart_faq")
troubleshooting_store = PGVectorStore.from_params(**db_config, table_name="techmart_troubleshooting")
other_store = PGVectorStore.from_params(**db_config, table_name="techmart_other")

catalog_index = VectorStoreIndex.from_vector_store(vector_store=catalog_store)
faq_index = VectorStoreIndex.from_vector_store(vector_store=faq_store)
troubleshooting_index = VectorStoreIndex.from_vector_store(vector_store=troubleshooting_store)
other_index = VectorStoreIndex.from_vector_store(vector_store=other_store)
