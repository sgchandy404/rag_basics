from pathlib import Path
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class GlobalVars:

    TRANSFORMER_MODEL:str = ''
    DEFAULT_CHUNK_SIZE:int = 0
    DEFAULT_CHUNK_OVERLAP: int = 0
    SIMILARITY_THRESHOLD: float = 0.5
    TOP_K:int = 3
    MAX_DISTANCE:float = 0.5

class ChunkData(BaseModel):

    id:int
    text:str
    origin:str

class EmbeddingData(BaseModel):
    id:int
    text:str
    embedding:list[float]
    origin:str

class ResultData(BaseModel):
    id:int
    query:str
    score: float
    chunk: str
    embedding_id: int


class InMemoryDB:
    def __init__(self):
        self.records = []

    def add(self,record:EmbeddingData):
        self.records.append(record)

def main()->None:

    load_env_vars('.env.local')

    doc_name = 'data\\HR_Policy.txt'

    text_data = read_file(doc_name)

    chunked_text = prepare_chunks(text_data,doc_name)

    transformer_model = SentenceTransformer(GlobalVars.TRANSFORMER_MODEL)

    faiss_index,embedded_data = embed_text(chunked_text,transformer_model)

    while True:
        query = fetch_query()
        results = search_query(query,faiss_index,embedded_data,transformer_model)
        display_results(results)


def display_results(results:list[ResultData])->None:

    if not results:
        print('No Relevant Data Found !')
        return None

    # top_k_sorted_results = sorted(results,
    #                               key = lambda x: x.score,
    #                               reverse=True)[:GlobalVars.TOP_K]

    print(f'Top {GlobalVars.TOP_K} chunks retrieved with their scores : ')
    for result in results:
        print('*'*60)
        print(f'Chunk : {result.chunk}')
        print(f'Euc Distance : {result.score}')

def search_query(query:str,faiss_index:faiss.Index,embeddings:list[EmbeddingData],transformer_model:SentenceTransformer)->list[ResultData]:

    result_list = []

    embedded_query = transformer_model.encode(query)

    embedded_query = np.array([embedded_query],dtype=np.float32)

    distance,indices = faiss_index.search(embedded_query,k=GlobalVars.TOP_K)

    for d,i in zip(distance[0],indices[0]):
        if d <= GlobalVars.MAX_DISTANCE:
            result_list.append(
                ResultData(
                    id = len(result_list),
                    query=query,
                    score=d,
                    embedding_id=embeddings[i].id,
                    chunk=embeddings[i].text
                )
            )

    return result_list

def cosine_similarity(query_embedding,chunk_embedding):

    return(np.dot(query_embedding,chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)))


def fetch_query()->str:

    query = ''
    query = input('Please enter query : ')

    if not query:
        raise RuntimeError('Empty Query Passed')

    if query.lower() == 'exit':
        exit(1)

    return query



def embed_text(chunked_text:list[ChunkData],transformer_model:SentenceTransformer)->tuple[faiss.Index,list[EmbeddingData]]:

    embedded_data = []

    texts = [chunk.text for chunk in chunked_text]
    embeddings = transformer_model.encode(texts)
    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)


    assert len(embeddings) == len(chunked_text)

    for embedding, chunk in zip(embeddings,chunked_text):
        chunk_id = chunk.id
        text = chunk.text
        origin = chunk.origin

        embedded_data.append(
            EmbeddingData(
                id = chunk_id,
                text = text,
                embedding=embedding.tolist(),
                origin=origin
            )
        )

    return faiss_index,embedded_data


def prepare_chunks(text:str,doc_name:str) -> list[ChunkData]:

    chunked_data = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = GlobalVars.DEFAULT_CHUNK_SIZE,
        chunk_overlap = GlobalVars.DEFAULT_CHUNK_OVERLAP
    )

    split_text = splitter.split_text(text)

    for index,chunk in enumerate(split_text):
        chunked_data.append(
            ChunkData(
                id=index,
                text=chunk,
                origin= doc_name
            )
        )

    return chunked_data

def read_file(input_file:str)->str:

    text_data = ''

    file_path = Path(input_file)
    if not file_path.exists():
        raise FileNotFoundError(f'Invalid Input Document Passed : {file_path.name}')

    print(f'Processing : {file_path.name}')

    with open(file_path,'r',encoding='utf8') as reader:
        text_data = reader.read()

    if not text_data:
        raise RuntimeError('Empty File Passed !!')
    return text_data

def load_env_vars(env_file:str)->None:

    env_path = Path(env_file)
    if not env_path.exists():
        raise FileNotFoundError('.env file not found in root dir')

    load_dotenv(env_file)
    GlobalVars.TRANSFORMER_MODEL = os.environ.get('DEFAULT_SENTENCE_TRANSFORMER_MODEL','')
    GlobalVars.DEFAULT_CHUNK_SIZE = int(os.environ.get('DEFAULT_CHUNK_SIZE',10))
    GlobalVars.DEFAULT_CHUNK_OVERLAP = int(os.environ.get('DEFAULT_CHUNK_OVERLAP',4))
    GlobalVars.SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD',0.5))
    GlobalVars.TOP_K = int(os.environ.get('TOP_K',3))
    GlobalVars.MAX_DISTANCE = float(os.environ.get('MAX_DISTANCE',0.5))

    if GlobalVars.DEFAULT_CHUNK_OVERLAP <= 0 or GlobalVars.DEFAULT_CHUNK_SIZE <= 0:
        raise KeyError('Invalid Values passed for DEFAULT_CHUNK_OVERLAP/DEFAULT_CHUNK_SIZE')

    if GlobalVars.SIMILARITY_THRESHOLD <=0 or GlobalVars.TOP_K <=0:
        raise KeyError('Invalid Values passed for SIMILARITY THRESHOLD/TOP_K')

    if not GlobalVars.TRANSFORMER_MODEL:
        raise KeyError('Default Transformer Model Not Found')


if __name__ == "__main__":
    main()