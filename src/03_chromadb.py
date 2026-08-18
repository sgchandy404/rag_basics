import os
import chromadb
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

class GlobalVars:
    DEFAULT_CHUNK_SIZE:int=60
    DEFAULT_CHUNK_OVERLAP:int=15
    TRANSFORMER_MODEL:str = ''
    DEFAULT_VECTOR_DB_PATH:Path
    TOP_K:int = 3

class ChunkData(BaseModel):

    id:int
    text:str
    origin:str

class EmbeddedData(BaseModel):

    id:int
    text:str
    embedding:list[float]
    origin:str

class ResultData(BaseModel):

    embedding_id: int
    text: str
    distance: float
    origin:str


def main():

    try:

        load_env_vars('.env.local')

        doc_name = 'data\\Article.txt'
        
        text_data = read_file(doc_name)

        chunked_text = prepare_chunks(text_data,doc_name)

        transformer_model = initialize_transformer()

        chroma_client = initialize_chroma()

        collection_name = 'research_papers'

        research_collection = get_collection(chroma_client,collection_name)

        if research_collection.count() == 0:
            print('Preparing and Storing Embeddings')
            embedded_text = prepare_embeddings(chunked_text,transformer_model)

            research_collection = populate_chroma(chroma_client,embedded_text,collection_name)

        while True:
            user_query = fetch_user_query()

            embedded_query = embed_user_query(user_query,transformer_model)

            retreived_data = retreive_relevant_data(embedded_query,research_collection)

            display_results(retreived_data)
    except Exception as e:
        print(f'Caught : {e}')
        exit(1)


def display_results(retreived_data:list[ResultData])->None:

    for result in retreived_data:
        print(f'Embedding ID : {result.embedding_id}')
        print(f'Text : {result.text}')
        print(f'Distance : {result.distance}')
        print(f'Origin : {result.origin}')


def get_collection(chroma_client:object,collection_name:str)->object:

    collection = chroma_client.get_or_create_collection(
        name = collection_name
    )
    return collection

def retreive_relevant_data(embedded_query:list[float],collection:object)->list[ResultData]:

    retreived_data = []

    results = collection.query(
        query_embeddings = [embedded_query],
        n_results = GlobalVars.TOP_K
    )

    ids = results['ids'][0]
    documents = results['documents'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    for id,document,distance,metadata in zip(ids,documents,distances,metadatas):
        retreived_data.append(
            ResultData(
                embedding_id=int(id),
                text=document,
                distance=distance,
                origin=metadata['origin']
            )
        )

    return retreived_data

def embed_user_query(query:str,transformer_model)->list[float]:

    embedded_query = transformer_model.encode(query).tolist()

    return embedded_query


def fetch_user_query()->str:

    query = input('Query : ')

    if not query:
        raise RuntimeError('Empty Query Passed')

    if query.lower() in ('exit','quit','bye','hasta la vista'):
        exit(1)

    return query    

def prepare_embeddings(chunk_data,transformer_model) -> list[EmbeddedData]:

    embedded_data = []
    text_list = [chunk.text for chunk in chunk_data]
    encoded_text = transformer_model.encode(text_list)

    assert len(chunk_data) == len(encoded_text)
    for embedding,chunk in zip(encoded_text,chunk_data):
        embedded_data.append(
            EmbeddedData(
                id=chunk.id,
                text=chunk.text,
                embedding=embedding.tolist(),
                origin=chunk.origin
            )
        )
    return embedded_data


def prepare_chunks(text:str,doc_name:str)->list[ChunkData]:

    chunk_data = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = GlobalVars.DEFAULT_CHUNK_SIZE,
        chunk_overlap = GlobalVars.DEFAULT_CHUNK_OVERLAP
    )

    split_data = text_splitter.split_text(text)

    for index,chunk in enumerate(split_data):
        chunk_data.append(
            ChunkData(
                id = index,
                text=chunk,
                origin=doc_name
            )
        )
    return chunk_data


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

def populate_chroma(chroma_client:object,embedded_data:list[EmbeddedData],collection_name:str)->object:

    collection = chroma_client.get_or_create_collection(
        name = collection_name
    )

    collection.add(
        ids = [str(data.id) for data in embedded_data],
        documents = [data.text for data in embedded_data],
        embeddings = [data.embedding for data in embedded_data],
        metadatas = [
            {'origin':data.origin} for data in embedded_data
        ]
    )
    return collection

def initialize_transformer()->object:
    return SentenceTransformer(GlobalVars.TRANSFORMER_MODEL)

def initialize_chroma()->object:
    return chromadb.PersistentClient(path=GlobalVars.DEFAULT_VECTOR_DB_PATH)


def load_env_vars(file_name:str)->None:

    load_dotenv(file_name)

    GlobalVars.DEFAULT_CHUNK_SIZE = int(os.environ.get('DEFAULT_CHUNK_SIZE',0))
    GlobalVars.DEFAULT_CHUNK_OVERLAP = int(os.environ.get('DEFAULT_CHUNK_OVERLAP',0))
    GlobalVars.TRANSFORMER_MODEL = os.environ.get('DEFAULT_SENTENCE_TRANSFORMER_MODEL','')
    GlobalVars.DEFAULT_VECTOR_DB_PATH = Path(os.environ.get('DEFAULT_VECTOR_DB_PATH','models'))
    GlobalVars.TOP_K = int(os.environ.get('TOP_K',3))

    if not GlobalVars.TRANSFORMER_MODEL:
        raise KeyError('Transformer Model Not Defined')

    if GlobalVars.DEFAULT_CHUNK_OVERLAP <=0 or GlobalVars.DEFAULT_CHUNK_SIZE <=0:
        raise KeyError('CHUNK SIZE NOT DEFINED PROPERLY')

    if GlobalVars.TOP_K<=0:
        raise KeyError('Invalid Value Passed for Top K')

    if not GlobalVars.DEFAULT_VECTOR_DB_PATH.exists():
        try:
            os.mkdir(GlobalVars.DEFAULT_VECTOR_DB_PATH,0o777)
        except PermissionError:
            raise PermissionError('Insuficcient Permissions to create required folder')

    

if __name__ == "__main__":
    main()