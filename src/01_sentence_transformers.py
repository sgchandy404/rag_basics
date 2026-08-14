from pathlib import Path
from dotenv import load_dotenv
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

class GlobalVars:

    TRANSFORMER_MODEL:str = ''
    DEFAULT_CHUNK_SIZE:int = 0
    DEFAULT_CHUNK_OVERLAP: int = 0

class ChunkData(BaseModel):

    id:int
    text:str
    origin:str

class EmbeddingData(BaseModel):
    id:int
    text:str
    embedding:list[float]
    origin:str

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

    stored_embeddings = embed_text(chunked_text)


def embed_text(chunked_text:list[ChunkData])->InMemoryDB:


    db = InMemoryDB()

    transformer_model = SentenceTransformer(GlobalVars.TRANSFORMER_MODEL)

    texts = [chunk.text for chunk in chunked_text]
    embeddings = transformer_model.encode(texts)

    assert len(embeddings) == len(chunked_text)

    for embedding, chunk in zip(embeddings,chunked_text):
        chunk_id = chunk.id
        text = chunk.text
        origin = chunk.origin

        db.add(
            EmbeddingData(
                id = chunk_id,
                text = text,
                embedding=embedding.tolist(),
                origin=origin
            )
        )

    return db


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
    GlobalVars.DEFAULT_CHUNK_SIZE = int(os.environ.get('DEFAULT_CHUNK_SIZE',400))
    GlobalVars.DEFAULT_CHUNK_OVERLAP = int(os.environ.get('DEFAULT_CHUNK_OVERLAP',60))

    if GlobalVars.DEFAULT_CHUNK_OVERLAP <= 0 or GlobalVars.DEFAULT_CHUNK_SIZE <= 0:
        raise KeyError('Invalid Values passed for DEFAULT_CHUNK_OVERLAP/DEFAULT_CHUNK_SIZE')

    if not GlobalVars.TRANSFORMER_MODEL:
        raise KeyError('Default Transformer Model Not Found')


if __name__ == "__main__":
    main()