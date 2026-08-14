from sentence_transformers import SentenceTransformer


transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = transformer_model.tokenizer

text = 'RAG systems retrieve relevant documents.'

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)
print(token_ids)

embeddings = transformer_model.encode(text)
print(embeddings)