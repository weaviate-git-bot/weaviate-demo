import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
import weaviate
import json

from dotenv import load_dotenv
import os

load_dotenv()
SAVE_DATA_TO_CSV_FILE_PATH = os.getenv('SAVE_DATA_TO_CSV_FILE_PATH')
CSV_FILE_PATH = os.getenv('CSV_FILE_PATH')
WEAVIATE_HOST = os.getenv('WEAVIATE_HOST')
WEAVIATE_SECRET_KEY = os.getenv('WEAVIATE_SECRET_KEY')

model_name = 'distiluse-base-multilingual-cased'
emb=HuggingFaceEmbeddings(model_name=model_name)

print('Loading data from CSV file...')
indexDataFrame = pd.read_csv(CSV_FILE_PATH)
indexDataFrame.reset_index(drop=True, inplace=True)
df=indexDataFrame

def generate_data_embeddings(df):
    df['embedding'] = df['context'].apply(lambda row: emb.embed_query(row))
    return df

print('Generating embeddings for data...')
df=generate_data_embeddings(df)

print('Saving data with embeddings to CSV file...')
df.to_csv(SAVE_DATA_TO_CSV_FILE_PATH, index=False)

print('Data saved to CSV file!, now uploading data to Weaviate...')
client = weaviate.Client(
    url = WEAVIATE_HOST,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_SECRET_KEY),
)

def weaviate_create_schema():
    schema = {
        "classes": [{
            "class": "Text",
            "description": "Contains the paragraphs of text along with their embeddings",
            "vectorizer": "none",
            "properties": [
                {
                    "name": "context",
                    "dataType": ["context"],
                },
                {
                    "name": "ogr",
                    "dataType":["ogr"],
                },
                {
                     "name": "libelle",
                    "dataType":["libelle"],
                },
                {
                    "name": "libelle_type",
                    "dataType":["libelle_type"]
                },
                 {
                    "name": "root",
                    "dataType":["root"]
                },
                {
                    "name": "root_type",
                    "dataType":["root_type"]
                },

            ]
        }]
    }
    client.schema.create(schema)

print('Creating schema in Weaviate...')

def weaviate_add_data(df):
    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for index, row in df.iterrows():
            context = row['context']
            ogr = row['ogr']
            libelle = row['libelle']
            libelle_type = row['libelle_type']
            ebd = row['embedding']
            batch_data = {
                "context": context,
                "ogr":ogr,
                "libelle":libelle,
                "libelle_type":libelle_type
            }
            batch.add_data_object(data_object=batch_data, class_name="Text", vector=ebd)

    print("Data Added!")

print('Adding data to Weaviate...')
weaviate_add_data(df)

def query(input_text, k):
    input_embedding = emb.embed_query(input_text)
    vec = {"vector": input_embedding}
    result = client \
        .query.get("Text", ["ogr", "context"]) \
        .with_near_vector(vec) \
        .with_limit(k) \
        .do()
    return result

input_text = "arrete"
k_vectors = 1
print('Querying Weaviate...')
result = query(input_text, k_vectors)
print(json.dumps(result, indent=4))
