import boto3
import json
import os
from io import BytesIO
from opensearchpy import OpenSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from botocore.exceptions import ClientError
import os

# Environment variables
ES_HOST = os.environ['ES_HOST']
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
SECRET_NAME = os.environ['SECRET_NAME']
INDEX_NAME = os.environ['INDEX_NAME']

def get_secret():
    """
    Retrieves OpenSearch credentials from AWS Secrets Manager.

    Returns:
        A dictionary containing the retrieved username and password, or None if an error occurs.
    """
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=session.region_name
    )
    try:
        secret_value = client.get_secret_value(SecretId=SECRET_NAME)
        return json.loads(secret_value['SecretString'])
    except ClientError as e:
        print(f"Error getting OpenSearch credentials from Secrets Manager: {e}")
        return None
        
def connect_with_opensearch():
    """
    Connects to the OpenSearch cluster using the retrieved credentials.

    Returns:
        An OpenSearch client object, or None if connection fails.
    """
    secret = get_secret()
    os_auth = (secret['username'], secret['password'])
    # os_auth = (ES_USERNAME, ES_PASSWORD)
    os_client = OpenSearch(
        hosts=[ES_HOST],
        http_auth=os_auth,
        use_ssl=True,
        verify_certs=True,
        pool_maxsize=20
    )
    if 'cluster_name' in os_client.info():
        print("Connected to OpenSearch")
        return os_client
    else:
        print("Error connecting to OpenSearch")
        return None


def create_index(index_name):
    """
    Creates an OpenSearch index with specific settings for KNN search.

    This is a one-time activity that should be run only once.

    Args:
        index_name: The name of the index to create.
    """
    try:
        os_client = connect_with_opensearch()
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "vector-index1": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "text": 
                    {
                        "type": "text"
                    },
                    "document-name": 
                    {
                        "type": "text"
                    }
                }
            }
        }
        response = os_client.indices.create(index_name, body=index_body)
        print(f"Index creation status: {response['status']}")
    except Exception as excp:
        print(f"Error creating index: {excp}")


 
def index_data(os_client, index_name, doc, list_of_tokens, document_name):
    """
    Indexes a document and its corresponding vector representation in OpenSearch.

    Args:
        os_client: The OpenSearch client object.
        index_name: The name of the index to index the data in.
        doc: The text document to be indexed.
        list_of_tokens: The vector representation of the document.
    """
    try:
        my_doc = {"text": doc, "vector-index1": list_of_tokens, 'document-name': document_name }
        response = os_client.index(index = index_name, body = my_doc, refresh = True)
    except Exception as excp:
        print("Exception in index data function")
        print(excp)



def split_text(letter):
    """
    Splits the given text into smaller chunks for processing.
    
    Args:
        letter: The text to be split.
    
    Returns:
        A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=200
    )
    docs = text_splitter.split_text(letter)
    return docs



def read_pdf(s3_bucket, s3_key):
    """
    Reads the content of a PDF file stored in an S3 bucket and returns the extracted text.

    Args:
        s3_bucket: The name of the S3 bucket where the PDF file is stored.
        s3_key: The key (path) of the PDF file within the S3 bucket.

    Returns:
        The extracted text content from all pages of the PDF.
    """
    s3 = boto3.client("s3")
    pdf_object = s3.get_object(Bucket=s3_bucket, Key=s3_key)
    pdf_bytes = pdf_object["Body"].read()
    pdf_file = BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        # Remove newline characters and append the text
        text += page.extract_text().replace('\n', ' ')

    return text


def generate_embeddings(doc):
    """
    Sends the provided text document to a SageMaker endpoint and retrieves the embeddings.

    Args:
        doc: The text document to generate embeddings for.

    Returns:
        The generated vector representation of the document.
    """
    client = boto3.client('runtime.sagemaker')
    payload = {"text_inputs": doc}
    encoded_json = json.dumps(payload).encode('utf-8')
    query_response = client.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='application/json', Body=encoded_json)
    model_predictions = json.loads(query_response['Body'].read())
    embeddings = model_predictions['embedding']
    return embeddings


def lambda_handler(event, context):
    os_client = connect_with_opensearch()
    # create_index(INDEX_NAME)
    s3_bucket = event['Records'][0]['s3']['bucket']['name']
    s3_key = event['Records'][0]['s3']['object']['key']
    text = read_pdf(s3_bucket, s3_key)    
    docs = split_text(text)
    i = 0
    for doc in docs:
        embeddings = generate_embeddings(doc)
        index_data(os_client, INDEX_NAME, doc, embeddings[0], s3_key)
        print(f"Doc {i} indexed")
        i+=1

    return {
        'statusCode': 200,
        'body': json.dumps("Done")
    }


    
