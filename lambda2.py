import os
from six.moves import urllib
import json
import os
import time
import sys
from langchain.llms.bedrock import Bedrock
import boto3
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from opensearchpy import OpenSearch
import requests
from langchain.chains.summarize import load_summarize_chain
from botocore.exceptions import ClientError


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
    os_client = OpenSearch(
        hosts = [ES_HOST],
        http_auth = os_auth,
        use_ssl = True,
        verify_certs = True,
        pool_maxsize=20
        )
    if 'cluster_name' in os_client.info():
        print("os client ", os_client)
        return os_client
    else:
        return None
        

def get_summary(input_text):
    """
    Calls the Bedrock model to generate a summary of the given text.

    Args:
        input_text: The text to be summarized.

    Returns:
        The summarized text as a string.
    """
    boto3_bedrock = boto3.client(service_name='bedrock-runtime')
    modelId = "amazon.titan-text-express-v1"
    payload = {
        "inputText": input_text,
        "textGenerationConfig": {
            "max_tokens_to_sample": 8192,
            "stop_sequences": [],
            "temperature": 0,
            "top_p": 1
        }
    }
    body = {
    "inputText": f"summarize the following text without adding additional information {input_text} Response(Summarized):",
    "textGenerationConfig": {"maxTokenCount": 4096, "stopSequences": [], "temperature": 0.1, "topP": 1}
} 
    response = boto3_bedrock.invoke_model(
        body=json.dumps(body),
        contentType='application/json',
        accept='application/json',
        modelId=modelId
    )

    return response['body'].read().decode()


def summarize_chain(docs):
    """
    Summarizes the provided text chunks using a chain of models.

    Args:
        docs: A list of text chunks.

    Returns:
        The summarized text as a string.
    """
    boto3_bedrock = boto3.client(service_name='bedrock-runtime')
    modelId = "amazon.titan-text-express-v1"
    llm = Bedrock(
        model_id=modelId,
        model_kwargs={
            "maxTokenCount": 8192,
            "stopSequences": [],
            "temperature": 0,
            "topP": 1
        },
        client=boto3_bedrock,
    )

    summary_chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose=True)

    output = ""

    start = time.time()

    try:
        print('done')
        output = summary_chain.run(docs)

    except ValueError as error:
        if  "AccessDeniedException" in str(error):
            print(f"\x1b[41m{error}\
            \nTo troubeshoot this issue please refer to the following resources.\
            \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
            \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")      
            class StopExecution(ValueError):
                def render_traceback(self):
                    pass
            raise StopExecution        
        else:
            raise error

    print("OUTPUT: \n")
    print(output)

    end = time.time()
    print(f"starttime: {start} -----  endtime: {end}")
  
 

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
    
    
def query_data(index_name, query, size = 3):
    """
    Queries the OpenSearch index for documents similar to the provided query using KNN search.

    Args:
        index_name: The name of the OpenSearch index to search in.
        query: The text query to use for searching.
        size (optional): The number of similar documents to retrieve (defaults to 5).

    Returns:
        The combined text content of the retrieved documents.
    """
    embeddings = generate_embeddings(query)
    query_vector = embeddings[0]
    os_client = connect_with_opensearch()
    search_body = {
        "size": size,
        "query": {
            "knn": {
                "vector-index1": {
                    "vector": query_vector,
                    "k": size
                }
            }
        }
    }
    response = os_client.search(index=index_name, body=search_body)
    
    print("RESPONSE SENDD BY OPEN SEARCH: ")
    text = ''
    docs = []
    scores = []
    for hit in response["hits"]["hits"]:
        text = text + hit['_source']['text']
        docs.append(hit["_source"]["document-name"])
        scores.append(hit["_score"])
    
    print("TEXT: ", text)
    print("DOC NAME: ", docs)
    print("Scores: ", scores)
    average_score = sum(scores) / len(scores)
    average_score = average_score * 100
    dict = {"text" : text, "reference" : docs, "accuracy" :average_score}
    return dict

    
def lambda_handler(event, context):
    event = json.loads(event["body"])

    query_res = query_data(INDEX_NAME, event["query"])
    
    if query_res['accuracy'] >= 50:
        res = get_summary(query_res['text'])
        
        res = json.loads(res)
        
        output_text = {"answer": res["results"][0]["outputText"], "source": list(set(query_res['reference'])), "accuracy" : query_res['accuracy']}
        
        return {
            'statusCode': 200,
            'body': json.dumps(output_text)
        }
    else:
        return {
            'statusCode': 200,
            'body': json.dumps("Answer not found")
        }
        
