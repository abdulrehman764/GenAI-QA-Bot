# GenAI-QA-Bot
This repository implements a Generative AI API using SageMaker and Bedrock models to generate responses to user-provided questions. It leverages pre-trained models for sentence embedding, vector storage, and text generation.


# Generative AI API Framework

## Introduction

This repository provides a framework for a Generative AI API built on top of AWS services like SageMaker and OpenSearch. The API utilizes pre-trained models to generate responses to user-provided questions.

## Technologies Used

- **Data Storage:** AWS S3
- **Sentence Embedding:** SageMaker JumpStart's `sentence-transformers/all-MiniLM-L6-v2` model
- **Vector Storage:** OpenSearch
- **Text Generation:** Bedrock Titan model

## Project Structure

- **lambda_1.py:** Script for indexing data in OpenSearch after generating embeddings.
- **lambda_2.py:** Script for the API endpoint that processes user queries, retrieves relevant data from OpenSearch, and generates responses using the Bedrock model.
- **requirements.txt:** File containing dependencies required for the project.

## Instructions

### Prerequisites

- An AWS account with SageMaker, OpenSearch, and Bedrock resources set up.
- Access to encrypted S3 bucket containing the data.

### Environment Variables

Set the following environment variables in your AWS account:

- `ES_HOST`: OpenSearch host endpoint.
- `ENDPOINT_NAME`: SageMaker endpoint name for sentence embedding model.
- `SECRET_NAME`: Name of the AWS Secrets Manager secret containing OpenSearch credentials.
- `INDEX_NAME`: Name of the OpenSearch index used for storing data.

### Deploying the API

1. Deploy the `lambda_1.py` script as a Lambda function. This function is responsible for indexing data in OpenSearch.
2. Deploy the `lambda_2.py` script as a Lambda function and configure it as the API endpoint. This function handles user queries, retrieves relevant data, and generates responses.

### Sending a Request

Send a POST request to the API gateway endpoint with the user query in the body of the request (JSON format).

#### Example Request:

{
  "query": "What is the capital of France?"
}


#### Example Response:
{
  "answer": "Paris",
  "source": ["document1.txt"],
  "accuracy": 87.5
}

