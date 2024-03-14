# FastAPI Service For Text To Speech Based On Voice Cloning

The FastAPI Service for Text-to-Speech Voice Cloning is a project designed to facilitate voice cloning using FastAPI. It offers endpoints tailored for generating speech in British and American accents. Leveraging models trained on the XTTS architecture, this service allows users to clone their voices and customize accents for generated speech.

The functionality of this service is centered around providing a reference audio, typically the user's voice, to allow the model to learn voice patterns. Users can input desired text, and the service generates speech with the user's voice, but in the selected accent—either British or American.

## Table of Contents
- [Imports](#imports)
- [Load Environment Variables](#load-environment-variables)
- [Define Model Directories](#define-model-directories)
- [Initialize Models](#initialize-models)
- [Load Model from Directory](#load-model-from-directory)
- [Load Accent Model](#load-accent-model)
- [Lifespan Handler](#lifespan-handler)
- [FastAPI Application Instance](#fastapi-application-instance)
- [AWS S3 Configuration](#aws-s3-configuration)
- [AWS Secrets Manager Configuration](#aws-secrets-manager-configuration)
- [API Key Authentication Configuration](#api-key-authentication-configuration)
- [Boto3 Session Initialization](#boto3-session-initialization)
- [API Key Retrieval](#api-key-retrieval)
- [Retrieve API Keys from Secrets Manager](#retrieve-api-keys-from-secrets-manager)
- [Function for Inference](#function-for-inference)
- [Rate Limit Test Route](#rate-limit-test-route)
- [British Model Route](#british-model-route)
- [American Model Route](#american-model-route)
- [Main Block](#main-block)

### Imports

The provided code imports various modules and libraries necessary for building a FastAPI service for text-to-speech (TTS) functionality. Here's a breakdown of the imports:

```python
from fastapi import FastAPI, File, UploadFile, Form
from TTS.api import TTS
import uvicorn
import torchaudio
from pathlib import Path
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import os
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse, FileResponse
import shutil
from datetime import date, datetime
import time
import secrets
import string
import boto3
from boto3 import session
import json
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException, Depends
from botocore.exceptions import ClientError
from limits.storage import RedisStorage
from limits.strategies import MovingWindowRateLimiter
from fastapi.responses import PlainTextResponse
import redis
from fastapi import Request
from contextlib import asynccontextmanager
import redis.asyncio as redis
from fastapi.websocket import WebSocket
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter, WebSocketRateLimiter
```
### Load Environment 

Using the Dotenv library, we import load_dotenv to load the environment variable file containing passwords and other sensitive information

```python
load_dotenv()
```

### Define Model Directories

Define the file paths for the models for British and American accents.

```python
BRITISH_MODEL_DIR = "/home/ubuntu/sfo-production/models/britV1"
AMERICAN_MODEL_DIR = "american"
```

### Load Model
The load_model function takes a directory path model_dir as input. It constructs file paths for the model's checkpoint, tokenizer, and configuration files within the specified directory. Then, it loads the model configuration, initializes an XTTS model based on this configuration, and sets the default speaker file path. Finally, it loads the fine-tuned model checkpoint with the specified configuration parameters and returns the loaded model.

```python
def load_model(model_dir):
    # Path to model files in the selected model directory
    CHECKPOINT_PATH = os.path.join(model_dir, "model.pth")
    TOKENIZER_PATH = os.path.join(model_dir, "vocab.json_")
    CONFIG_PATH = os.path.join(model_dir, "config.json")
    # Load model configuration
    config = XttsConfig()
    config.load_json(CONFIG_PATH)
    # Initialize the XTTS model
    model = Xtts.init_from_config(config)
    # Set default speaker file path
    default_speaker_file_path = os.path.join(model_dir, "speakers_xtts.pth")
    # Load the fine-tuned model checkpoint
    model.load_checkpoint(config, checkpoint_path=CHECKPOINT_PATH, vocab_path=TOKENIZER_PATH, speaker_file_path=default_speaker_file_path, use_deepspeed=False)
    return model
```

### Loading Accent Model 
The load_accent_model function is responsible for loading the appropriate model based on the specified accent type. It takes the accent_type as input, which is expected to be a string indicating the desired accent ("british" or "american").

```python
def load_accent_model(accent_type):
    if accent_type.lower() == "british":
        return load_model(BRITISH_MODEL_DIR)
    elif accent_type.lower() == "american":
        return load_model(AMERICAN_MODEL_DIR)
    else:
        raise ValueError("Invalid accent type. Choose 'british' or 'american'.")
```

### 'lifespan function'
The function serves as an asynchronous context manager responsible for managing the initialization and cleanup tasks during the lifespan of the FastAPI application. Upon entering the context managed block, it prints the server's start time and initializes the necessary models for text-to-speech functionality, including both American and British accents. Additionally, it establishes a connection to a Redis server for rate limiting purposes. Subsequently, it initializes the FastAPI limiter with the Redis connection and yields control back to the caller. Finally, upon exiting the context managed block, it closes the FastAPI limiter. This function encapsulates the essential setup and teardown operations required for the application's lifecycle, ensuring efficient management of resources and services.

```python
@asynccontextmanager
async def lifespan(_: FastAPI):
    print("Server start time: ",datetime.now())
    global tts_model
    global american_model
    global british_model

    # Load your TTS models here (American)
    tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    
    # american_model = load_accent_model("american")
    
    #Load British Model on startup
    british_model = british_model = load_accent_model("british")

    redis_connection = redis.from_url("redis://localhost:6379/0", encoding="utf8")
    await FastAPILimiter.init(redis_connection)
    yield
    await FastAPILimiter.close()
```

### Initializing FastAPI

```python
app = FastAPI(lifespan=lifespan)
```
### AWS Configuration and Secrets Management

#### AWS Region Configuration
- **AWS Region:** The AWS region where the Secrets Manager is 
configured.
- **Secrets Manager Setup**Name of the secret stored in the AWS Secrets Manager.
- **API Key Header Configuration**Configuration for the API key header used in requests.
- **AWS Credentials**
- Access Key ID: The AWS access key ID fetched from environment variables.
- Secret Access Key: The AWS secret access key fetched from environment variables. 

  ```python
  AWS_REGION = os.getenv("REGION")
  SECRET_NAME = os.getenv("AWS_SECRET_MANAGER")
  API_KEY_NAME = "x-api-key"
  api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
  aws_access_key_id = os.getenv("AWS_ID")
  aws_secret_access_key = os.getenv("AWS_KEY")
  ```

- **Boto3 Session Initialization** Initializes a boto3 session with the provided AWS credentials and region
```python
aws_session = session.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=AWS_REGION
)
```

- **Secrets Manager Client Initialization** Initializes a client for interacting with the AWS Secrets Manager service.
```python
secrets_manager = aws_session.client('secretsmanager')
```


