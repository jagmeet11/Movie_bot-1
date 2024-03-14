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

### Fetching API
- **Asynchronous Function:** The function is asynchronous, allowing non-blocking execution for improved performance.
- **Parameter:** Accepts an API key as a string parameter obtained from the `APIKeyHeader` security dependency.
- **Debug Print:** Prints the received API key for debugging purposes.
- **Retrieve API Keys:** Calls the `retrieve_api_keys` function to retrieve stored API keys from the AWS Secrets Manager.
- **Validation:** Validates the received API key against the retrieved keys. If the API key is not found in the retrieved keys, it raises an `HTTPException` with a status code of 403 (Forbidden) and a message indicating an invalid API key.
- **Return:** Returns the validated API key.

```python
async def get_api_key(api_key: str = Security(api_key_header)):
    print(f"Received API key: {api_key}")  # Debug print
    api_keys = await retrieve_api_keys(SECRET_NAME)
    if api_key not in api_keys.values():
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
```
### Function Retirve keys

- **Asynchronous Function:** The function is asynchronous, allowing non-blocking execution for improved performance.
- **Parameter:** Accepts the name of the secret as a string parameter.
- **AWS Secrets Manager Retrieval:** Utilizes the `secrets_manager.get_secret_value` method to retrieve the secret from the AWS Secrets Manager based on the provided secret name.
- **Secret Retrieval:** Retrieves the secret string from the response.
- **Debug Print:** Prints the retrieved secret string for debugging purposes.
- **JSON Parsing:** Parses the secret string as JSON to obtain the API keys.
- **Error Handling:** Handles potential errors such as `ClientError` and other exceptions. In case of errors, it raises an `HTTPException` with a status code of 500 (Internal Server Error) and provides appropriate error details.

```python
async def retrieve_api_keys(secret_name: str):
    try:
        get_secret_value_response = secrets_manager.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response['SecretString']
        print(f"Retrieved secret: {secret}")  # Debug print
        api_keys = json.loads(secret)
        return api_keys
    except ClientError as e:
        print(f"An error occurred: {e.response['Error']['Message']}")
        raise HTTPException(status_code=500, detail=f"Secrets Manager Service Error: {e.response['Error']['Message']}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")
```

### perform_inference Function
- **Description:** Performs inference using the provided model, input text, and reference audio.
- **Parameters:**
  - `model`: The TTS model to use for inference.
  - `text`: The input text to generate speech from.
  - `reference_audio`: The path to the reference audio file for conditioning.
- **Process:**
  1. Extracts conditioning latents and speaker embedding using the reference audio.
  2. Performs inference to generate speech based on the input text, conditioning latents, and speaker embedding.
- **Return Value:** Returns the output of the inference process.

```python
def perform_inference(model, text, reference_audio):
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=reference_audio)
    output = model.inference(text, "en", gpt_cond_latent, speaker_embedding)
    return output
```

### Test Route for Rate Limits (/api/rate)
- **Description:** Provides a test route to check the rate limit functionality.
- **Endpoint:** `/api/rate`
- **HTTP Method:** GET
- **Rate Limit:** 1 request per 60 seconds
- **Dependencies:** Utilizes the `RateLimiter` dependency to enforce rate limits.
- **API Key Dependency:** Optionally depends on the `get_api_key` function to validate API keys.
- **Response:** Returns a string indicating that the endpoint is working if the request is within the rate limit.

```python
@app.get("/api/rate", dependencies=[Depends(RateLimiter(times=1, seconds=60))])
async def test(api_key: str = Depends(get_api_key)):
    return "Endpoint is working"
```

### British Model Route
- **Description:** Handles requests to generate British-accented speech using the provided input text and WAV file.
- **Endpoint:** `/api/british`
- **HTTP Method:** POST
- **Rate Limit:** 10 requests per 60 seconds
- **Dependencies:** Depends on the `RateLimiter` dependency to enforce rate limits. Additionally, depends on the `get_api_key` function for API key validation.
- **Parameters:**
  - `api_key`: API key for authentication and authorization.
  - `text`: Input text to generate speech.
  - `wav_file`: Uploaded WAV file containing reference audio.
- **Process:**
  1. Starts measuring the execution time.
  2. Generates a unique identifier for file names based on the current date and a random string.
  3. Loads the British accent model.
  4. Defines input and output file names based on the unique identifier.
  5. Saves the uploaded WAV file with the new name.
  6. Performs inference to generate British-accented speech.
  7. Saves the generated speech as a WAV file.
  8. Uploads input and output files to an S3 bucket and generates presigned URLs.
  9. Deletes the local files after uploading to S3.
  10. Ends the execution time measurement.
- **Response:** Returns a JSON object containing details about the generated speech, including presigned URLs for input and output files, processing time, and inference time.

```python
@app.post("/api/british", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def generate_speech_british(api_key: str = Depends(get_api_key),text: str = Form(...), wav_file: UploadFile = File(...)):
    # Start time measurement
    start_time = time.time()  

    # Generate a unique identifier for the file names
    random_string = ''.join(secrets.choice(string.ascii_uppercase + string.ascii_lowercase) for i in range(7))
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")
    unique_identifier = f"{random_string}_{d4}"


    try:
        british_model = load_accent_model("british")

        # Define the input file name with the same unique identifier
        original_filename = Path(wav_file.filename)
        input_file_path = f"audio_inputs/input_{original_filename.stem}_{unique_identifier}{original_filename.suffix}"

        # Define the output file name with the unique identifier
        out_name = f"audio_outputs/output_{original_filename.stem}_{unique_identifier}.wav"


        # Save the uploaded file with the new name
        with open(input_file_path, "wb") as buffer:
            shutil.copyfileobj(wav_file.file, buffer)

        # Perform Inference
        start_time_inf = time.time() 
        output = perform_inference(british_model, text, input_file_path)
        torchaudio.save(out_name, torch.tensor(output["wav"]).unsqueeze(0), 24000)
        end_time_inf = time.time()

        # Upload files to S3 and generate presigned URLs
        input_key = f"inputs/{Path(input_file_path).name}"
        output_key = f"outputs/{Path(out_name).name}"
        
        # Upload the input file
        s3.upload_file(Filename=input_file_path, Bucket=BUCKET_NAME, Key=input_key)
        # Generate a presigned URL for the input file
        input_presigned_url = s3.generate_presigned_url('get_object', Params={'Bucket': BUCKET_NAME, 'Key': input_key}, ExpiresIn=3600)
        
        # Upload the output file
        s3.upload_file(Filename=out_name, Bucket=BUCKET_NAME, Key=output_key)
        # Generate a presigned URL for the output file
        output_presigned_url = s3.generate_presigned_url('get_object', Params={'Bucket': BUCKET_NAME, 'Key': output_key}, ExpiresIn=3600)

        # Delete the local files after uploading to S3
        os.remove(input_file_path)
        os.remove(out_name)

        # End time measurement
        end_time = time.time()  

        processing_time = end_time - start_time  # Calculate processing time
        infer_time = end_time_inf - start_time_inf # Calculate inference time

        return {
            "message": "Output Generated.",
            "s3_input_presigned_url": input_presigned_url,
            "s3_output_presigned_url": output_presigned_url,
            "process_time": processing_time,
            "infer_time": infer_time
        }

    except Exception as e:
        print(str(e))
        return {"error": "There was an error"}
```

### American Model Route
- **Description:** Handles requests to generate American-accented speech using the provided input text and WAV file.
- **Endpoint:** `/api/american`
- **HTTP Method:** POST
- **Rate Limit:** 10 requests per 60 seconds
- **Dependencies:** Depends on the `RateLimiter` dependency to enforce rate limits. Additionally, depends on the `get_api_key` function for API key validation.
- **Parameters:**
  - `api_key`: API key for authentication and authorization.
  - `text`: Input text to generate speech.
  - `wav_file`: Uploaded WAV file containing reference audio.
  - `language`: Language for speech generation (default is "en").
  - `split_sentences`: Boolean indicating whether to split text into sentences (default is True).
- **Process:**
  1. Starts measuring the execution time.
  2. Generates a unique identifier for file names based on the current date and a random string.
  3. Defines input and output file names based on the unique identifier.
  4. Saves the uploaded WAV file with the new name.
  5. Calls the TTS function to generate American-accented speech with the provided parameters.
  6. Uploads input and output files to an S3 bucket and generates presigned URLs.
  7. Deletes the local files after uploading to S3.
  8. Ends the execution time measurement.
- **Response:** Returns a JSON object containing details about the generated speech, including presigned URLs for input and output files, processing time, and inference time.

```python
@app.post("/api/american",dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def generate_speech_american(api_key: str = Depends(get_api_key),text: str = Form(...), wav_file: UploadFile = File(...), language: str = "en", split_sentences: bool = True):
    
    # Start time measurement
    start_time = time.time()  

    # Generate a unique identifier for the file names
    random_string = ''.join(secrets.choice(string.ascii_uppercase + string.ascii_lowercase) for i in range(7))
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")
    unique_identifier = f"{random_string}_{d4}"

    try:
        # Define the input file name with the same unique identifier
        original_filename = Path(wav_file.filename)
        input_file_path = f"audio_inputs/input_{original_filename.stem}_{unique_identifier}{original_filename.suffix}"

        # Define the output file name with the unique identifier
        out_name = f"audio_outputs/output_{original_filename.stem}_{unique_identifier}.wav"


        # Save the uploaded file with the new name
        with open(input_file_path, "wb") as buffer:
            shutil.copyfileobj(wav_file.file, buffer)

        start_time_inf = time.time() 

        # Call the TTS function with the provided parameters
        tts_model.tts_to_file(text=text,
                            file_path=out_name,
                            speaker_wav=[input_file_path],
                            language=language,
                            split_sentences=split_sentences)
        
        end_time_inf = time.time()

        # Upload files to S3 and generate presigned URLs
        input_key = f"inputs/{Path(input_file_path).name}"
        output_key = f"outputs/{Path(out_name).name}"
        
        # Upload the input file
        s3.upload_file(Filename=input_file_path, Bucket=BUCKET_NAME, Key=input_key)
        # Generate a presigned URL for the input file
        input_presigned_url = s3.generate_presigned_url('get_object', Params={'Bucket': BUCKET_NAME, 'Key': input_key}, ExpiresIn=3600)
        
        # Upload the output file
        s3.upload_file(Filename=out_name, Bucket=BUCKET_NAME, Key=output_key)
        # Generate a presigned URL for the output file
        output_presigned_url = s3.generate_presigned_url('get_object', Params={'Bucket': BUCKET_NAME, 'Key': output_key}, ExpiresIn=3600)

        # Delete the local files after uploading to S3
        os.remove(input_file_path)
        os.remove(out_name)

        # End time measurement
        end_time = time.time()  

        processing_time = end_time - start_time  # Calculate processing time
        infer_time = end_time_inf - start_time_inf # Calculate inference time

        return {
            "message": "Output Generated.",
            "s3_input_presigned_url": input_presigned_url,
            "s3_output_presigned_url": output_presigned_url,
            "process_time": processing_time,
            "infer_time": infer_time
        }
    except Exception as e:
        print(str(e))
        return {"error": "There was an error."}
```