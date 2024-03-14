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

# Imports

The provided code imports various modules and libraries necessary for building a FastAPI service for text-to-speech (TTS) functionality. Here's a breakdown of the imports:

```python```
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
from fastapi_limiter.depends import RateLimiter,       WebSocketRateLimiter

# Load Environment Variables

Using the Dotenv library we import load_dotenv which loads the environment variable file which contains the passwords and other sensitive information 

``` Load_dotenv() ```

# Model Initialization 

The file path for the models for british and american accents have to be defined 

``` BRITISH_MODEL_DIR = "/home/ubuntu/sfo-production/models/britV1" ```
```AMERICAN_MODEL_DIR = "american"```

# Load Model

A fucntion Load_model is defined to load a TTS model from a specified directory.

The function takes in the file path as a parameter and then 

model parameters:

```    CHECKPOINT_PATH = os.path.join(model_dir, "model.pth")   ```
```      TOKENIZER_PATH = os.path.join(model_dir, "vocab.json_")```
   ``` CONFIG_PATH = os.path.join(model_dir, "config.json")```


