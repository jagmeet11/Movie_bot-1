# FastAPI Voice Cloning Service 

This project is a FastAPI-based Service for Voice cloning. It provides endpoints for generating speech in both British and American accents by leveraging models trained on the XTTS architecture.

## Setup

1. **Clone Repository**: Clone this repository to your local machine.
   - **Command Line Prompt**: 
     ```bash
     git clone https://github.com/raen-ai/SFO-API-POC.git
     ```

2. **Create Environment**: Creates a virtual environment for the project.
   - **Command Line Prompt**: 
     ```bash
     python -m venv env
     ```
     or 
     ```bash
     python3 -m venv env
     ```

3. **Activate Environment**: Activates virtual environment.
   - **Command Line Prompt**:
     For Windows:
       ```bash
       env/Scripts/Activate.ps1
       ```
       or (optional)
       ```bash
       env/Scripts/Activate.bat
       ```
     For Linux:
       ```bash
       source env/bin/activate
       ```


## Endpoints

### British Accent Speech Generation
- **Endpoint**: `/api/british`
- **Method**: `POST`
- **Parameters**:
  - `text`: Text to be converted to speech.
  - `wav_file`: Uploaded WAV file for reference audio.
- **Rate Limit**: 10 requests per minute.
- **Response**:
  - `message`: Output status message.
  - `s3_input_presigned_url`: Presigned URL for input audio file.
  - `s3_output_presigned_url`: Presigned URL for output speech file.
  - `process_time`: Processing time in seconds.
  - `infer_time`: Inference time in seconds.

### American Accent Speech Generation
- **Endpoint**: `/api/american`
- **Method**: `POST`
- **Parameters**:
  - `text`: Text to be converted to speech.
  - `wav_file`: Uploaded WAV file for reference audio.
  - `language`: Language for TTS (default: "en").
  - `split_sentences`: Whether to split sentences (default: True).
- **Rate Limit**: 10 requests per minute.
- **Response**:
  - `message`: Output status message.
  - `s3_input_presigned_url`: Presigned URL for input audio file.
  - `s3_output_presigned_url`: Presigned URL for output speech file.
  - `process_time`: Processing time in seconds.
  - `infer_time`: Inference time in seconds.


## Authentication

API key authentication is implemented using the `x-api-key` header. Ensure that you provide a valid API key with each request.

## Error Handling

Errors are handled appropriately, and detailed error messages are provided in case of failures.