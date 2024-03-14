# FastAPI Service For Text To Speech Based On Voice Cloning

The FastAPI Service for Text-to-Speech Voice Cloning is a project designed to facilitate voice cloning using FastAPI. It offers endpoints tailored for generating speech in British and American accents. Leveraging models trained on the XTTS architecture, this service allows users to clone their voices and customize accents for generated speech.

The functionality of this service is centered around providing a reference audio, typically the user's voice, to allow the model to learn voice patterns. Users can input desired text, and the service generates speech with the user's voice, but in the selected accent—either British or American.

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
     - For Windows:
       ```bash
       env/Scripts/Activate.ps1
       ```
       or (optional)
       ```bash
       env/Scripts/Activate.bat
       ```
     - For Linux:
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