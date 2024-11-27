from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Transcription
from .serializers import TranscriptionSerializer
from .whisper_client import (WhisperTranscriptionClient, WhisperLocalTranscriptionClient, 
                           YouTubeTranscriptionClient)
from rest_framework.decorators import api_view
import requests
from django.conf import settings
import logging
import os
import re
from pytube import YouTube
from rest_framework.response import Response
from rest_framework import status
import re

# Configure logging
logger = logging.getLogger(__name__)

class TranscriptionViewSet(viewsets.ModelViewSet):
    queryset = Transcription.objects.all()
    serializer_class = TranscriptionSerializer

    @action(detail=False, methods=['post'])
    def transcribe(self, request):
        source_id = request.data.get('source_id')
        transcription_type = request.data.get('transcription_type')
        language = request.data.get('language', 'en')
        user_prompt = request.data.get('user_prompt', '')
        print(f"User prompt received in transcribe: {user_prompt}")  # Debug statement

        if not source_id or not transcription_type:
            return Response(
                {'error': 'source_id and transcription_type are required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Create client first
            if transcription_type == 'youtube':
                client = YouTubeTranscriptionClient()
            elif transcription_type == 'whisper_local':
                client = WhisperLocalTranscriptionClient()
            else:  # whisper_remote
                client = WhisperTranscriptionClient()

            # Get transcription result
            if transcription_type == 'youtube':
                result = client.transcribe_video(source_id)
            elif transcription_type == 'whisper_local':
                result = client.transcribe_local_audio(source_id)
            else:
                result = client.transcribe_audio(source_id, language)

            # Create Transcription record
            transcription = Transcription.objects.create(
                source_id=source_id,
                transcription_type=transcription_type,
                transcript=result.transcript,
                success=result.success,
                error=result.error or ''
            )

            # Create TimedWord records
            if result.success and result.timed_words:
                for word_data in result.timed_words:
                    transcription.timed_words.create(
                        word=word_data.word,
                        start=word_data.start,
                        end=word_data.end,
                        confidence=word_data.confidence
                    )

            # Generate summary if transcription was successful
            if result.success and transcription_type != 'youtube':
                whisper_client = WhisperTranscriptionClient()
                summary = whisper_client._generate_summary(result, user_prompt)
                transcription.summary = summary
                transcription.save()

            serializer = self.get_serializer(transcription)
            return Response(serializer.data)

        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@api_view(['POST'])
def test_whisper_remote(request):
    """Test Remote Whisper API Transcription"""
    print("Received request for remote transcription.")  # Debug statement
    try:
        print(f"Request data: {request.data}")  # Debug statement
        audio_url = request.data.get('audio_url')
        language = request.data.get('language', 'en')
        user_prompt = request.data.get('user_prompt', '')
        print(f"User prompt received in test_whisper_remote: {user_prompt}")  # Debug statement
        
        if not audio_url:
            print("Error: audio_url is required")
            return Response(
                {"error": "audio_url is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        print(f"Audio URL received: {audio_url}")  # Debug statement
        client = WhisperTranscriptionClient()
        
        # Test server health with extended timeout
        try:
            health_response = requests.get(
                f"{client.base_url}/health",
                timeout=30,  # Increased timeout
                # verify=getattr(settings, 'VERIFY_SSL', True)  # Removed SSL verification setting
            )
            health_status = "OK" if health_response.status_code == 200 else "Failed"
            print(f"Health check status: {health_status}")
        except requests.exceptions.RequestException as e:
            health_status = f"Failed: {str(e)}"
            print(f"Health check error: {e}")

        # Validate audio URL
        if not client._check_audio_url(audio_url):
            print("Error: Audio URL is not accessible")
            return Response(
                {"error": "Audio URL is not accessible"}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        print("Starting transcription...")  # Debug statement
        # Test transcription
        result = client.transcribe_audio(audio_url, language)
        print(f"Transcription result: {result}")
        
        response_data = {
            "server_health": health_status,
            "transcription_success": result.success,
            "transcript": result.transcript if result.success else None,
            "error": result.error if not result.success else None,
            "summary": None,
            "timed_words": result.timed_words if result.success else None
        }

        if result.success:
            summary = client._generate_summary(result, user_prompt)
            response_data["summary"] = summary
            print(f"Generated summary: {summary}")

        print("Returning response data.")  # Debug statement
        return Response(response_data)

    except Exception as e:
        print(f"Error in test_whisper_remote: {e}")
        return Response(
            {"error": f"Server error: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def test_whisper_local(request):
    """Test Local Whisper Transcription"""
    # Get the uploaded audio file from the request
    audio_file = request.FILES.get('audio_file')
    
    if not audio_file:
        return Response(
            {"error": "Audio file is required"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    # Save the uploaded file temporarily
    file_path = f"/tmp/{audio_file.name}"  # Adjust the path as needed
    with open(file_path, 'wb+') as destination:
        for chunk in audio_file.chunks():
            destination.write(chunk)

    try:
        user_prompt = request.data.get('user_prompt', '')
        print(f"User prompt received in test_whisper_local: {user_prompt}")  # Debug statement
        local_client = WhisperLocalTranscriptionClient()
        result = local_client.transcribe_local_audio(file_path)
        
        # Log the transcription result
        logger.info(f"Transcription result: {result}")  # Log the result

        # Convert timed_words to a serializable format
        timed_words_serializable = [
            word.to_dict() for word in result.timed_words
        ] if result.success else None

        response_data = {
            "transcription_success": result.success,
            "transcript": result.transcript if result.success else None,
            "error": result.error if not result.success else None,
            "summary": None,
            "timed_words": timed_words_serializable
        }

        if result.success:
            remote_client = WhisperTranscriptionClient()
            summary = remote_client._generate_summary(result, user_prompt)
            logger.info(f"Generated summary: {summary}")  # Log the generated summary
            response_data["summary"] = summary

        return Response(response_data)

    except Exception as e:
        # Catch block to handle exceptions during transcription
        logger.error(f"An error occurred during transcription: {str(e)}")  # Log the error
        return Response(
            {"error": f"An error occurred during transcription: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
# @api_view(['POST'])
# def test_whisper_local(request):
#     """Test Local Whisper Transcription"""
#     # Hardcode the audio file path for testing
#     file_path = r"C:\Users\91966\Desktop\transcript\sample4.m4a"  # Adjusted to hardcoded path

#     # Check if the file exists (optional, for safety)
#     if not os.path.exists(file_path):
#         return Response(
#             {"error": "Audio file does not exist at the specified path"}, 
#             status=status.HTTP_400_BAD_REQUEST
#         )

#     try:
#         local_client = WhisperLocalTranscriptionClient()
#         result = local_client.transcribe_local_audio(file_path)
#     except Exception as e:  # Added exception handling
#         logger.error(f"An error occurred during local transcription: {str(e)}")
#         return Response(
#             {"error": f"An error occurred during transcription: {str(e)}"}, 
#             status=status.HTTP_500_INTERNAL_SERVER_ERROR
#         )

#     # Log the transcription result
#     logger.info(f"Transcription result: {result}")  # Log the result

#     # Convert timed_words to a serializable format
#     timed_words_serializable = [
#         word.to_dict() for word in result.timed_words
#     ] if result.success else None

#     response_data = {
#         "transcription_success": result.success,
#         "transcript": result.transcript if result.success else None,
#         "error": result.error if not result.success else None,
#         "summary": None,
#         "timed_words": timed_words_serializable
#     }

#     if result.success:
#         remote_client = WhisperTranscriptionClient()
#         summary = remote_client._generate_summary(result)
#         logger.info(f"Generated summary: {summary}")  # Log the generated summary
#         response_data["summary"] = summary

#     return Response(response_data)

@api_view(['POST'])
# def test_youtube(request):
#     """Test YouTube Transcription"""
#     print("Received request for YouTube transcription.")  # Debug statement
#     youtube_url = request.data.get('youtube_url')  # Changed from video_id to youtube_url
#     user_prompt = request.data.get('user_prompt', '')  # Retrieve user_prompt from request data
#     print(f"User prompt received in test_youtube: {user_prompt}")  # Debug statement
    
#     if not youtube_url:
#         print("Error: youtube_url is required.")  # Debug statement
#         return Response(
#             {"error": "youtube_url is required"}, 
#             status=status.HTTP_400_BAD_REQUEST
#         )

#     # Extract video ID from the YouTube URL
#     video_id = extract_video_id(youtube_url)  # New function to extract video ID
#     if not video_id:
#         return Response(
#             {"error": "Invalid YouTube URL"}, 
#             status=status.HTTP_400_BAD_REQUEST
#         )

#     youtube_client = YouTubeTranscriptionClient()
    
#     # Get transcription
#     print("Transcribing video...")  # Debug statement
#     result = youtube_client.transcribe_video(video_id)
    
#     response_data = {
#         "transcription_success": result.success,
#         "transcript": result.transcript if result.success else None,
#         "error": result.error if not result.success else None,
#         "summary": None
#     }

#     if result.success:
#         print("Transcription successful, generating summary...")  # Debug statement
#         print(f"Result for summary generation: {result}")  # Debug statement to check result
#         print(f"User prompt for summary generation: {user_prompt}")  # Debug statement to check user prompt
#         whisper_client = WhisperTranscriptionClient()  # Instantiate the client
#         summary = whisper_client._generate_summary(result, user_prompt)  # Use the retrieved user_prompt
#         print(f"Generated summary: {summary}")  # Debug statement to check generated summary
#         response_data["summary"] = summary

#     print("Returning response data.")  # Debug statement
#     return Response(response_data)

# def extract_video_id(url):
#     """Extract video ID from YouTube URL."""
#     # Regular expression to match YouTube video ID without look-behind
#     match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})', url)
#     return match.group(1) if match else None  # Return the first capturing group



def test_youtube(request):
    """Test YouTube Transcription."""
    print("Received request for YouTube transcription.")  # Debug statement
    youtube_url = request.data.get('youtube_url')  # Changed from video_id to youtube_url
    user_prompt = request.data.get('user_prompt', '')  # Retrieve user_prompt from request data
    print(f"User prompt received in test_youtube: {user_prompt}")  # Debug statement

    if not youtube_url:
        print("Error: youtube_url is required.")  # Debug statement
        return Response(
            {"error": "youtube_url is required"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    # Extract video ID from the YouTube URL
    try:
        video_id = extract_video_id(youtube_url)
    except ValueError as e:
        print(f"Error extracting video ID: {e}")  # Debug statement
        return Response(
            {"error": "Invalid YouTube URL"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    youtube_client = YouTubeTranscriptionClient()

    # Get transcription
    print("Transcribing video...")  # Debug statement
    result = youtube_client.transcribe_video(video_id)

    # Check if result is valid
    if isinstance(result, str):  # If result is a string, it indicates an error
        print(f"Transcription error: {result}")  # Debug statement
        return Response(
            {"error": result}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    # Ensure result is an object with expected attributes
    if not hasattr(result, 'success') or not hasattr(result, 'transcript'):
        print(f"Unexpected result format: {result}")  # Debug statement
        return Response(
            {"error": "Unexpected result format from transcription service."}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    response_data = {
        "transcription_success": result.success,
        "transcript": result.transcript if result.success else None,
        "error": result.error if not result.success else None,
        "summary": None
    }

    if result.success:
        print("Transcription successful, generating summary...")  # Debug statement
        whisper_client = WhisperTranscriptionClient()  # Instantiate the client
        try:
            # Debug statement to log the type and content of result
            print(f"Result type: {type(result)}, Result content: {result}")  # Debug statement
            
            # Check if result has the expected attributes
            if hasattr(result, 'transcript'):
                print(f"Transcript: {result.transcript}")  # Debug statement
            else:
                print("Error: 'result' does not have a 'transcript' attribute.")  # Debug statement
            
            # Ensure result.transcript is valid before generating summary
            summary = whisper_client._generate_summary(result.transcript, user_prompt)  # Use the retrieved user_prompt
            response_data["summary"] = summary
        except Exception as e:
            print(f"Error generating summary: {e}")  # Debug statement
            response_data["summary_error"] = str(e)

    print("Returning response data.")  # Debug statement
    return Response(response_data)


def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})', url)
    if match:
        return match.group(1)
    raise ValueError("Invalid YouTube URL format.")


@api_view(['GET'])
def test_whisper_connection(request):
    """Test Whisper API Connection"""
    try:
        client = WhisperTranscriptionClient()
        print("Attempting to connect to Whisper API.")

        # Test server connection
        try:
            health_response = requests.get(
                f"http://20.244.108.21:8000/health",
                timeout=10,
                headers=client._get_headers()
            )
            logger.info("Received response from Whisper API health check.")

            return Response({
                "status": "success",
                "server_health": "OK" if health_response.status_code == 200 else "Failed",
                "status_code": health_response.status_code,
                "response": health_response.text
            })
            
        except requests.exceptions.RequestException as e:
            logger.error(f"RequestException occurred: {str(e)}")
            return Response({
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return Response({
            "status": "error",
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def generate_summary(request):
    user_prompt = request.data.get('prompt', '')
    
    # Assuming you have a way to get the transcription result
    # For example, you might want to pass the transcription result from the frontend
    transcription_result = ...  # Get the transcription result from your logic

    client = WhisperTranscriptionClient()  # Or however you instantiate your client
    summary = client._generate_summary(transcription_result, user_prompt)  # Pass the user prompt

    return Response({'summary': summary})




