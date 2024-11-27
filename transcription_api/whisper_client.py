import requests
from typing import Dict, Any, Optional
from dataclasses import dataclass
from typing import List
import json
import google.generativeai as genai
import os
import whisper
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

# Load the .env file from the current directory
load_dotenv()

# Now you can access the environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

@dataclass
class TimedWord:
    word: str
    start: float
    end: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert TimedWord to a dictionary for JSON serialization."""
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence
        }

@dataclass
class TranscriptionResponse:
    transcript: str
    timed_words: List[TimedWord]
    success: bool
    error: Optional[str] = None

class YouTubeTranscriptionClient:
    def transcribe_video(self, video_id: str, languages: List[str] = ['en'], 
                        preserve_formatting: bool = False) -> TranscriptionResponse:
        """
        Transcribe a YouTube video using its ID.
        
        Args:
            video_id: The YouTube video ID
            languages: List of language codes in order of preference
            preserve_formatting: Whether to keep HTML formatting like <i> and <b>
        """
        try:
            # Get transcript from YouTube
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find transcript in preferred languages
            transcript = transcript_list.find_transcript(languages)
            
            # Get the actual transcript data
            transcript_data = transcript.fetch()
            
            # Combine all text segments
            full_transcript = ' '.join(segment['text'] for segment in transcript_data)
            
            # Convert YouTube transcript format to TimedWord format
            timed_words = [
                TimedWord(
                    word=segment['text'],
                    start=segment['start'],
                    end=segment['start'] + segment['duration'],
                    confidence=1.0
                )
                for segment in transcript_data
            ]
            
            return TranscriptionResponse(
                transcript=full_transcript,
                timed_words=timed_words,
                success=True
            )
            
        except Exception as e:
            return TranscriptionResponse(
                transcript="",
                timed_words=[],
                success=False,
                error=f"Error fetching YouTube transcript: {str(e)}"
            )

    def get_available_transcripts(self, video_id: str) -> List[Dict[str, Any]]:
        """Get metadata about available transcripts for a video."""
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available_transcripts = []
            
            for transcript in transcript_list:
                available_transcripts.append({
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable,
                    'translation_languages': transcript.translation_languages
                })
                
            return available_transcripts
        except Exception as e:
            return []


class WhisperLocalTranscriptionClient:
    def __init__(self, model_size: str = "base"):
        import torch
        
        # Determine device and precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the model based on the specified size
        self.model = whisper.load_model(model_size)
        self.model.to(self.device)
        
        # Set appropriate precision based on device
        if self.device.type == "cpu":
            # Force FP32 on CPU
            self.model.eval()
            self.model = self.model.float()
        else:
            # Use FP16 on GPU
            self.model = self.model.half()

    def transcribe_local_audio(self, file_path: str) -> TranscriptionResponse:
        try:
            # Ensure model is in eval mode
            self.model.eval()
            
            # Transcribe with appropriate device settings
            result = self.model.transcribe(
                file_path,
                fp16=(self.device.type == "cuda")  # Only use fp16 with CUDA
            )
            
            timed_words = [
                TimedWord(
                    word=segment["text"],
                    start=segment["start"],
                    end=segment["end"],
                    confidence=1.0,
                )
                for segment in result.get("segments", [])
            ]
            return TranscriptionResponse(
                transcript=result["text"],
                timed_words=timed_words,
                success=True
            )
        except Exception as e:
            return TranscriptionResponse(
                transcript="",
                timed_words=[],
                success=False,
                error=f"Error during transcription: {str(e)}"
            )


class WhisperTranscriptionClient:
    def __init__(self):
        self.base_url =  os.getenv('WHISPER_BASE_URL')
        
        
        self.auth_token = os.getenv('WHISPER_AUTH_TOKEN')
        if not self.auth_token:
            raise ValueError("WHISPER_AUTH_TOKEN is not set in the environment variables.")
        self.timeout = 300
        
        # Initialize Gemini only if API key is available
        if not GEMINI_API_KEY:
            self.model = None
            return
            
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            self.model = None

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.auth_token}",  # Changed to Bearer token format
            "token": self.auth_token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        return headers

    def _check_audio_url(self, audio_url: str) -> bool:
        """Validate if audio URL is accessible"""
        try:
            head_response = requests.head(audio_url, timeout=10)
            content_type = head_response.headers.get('Content-Type', '')
            return head_response.status_code == 200
        except Exception as e:
            return False
        
    def transcribe_audio(self, audio_url: str, language: str = "en") -> TranscriptionResponse:
        try:
            # Validate audio URL first
            if not self._check_audio_url(audio_url):
                return TranscriptionResponse(
                    transcript="",
                    timed_words=[],
                    success=False,
                    error="Audio URL is not accessible"
                )

            endpoint = f"{self.base_url}/predict/v2"
            
            # Try different payload formats
            payload = {
                "audio_url": audio_url,
                "language": language,
                "task": "transcribe",
                "return_timestamps": True
            }

            # Try making request with different methods
            try:
                # Method 1: Standard JSON request
                response = requests.post(
                    endpoint,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
                if response.status_code != 200:
                    raise requests.exceptions.RequestException(f"Request failed with status {response.status_code}")
                
            except requests.exceptions.RequestException as e:
                # Method 2: Form data request
                form_data = {
                    "audio_url": (None, audio_url),
                    "language": (None, language)
                }
                response = requests.post(
                    endpoint,
                    headers={k: v for k, v in self._get_headers().items() if k != 'Content-Type'},
                    files=form_data,
                    timeout=self.timeout
                )

            if response.status_code != 200:
                return TranscriptionResponse(
                    transcript="",
                    timed_words=[],
                    success=False,
                    error=f"Server error: {response.text}"
                )

            result = response.json()
            
            # Handle different response formats
            if isinstance(result, str):
                return TranscriptionResponse(
                    transcript=result,
                    timed_words=[],
                    success=True
                )
                
            timed_words = []
            if "timed_words" in result:
                timed_words = [
                    TimedWord(
                        word=word.get("text", ""),
                        start=word.get("start", 0.0),
                        end=word.get("end", 0.0),
                        confidence=word.get("confidence", 0.0)
                    ).to_dict()  # Convert to dictionary for serialization
                    for word in result["timed_words"]
                ]
                
            return TranscriptionResponse(
                transcript=result.get("transcript", ""),
                timed_words=timed_words,
                success=True
            )
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error during transcription: {str(e)}"
            return TranscriptionResponse(
                transcript="",
                timed_words=[],
                success=False,
                error=error_msg
            )
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            return TranscriptionResponse(
                transcript="",
                timed_words=[],
                success=False,
                error=error_msg
            )

    def _generate_summary(self, transcript: str, user_prompt: str) -> str:
        """Generate a summary of the transcript using Gemini and user prompt
        
        Args:
            transcript: The text transcript to summarize
            user_prompt: User-provided context or requirements for the summary
            
        Returns:
            str: Generated summary or empty string if generation fails
        """
        if self.model is None:
            print("Model is not initialized")  # Debug statement
            return ""
        
        try:
            print(f"Transcript for summary generation: {transcript}")  # Debug statement
            print(f"User prompt for summary generation: {user_prompt}")  # Debug statement

            prompt = f"""Please provide a concise summary of the following transcript. 
            Focus on the main points and key messages.

            Transcript:
            {transcript}

            User Prompt:
            {user_prompt}

            Summary:"""

            # Generate summary using the Gemini model
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            print(f"Generated summary: {summary}")  # Debug statement
            return summary
            
        except Exception as e:
            print(f"Error during summary generation: {str(e)}")  # Debug statement
            return ""

    def _generate_summary_youtube(self, transcript: str, user_prompt: str) -> str:
        """Generate a summary of the YouTube transcript using Gemini and user prompt
        
        Args:
            transcript: The text transcript to summarize
            user_prompt: User-provided context or requirements for the summary
            
        Returns:
            str: Generated summary or empty string if generation fails
        """
        if self.model is None:
            print("Model is not initialized")  # Debug statement
            return ""
        
        try:
            print(f"Transcript for YouTube summary generation: {transcript}")  # Debug statement
            print(f"User prompt for YouTube summary generation: {user_prompt}")  # Debug statement

            prompt = f"""Please provide a concise summary of the following YouTube transcript. 
            Focus on the main points and key messages.

            Transcript:
            {transcript}

            User Prompt:
            {user_prompt}

            Summary:"""

            # Generate summary using the Gemini model
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            print(f"Generated YouTube summary: {summary}")  # Debug statement
            return summary
            
        except Exception as e:
            print(f"Error during YouTube summary generation: {str(e)}")  # Debug statement
            return ""


def test_transcription():
    client = WhisperTranscriptionClient()
    
    # Test with multiple audio formats
    test_urls = [
        "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav",  # WAV file
    ]
    
    for audio_url in test_urls:
        result = client.transcribe_audio(audio_url=audio_url)
        
        if result.success:
            print("Transcription successful!")
            print(f"Transcript: {result.transcript}")
            
            # Generate and print summary
            summary = client._generate_summary(result.transcript)
            if summary:
                print("\nSummary:")
                print(summary)
            else:
                print("\nCould not generate summary")
        else:
            print(f"Transcription failed: {result.error}")

if __name__ == "__main__":
    print("\n=== Testing All Transcription Methods ===\n")

    # 1. Test Server Connection and Remote Whisper API
    print("1. Testing Remote Whisper API Transcription:")
    print("-" * 50)
    client = WhisperTranscriptionClient()
    
    # Test server health
    try:
        response = requests.get(f"{client.base_url}/health")
        print("Server health check: OK")
    except Exception as e:
        print(f"Server health check failed: {str(e)}")
    
    # Test remote transcription
    test_audio_url = "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg.wav"
    result = client.transcribe_audio(test_audio_url)
    if result.success:
        print("Remote transcription successful!")
        print(f"Transcript: {result.transcript}")
        summary = client._generate_summary(result.transcript)
        if summary:
            print("\nSummary:")
            print(summary)
    else:
        print(f"Remote transcription failed: {result.error}")

    # 2. Test Local Whisper
    print("\n2. Testing Local Whisper Transcription:")
    print("-" * 50)
    local_client = WhisperLocalTranscriptionClient()
    file_path = "sample4.m4a"  # Replace with your local audio file
    
    result = local_client.transcribe_local_audio(file_path)
    if result.success:
        print("Local transcription successful!")
        print(f"Transcript: {result.transcript}")
        # Use the remote client's Gemini integration for summary
        summary = client._generate_summary(result.transcript)
        if summary:
            print("\nSummary:")
            print(summary)
    else:
        print(f"Local transcription failed: {result.error}")

    # 3. Test YouTube Transcription
    print("\n3. Testing YouTube Transcription:")
    print("-" * 50)
    video_id = "iCvmsMzlF7o"
    youtube_client = YouTubeTranscriptionClient()
    
    # First get available transcripts
    available_transcripts = youtube_client.get_available_transcripts(video_id)
    if available_transcripts:
        print(f"Number of available transcripts: {len(available_transcripts)}")
    
    result = youtube_client.transcribe_video(video_id)
    if result.success:
        print("\nYouTube transcription successful!")
        print(f"Transcript: {result.transcript}")
        # Use the remote client's Gemini integration for summary
        summary = client._generate_summary_youtube(result.transcript)
        if summary:
            print("\nSummary:")
            print(summary)
    else:
        print(f"YouTube transcription failed: {result.error}")