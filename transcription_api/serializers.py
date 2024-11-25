from rest_framework import serializers
from .models import Transcription, TimedWord

class TimedWordSerializer(serializers.ModelSerializer):
    class Meta:
        model = TimedWord
        fields = ['word', 'start', 'end', 'confidence']

class TranscriptionSerializer(serializers.ModelSerializer):
    timed_words = TimedWordSerializer(many=True, read_only=True)
    
    class Meta:
        model = Transcription
        fields = ['id', 'source_id', 'transcription_type', 'transcript', 
                 'summary', 'created_at', 'success', 'error', 'timed_words']
