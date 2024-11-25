from django.db import models

class Transcription(models.Model):
    TRANSCRIPTION_TYPES = (
        ('youtube', 'YouTube'),
        ('whisper_local', 'Whisper Local'),
        ('whisper_remote', 'Whisper Remote'),
    )
    
    source_id = models.CharField(max_length=255)  # URL or video ID
    transcription_type = models.CharField(max_length=20, choices=TRANSCRIPTION_TYPES)
    transcript = models.TextField(blank=True)
    summary = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    success = models.BooleanField(default=False)
    error = models.TextField(blank=True)

class TimedWord(models.Model):
    transcription = models.ForeignKey(Transcription, related_name='timed_words', on_delete=models.CASCADE)
    word = models.CharField(max_length=255)
    start = models.FloatField()
    end = models.FloatField()
    confidence = models.FloatField()
