from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create a router and register our viewset
router = DefaultRouter()
router.register(r'transcriptions', views.TranscriptionViewSet, basename='transcription')

# The API URLs are determined automatically by the router
urlpatterns = [
    path('', include(router.urls)),
    # Test endpoints
    path('test/whisper-remote/', views.test_whisper_remote, name='test-whisper-remote'),
    path('test/whisper-local/', views.test_whisper_local, name='test-whisper-local'),
    path('test/youtube/', views.test_youtube, name='test-youtube'),
] 