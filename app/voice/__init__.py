"""Local voice assistant for the Space Administrator Agent.

Everything runs on this machine: speech detection (Silero VAD), speech-to-text (NVIDIA Parakeet
via onnx-asr), the language model (Ollama) and text-to-speech (Kokoro). No audio or text leaves
the computer. See app/voice/conversation.py for the LangGraph conversation.
"""
