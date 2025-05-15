"""
Agent module for RAG Showdown.

This module provides a simple agent implementation with tool-calling capabilities.
"""

from .agent import Agent
from .tools import get_weather, get_news
from .prompts import tools_prompt

__all__ = ["Agent", "get_weather", "get_news", "tools_prompt"]
