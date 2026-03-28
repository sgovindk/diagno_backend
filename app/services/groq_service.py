"""
Groq LLM service for generating responses.
Integrates with Groq API for llama model inference.
"""

from typing import Optional
from groq import Groq, AsyncGroq
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class GroqService:
    """
    Service for interacting with Groq LLM API.
    Provides async and sync methods for text generation.
    """
    
    SYSTEM_PROMPT = """You are an expert electrical diagnostic assistant. 
You have access to technical manuals and diagnostic procedures. 
Answer only using the provided context. If the information is not in the context, say "I don't have information about this in the available manuals."
Be concise, accurate, and helpful. Provide step-by-step procedures when applicable."""
    
    def __init__(self, api_key: str = settings.GROQ_API_KEY, model: str = settings.GROQ_MODEL):
        """
        Initialize Groq service.
        
        Args:
            api_key: Groq API key
            model: Model name to use (default: llama-3.1-8b-instant)
        """
        if not api_key:
            raise ValueError("GROQ_API_KEY is required")
        
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=api_key)
        self.async_client = AsyncGroq(api_key=api_key)
        logger.info(f"Groq service initialized with model: {model}")
    
    def generate_answer(
        self,
        query: str,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate an answer using Groq LLM (synchronous).
        
        Args:
            query: User question
            context: Retrieved context from FAISS
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated answer text
            
        Raises:
            RuntimeError: If API call fails
        """
        try:
            user_message = f"""Context:
{context}

Question: {query}"""

            message = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=settings.GROQ_TIMEOUT
            )
            
            answer = message.choices[0].message.content
            logger.debug(f"Generated answer for query: {query[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer from Groq: {str(e)}")
            raise RuntimeError(f"Failed to generate answer: {str(e)}")
    
    async def generate_answer_async(
        self,
        query: str,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Generate an answer using Groq LLM (asynchronous).
        
        Args:
            query: User question
            context: Retrieved context from FAISS
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated answer text
            
        Raises:
            RuntimeError: If API call fails
        """
        try:
            user_message = f"""Context:
{context}

Question: {query}"""

            message = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=settings.GROQ_TIMEOUT
            )
            
            answer = message.choices[0].message.content
            logger.debug(f"Generated answer for query: {query[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer from Groq: {str(e)}")
            raise RuntimeError(f"Failed to generate answer: {str(e)}")


# Global Groq service instance
_groq_service: Optional[GroqService] = None


def get_groq_service() -> GroqService:
    """
    Get or create the global Groq service instance.
    
    Returns:
        Groq service instance
        
    Raises:
        ValueError: If GROQ_API_KEY is not set
    """
    global _groq_service
    if _groq_service is None:
        _groq_service = GroqService()
    return _groq_service
