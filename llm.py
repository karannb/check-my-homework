"""
Basic file containing the system prompt and handling for page breaks.
"""
from __future__ import annotations
import os
import re
from google import genai
from google.genai import types

# Default model - Gemini 2.5 Flash
DEFAULT_MODEL: str = "gemini-3-flash-preview"

SYSTEM_PROMPT = """
You are a strict grader that checks for mistakes in homework PDFs.
You will be given the text content of each page of a homework PDF sequentially.
Some questions / answers may span multiple pages, in this case, only give feedback after reading the last page of the question.
Your task is to identify any mistakes in the homework and provide feedback on how to correct them.

Do not generate new answers to the questions, only provide feedback on the existing content.
Your feedback should be concise enough to fit within three paragraphs (excluding the math), but do not hold back on even style or tone.
Be as critical as possible, but also provide constructive feedback on how to improve.

Formatting Guidelines:
- If some question has not been attempted by the student, do not attempt it, or provide feedback, just say "Question not attempted, no feedback provided."
- At the end, always score the answer in the range of 0-5, and provide a brief justification for the score.
0 being the lowest (when incorrect answer) going up to 5 (when the answer is perfect).
- When you start a question, always start with "START Question X", when you end a question, always end with "END Question X", where X is the question number.
- Some pages may end with an incomplete answer, in this case do not generate the END tag until you have read the entire answer across multiple pages, and only provide feedback after the END tag.
"""


class FeedbackValidationError(Exception):
    """Raised when the LLM response doesn't match expected format."""
    pass


class Agent:
    model: str
    client: genai.Client
    conversation_history: list[types.Content]
    feedback_history: list[str]
    open_questions: set[str]

    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEFAULT_MODEL
        
        # Initialize the Gemini client (uses GEMINI_API_KEY or GOOGLE_API_KEY env var)
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set")
        
        self.client = genai.Client(api_key=api_key)
        
        # Store conversation history as list of Content objects
        self.conversation_history = []
        self.feedback_history = []
        self.open_questions = set()
    
    def add_page(self, page_image: bytes, page_number: int | None = None) -> None:
        """
        Add a page's image to the conversation context.
        
        Args:
            page_image: The page image as bytes (PNG format).
            page_number: Optional page number for context.
        """
        # Build multimodal content parts
        parts: list[types.Part] = []
        
        if page_number:
            parts.append(types.Part.from_text(text=f"[Page {page_number}]"))
        
        # Add image as inline data
        parts.append(types.Part.from_bytes(data=page_image, mime_type="image/png"))
        
        # Add to conversation history
        self.conversation_history.append(types.Content(role="user", parts=parts))
    
    def get_feedback(self, validate: bool = True) -> str:
        """
        Get feedback from the LLM for the current conversation.
        
        Args:
            validate: If True, validates the response format matches system prompt guidelines.
        
        Returns:
            The feedback text from the LLM.
        
        Raises:
            FeedbackValidationError: If validation is enabled and the response format is invalid.
        """
        # Call Gemini API with conversation history
        response: types.GenerateContentResponse = self.client.models.generate_content(
            model=self.model,
            contents=self.conversation_history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            )
        )
        
        # Extract the actual text content from the response
        feedback_text: str = response.text
        
        # Store assistant response in conversation history
        self.conversation_history.append(types.Content(
            role="model",
            parts=[types.Part.from_text(text=feedback_text)]
        ))
        self.feedback_history.append(feedback_text)
        
        if validate:
            self._validate_response(feedback_text)
        
        return feedback_text

    def _validate_response(self, response: str) -> None:
        """
        Validate that the response follows the expected format from the system prompt.
        
        This validation is lenient to handle various page scenarios:
        - Instruction pages (no questions)
        - Question starting on this page (START but no END)
        - Question ending on this page (END but no START on this page)
        - Question spanning entire page (no START or END)
        - Complete question on one page (START and END)
        
        Only checks for score when a question is fully completed (has END marker).
        """
        # Find all START and END markers with their question numbers
        start_matches: list[str] = re.findall(r"START Question (\d+)", response, re.IGNORECASE)
        end_matches: list[str] = re.findall(r"END Question (\d+)", response, re.IGNORECASE)
        
        # Track opened and closed questions
        for q_num in start_matches:
            self.open_questions.add(q_num)
        
        # When a question ends, it should have a score in this response
        for q_num in end_matches:
            self.open_questions.discard(q_num)
        
        # Only validate score if we have completed questions (END markers present)
        if end_matches:
            score_pattern = r"\b([0-5])\s*/\s*5\b|\bscore[:\s]+([0-5])\b|\b([0-5])\s+out\s+of\s+5\b"
            if not re.search(score_pattern, response, re.IGNORECASE):
                raise FeedbackValidationError(
                    f"Response has END marker for question(s) {end_matches} but missing required score (0-5). "
                    "Expected a score like '3/5', 'Score: 3', or '3 out of 5'."
                )
    
    def get_all_feedback(self) -> str:
        """Return all collected feedback as a single string."""
        return "\n\n" + "=" * 50 + "\n\n".join(self.feedback_history)

    def reset(self) -> None:
        """Reset the conversation and feedback history."""
        self.conversation_history = []
        self.feedback_history = []
        self.open_questions = set()

    def get_open_questions(self) -> set[str]:
        """Return the set of questions that have started but not ended."""
        return self.open_questions.copy()