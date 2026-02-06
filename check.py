"""
Run this script to check your homework PDFs for mistakes.
It will call LLMs on every page sequentially and provide feedback on any errors found.
Make sure to have the necessary libraries installed and your API keys set up for the LLMs you intend to use.

Usage:
python3 check.py --pdf_path /path/to/your/homework.pdf --model gemini-2.5-flash-preview-05-20 --output_path /path/to/output/feedback.txt

Requirements:
    pip install pdf2image google-genai pillow

    Also requires poppler for pdf2image:
    - macOS: brew install poppler
    - Ubuntu: sudo apt-get install poppler-utils
    - Windows: Download from https://github.com/osber/poppler-windows/releases

Environment Variables:
    GEMINI_API_KEY or GOOGLE_API_KEY: Your Google Gemini API key
"""

from __future__ import annotations

import os
import sys
import io
import time
import argparse
from pdf2image import convert_from_path

from llm import Agent, FeedbackValidationError, DEFAULT_MODEL

# Default delay between API calls (seconds) - for free tier rate limiting
DEFAULT_DELAY: int = 20

def extract_images_from_pdf(pdf_path: str, dpi: int = 200) -> list[bytes]:
    """
    Convert each page of a PDF to an image.
    
    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (default: 200 for good quality/size balance).
    
    Returns:
        List of PNG image bytes, one per page.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Convert PDF pages to PIL images
    images = convert_from_path(pdf_path, dpi=dpi)
    
    # Convert each PIL image to PNG bytes
    page_images: list[bytes] = []
    for img in images:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        page_images.append(buffer.getvalue())
    
    return page_images


def save_page_feedback(output_dir: str, page_num: int, feedback: str) -> str:
    """
    Save feedback for a single page to its own file.
    
    Args:
        output_dir: Directory to save feedback files.
        page_num: Page number (1-indexed).
        feedback: The feedback text to save.
    
    Returns:
        The filepath where the feedback was saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath: str = os.path.join(output_dir, f"page_{page_num:03d}.txt")
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"=== Page {page_num} Feedback ===\n\n")
        f.write(feedback)
    
    return filepath


def check_homework(
    pdf_path: str,
    model: str | None = None,
    output_path: str | None = None,
    output_dir: str | None = None,
    validate: bool = True,
    verbose: bool = True,
    dpi: int = 200,
    delay: float = DEFAULT_DELAY,
) -> str:
    """
    Check a homework PDF for mistakes using an LLM.
    
    Args:
        pdf_path: Path to the homework PDF.
        model: LLM model to use (default: Gemini Flash).
        output_path: Optional path to save combined feedback to a file.
        output_dir: Optional directory to save per-page feedback files.
        validate: Whether to validate LLM response format.
        verbose: Whether to print progress to console.
        dpi: Resolution for rendering PDF pages (default: 200).
        delay: Delay in seconds between API calls (default: 20 for free tier).
    
    Returns:
        The complete feedback from the LLM.
    """
    # Extract page images from PDF
    if verbose:
        print(f"📄 Reading PDF: {pdf_path}")
    
    page_images = extract_images_from_pdf(pdf_path, dpi=dpi)
    
    if verbose:
        print(f"📑 Found {len(page_images)} pages")
    
    # Set up output directory for per-page files
    if output_dir is None:
        # Default: create a directory next to the PDF named <pdf_name>_feedback
        pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.join(os.path.dirname(pdf_path) or ".", f"{pdf_basename}_feedback")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"📁 Saving per-page feedback to: {output_dir}")
    
    # Initialize the agent
    agent = Agent(model=model)
    
    if verbose:
        print(f"🤖 Using model: {agent.model}")
        print(f"⏱️  Delay between calls: {delay}s")
        print("-" * 50)
    
    all_feedback: list[str] = []
    
    # Process each page
    for i, page_image in enumerate(page_images, start=1):
        if verbose:
            print(f"📖 Processing page {i}/{len(page_images)}...")
        
        # Add page image to conversation
        agent.add_page(page_image, page_number=i)
        
        # Get feedback for this page
        try:
            feedback = agent.get_feedback(validate=validate)
            all_feedback.append(f"=== Page {i} Feedback ===\n{feedback}")
            
            # Save feedback immediately to per-page file
            filepath = save_page_feedback(output_dir, i, feedback)
            if verbose:
                print(f"   💾 Saved to {filepath}")
            
            if verbose:
                print(f"✅ Page {i} processed successfully")
        
        except FeedbackValidationError as e:
            warning_msg = f"⚠️  Page {i}: Validation warning - {e}"
            raw_feedback = agent.feedback_history[-1] if agent.feedback_history else 'No feedback received'
            all_feedback.append(f"=== Page {i} Feedback ===\n[VALIDATION WARNING: {e}]\n{raw_feedback}")
            
            # Still save the feedback even with validation warning
            filepath = save_page_feedback(output_dir, i, f"[VALIDATION WARNING: {e}]\n\n{raw_feedback}")
            if verbose:
                print(f"   💾 Saved to {filepath}")
                print(warning_msg)
        
        # Rate limiting delay (skip after last page)
        if i < len(page_images) and delay > 0:
            if verbose:
                print(f"   ⏳ Waiting {delay}s before next request...")
            time.sleep(delay)
    
    # Check for incomplete questions
    open_questions = agent.get_open_questions()
    if open_questions and verbose:
        print(f"⚠️  Questions still open at end of document: {open_questions}")
    
    # Combine all feedback
    complete_feedback = "\n\n".join(all_feedback)
    
    # Add summary header
    summary = f"""
{'=' * 60}
HOMEWORK FEEDBACK REPORT
{'=' * 60}
PDF: {pdf_path}
Model: {agent.model}
Total Pages: {len(page_images)}
Open Questions: {open_questions if open_questions else 'None'}
{'=' * 60}

{complete_feedback}

{'=' * 60}
END OF REPORT
{'=' * 60}
"""
    
    # Save to file if output path specified
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)
        
        if verbose:
            print(f"\n💾 Feedback saved to: {output_path}")
    
    if verbose:
        print("\n✨ Done!")
    
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check homework PDFs for mistakes using LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 check.py --pdf_path homework.pdf
    python3 check.py --pdf_path homework.pdf --model gpt-4o --output_path feedback.txt
    python3 check.py --pdf_path homework.pdf --no-validate
        """
    )
    
    parser.add_argument(
        "--pdf_path", "-p",
        required=True,
        help="Path to the homework PDF file."
    )
    
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})."
    )
    
    parser.add_argument(
        "--output_path", "-o",
        default=None,
        help="Path to save the feedback output (optional)."
    )
    
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable response format validation."
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output."
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for rendering PDF pages (default: 300). Higher = better quality but larger images."
    )
    
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay in seconds between API calls (default: {DEFAULT_DELAY}). Set to 0 if on paid tier."
    )
    
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save per-page feedback files (default: <pdf_name>_feedback/)."
    )
    
    args = parser.parse_args()
    
    try:
        feedback = check_homework(
            pdf_path=args.pdf_path,
            model=args.model,
            output_path=args.output_path,
            output_dir=args.output_dir,
            validate=not args.no_validate,
            verbose=not args.quiet,
            dpi=args.dpi,
            delay=args.delay
        )
        
        # Print feedback if no output file specified
        if not args.output_path:
            print("\n" + feedback)
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
