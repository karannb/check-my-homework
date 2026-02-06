# check-my-homework

A simple python script which calls LLMs on every page sequentially of your homework PDFs to check for mistakes.
Mostly vibe-coded.

## Usage
```bash
uv sync
source .venv/bin/activate
python check_my_homework.py --pdf_path path/to/your/homework.pdf --output_path path/to/feedback.txt
```
The run will also generate a [pdf_name]_feedback directory with per-page feedback files.