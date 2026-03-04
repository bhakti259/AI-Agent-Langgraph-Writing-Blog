# AI Agent (LangGraph) — Blog Writing

This repository contains two LangGraph-based blog generators:

1. **Basic writer** (`1_basic_blog_writing_agent.py`)  
   Plans sections and writes a Markdown blog.
2. **Research + image writer** (`2_research_blog_writing_agent.py`)  
   Adds routing, optional Tavily research, and image enrichment (Gemini + local SVG fallback).

## Project structure

- `1_basic_blog_writing_agent.py` — planner → workers → reducer flow
- `2_research_blog_writing_agent.py` — router → optional research → planner → workers → reducer subgraph with images
- `images/` — generated images (PNG or local fallback SVG)
- `output.md` — latest generated blog output from the research writer
- `requirements.txt` — project dependencies
- `.env.example` — environment variable template

## Requirements

- Python **3.11+** (Python 3.11/3.12 recommended for best compatibility)
- OpenAI API key (`OPENAI_API_KEY`)
- Tavily API key (`TAVILY_API_KEY`) for research mode
- Google AI API key (`GOOGLE_API_KEY`) for Gemini image generation (optional if fallback images are acceptable)

## Setup

### 1) Create and activate a virtual environment

#### Windows (cmd)

```bat
python -m venv .venv
call .venv\Scripts\activate.bat
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

- Copy `.env.example` to `.env`
- Fill in keys

Example:

```dotenv
OPENAI_API_KEY="your_openai_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"
GOOGLE_API_KEY="your_google_ai_api_key_here"

# Optional observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
LANGCHAIN_PROJECT="langgraph-blog-agent"
```

## Run

### Basic writer

```powershell
& ".venv/Scripts/python.exe" "1_basic_blog_writing_agent.py"
```

### Research + image writer

```powershell
& ".venv/Scripts/python.exe" "2_research_blog_writing_agent.py"
```

## Research writer workflow

`START → router → (research or orchestrator) → worker fanout → reducer-subgraph → END`

Reducer subgraph:

`merge_content → decide_images → generate_and_place_images`

- **Router** decides `closed_book`, `hybrid`, or `open_book`
- **Research node** gathers evidence via Tavily when needed
- **Orchestrator** builds a structured section plan
- **Workers** write section markdown
- **Image planner** inserts placeholders like `[[IMAGE_1]]`
- **Image generator** tries Gemini first, then falls back to local SVG diagrams when needed

## Output behavior

- `output.md` is always written by the research workflow.
- Images are written into `images/`.
- Placeholder tokens are replaced with Markdown image links.

### Fallback behavior

- If image planning times out/fails, deterministic placeholders/specs are generated.
- If Gemini image generation fails (quota, timeout, API issue), local SVG fallback diagrams are created so the blog still contains visual assets.

## Notes

- `.env` is git-ignored to protect secrets.
- Python 3.14 currently shows compatibility warnings in some dependencies; Python 3.11/3.12 is recommended for a quieter run.
