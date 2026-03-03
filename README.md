# AI Agent (LangGraph) — Blog Writing

This project contains two LangGraph-based blog generators:

1. **Basic writer** (`1_basic_blog_writing_agent.py`)  
   Plans sections and writes a complete Markdown blog.
2. **Research writer** (`2_research_blog_writing_agent.py`)  
   Adds routing + web research (Tavily) before planning and drafting sections.

## Project structure

- `1_basic_blog_writing_agent.py` — basic planner → workers → reducer flow
- `2_research_blog_writing_agent.py` — router → optional research → planner → workers → reducer
- `requirements.txt` — project dependencies
- `.env.example` — environment variable template
- `unlocking_learning_potential_the_benefits_of_using_ai_in_education.md` — sample generated output

## Requirements

- Python **3.11+** (Python 3.11/3.12 recommended for best ecosystem compatibility)
- OpenAI API key
- Tavily API key (required for research-enabled flow)

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
- Fill in required keys

Example:

```dotenv
OPENAI_API_KEY="your_openai_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"

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

### Research writer (uses Tavily)

```powershell
& ".venv/Scripts/python.exe" "2_research_blog_writing_agent.py"
```

## How the research writer works

The `2_research_blog_writing_agent.py` workflow runs this graph:

`START → router → (research or orchestrator) → worker fanout → reducer → END`

- **Router** decides `closed_book`, `hybrid`, or `open_book`
- **Research node** gathers evidence URLs/snippets via Tavily when needed
- **Orchestrator** creates a structured section plan
- **Worker nodes** draft section Markdown
- **Reducer** joins sections and writes final blog output

## Output

- A Markdown file is generated in the project folder.
- In the research script, filename is based on `plan.blog_title`.

Example generated file:

- `State of Multimodal LLMs in 2026.md`

## Notes

- `.env` is git-ignored to protect secrets.
- If you encounter Python 3.14 compatibility warnings from dependencies, prefer Python 3.11/3.12.
- You may see deprecation warnings for `TavilySearchResults`; the script still runs, but future cleanup can migrate to `langchain_tavily`.
