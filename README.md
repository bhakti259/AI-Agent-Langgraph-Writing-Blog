# AI Agent (LangGraph) - Writing Blog

A basic **LangGraph-powered blog writing agent** that:

1. Creates a blog plan from a topic
2. Fans out section-writing tasks
3. Reduces sections into a final markdown blog file

Current script: `1_basic_blog_writing_agent.py`

## Project structure

- `1_basic_blog_writing_agent.py` - main LangGraph workflow
- `requirements.txt` - Python dependencies
- `.env.example` - environment variable template
- `the_benefits_of_using_ai_in_education.md` - sample generated output

## Requirements

- Python 3.11+ (3.11/3.12 recommended for best LangChain compatibility)
- OpenAI API key

## Setup

1. Create and activate virtual environment

### Windows (cmd)

```bat
python -m venv .venv
call .venv\Scripts\activate.bat
```

### Windows (PowerShell)

```powershell
python -m venv .venv
. .venv/Scripts/Activate.ps1
```

1. Install dependencies

```bash
pip install -r requirements.txt
```

1. Configure environment variables

- Copy `.env.example` to `.env`
- Fill in your keys (especially `OPENAI_API_KEY`)

Example:

```dotenv
OPENAI_API_KEY="your_openai_api_key_here"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langsmith_api_key_here"
LANGCHAIN_PROJECT="langgraph-chatbot"
```

## Run

### cmd

```bat
"D:\Python Projects\Langgraph_AI_Agent_Writing_Blog\.venv\Scripts\python.exe" "D:\Python Projects\Langgraph_AI_Agent_Writing_Blog\1_basic_blog_writing_agent.py"
```

### PowerShell

```powershell
& "D:/Python Projects/Langgraph_AI_Agent_Writing_Blog/.venv/Scripts/python.exe" "D:/Python Projects/Langgraph_AI_Agent_Writing_Blog/1_basic_blog_writing_agent.py"
```

## Output

The script writes a markdown blog file in the project folder, named from the generated blog title (for example: `the_benefits_of_using_ai_in_education.md`).

## Notes

- `.env` is ignored by git to protect secrets.
- If you see warnings about Python 3.14 compatibility, consider using Python 3.11/3.12.
