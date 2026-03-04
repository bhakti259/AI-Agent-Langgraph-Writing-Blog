# LangGraph Blog Writing Agent - Architecture Diagram

This diagram represents the end-to-end architecture of the project, including entry points, LangGraph pipeline nodes, external services, and generated artifacts.

```mermaid
flowchart LR
    U[User]

    subgraph Interface[Interface Layer]
      ST[Streamlit Frontend\n3_research_blog_writing_agent_frontend.py]
      CLI1[CLI Basic\n1_basic_blog_writing_agent.py]
      CLI2[CLI Research\n2_research_blog_writing_agent.py]
    end

    subgraph Core[LangGraph Backend\nresearch_blog_wriging_agent_backend.py]
      R[Router Node]
      RE[Research Node]
      O[Orchestrator Node]
      W[Worker Fanout\nN Section Workers]
      M[Merge Content]
      DI[Decide Images]
      GI[Generate and Place Images]
    end

    subgraph External[External Services]
      OAI[OpenAI Chat Models]
      TV[Tavily Search API]
      GEM[Gemini Image API]
    end

    subgraph Outputs[Artifacts]
      MD[output.md]
      IMG[images/* PNG or SVG]
      BUNDLE[Markdown + images ZIP]
    end

    U --> ST
    U --> CLI1
    U --> CLI2

    ST -->|invoke app| R
    CLI2 -->|run topic| R
    CLI1 -->|basic planner flow| O

    R -->|closed_book| O
    R -->|hybrid/open_book| RE
    RE --> O

    O --> W
    W --> M
    M --> DI
    DI --> GI

    GI --> MD
    GI --> IMG
    ST --> BUNDLE

    R -.routing decisions.-> OAI
    O -.structured planning.-> OAI
    W -.section drafting.-> OAI
    DI -.image spec planning.-> OAI

    RE --> TV
    GI --> GEM
    GI -->|fallback if API fails| IMG
```

## Notes

- `closed_book` skips web research and moves directly to planning.
- `hybrid/open_book` uses Tavily evidence before planning.
- Image generation first attempts Gemini and falls back to local SVG diagrams when needed.
- Final markdown is written to `output.md`, with images in `images/`.
