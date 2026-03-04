"""LangGraph technical blog writer with optional research and image enrichment.

Pipeline overview:
1) Route topic to decide if web research is needed.
2) Gather evidence (optional).
3) Plan blog sections.
4) Fan out section-writing workers.
5) Merge sections and enrich with images (Gemini + local fallback).
6) Write final markdown to output.md.
"""

from typing import Literal, TypedDict, Annotated, List, Optional
import operator
import os
import re
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langchain_community.tools import TavilySearchResults  # pyright: ignore[reportMissingImports]

load_dotenv()
# Absolute directory for all file I/O (images, markdown outputs).
SCRIPT_DIR = Path(__file__).resolve().parent

#=======================================================================
# Define all schemas

class Task(BaseModel):
    id: str
    title: str
    goal: str = Field(
        ..., 
        description="The goal of the task describing what reader should be able to understand")
    bullets: List[str] = Field(
        ..., 
        min_length=3,
        max_length=5,
        description="A list of 3 to 5 non-overlapping bullet points that should be covered in the section."
    )
    target_words: int = Field(..., description="The target word count for the section. 120-150 words.")
    tags: List[str] = Field(..., description="A list of tags associated with the section.")
    requires_research: bool = Field(..., description="Indicates whether the section requires additional research.")
    requires_citations: bool = Field(..., description="Indicates whether the section requires citations.")
    requires_code: bool = Field(..., description="Indicates whether the section requires code snippets.")
   
   
    """ section_type: Literal[
        "introduction", "body", "example","common_mistakes","conclusion"
        ] = Field(
        ...,
        description="The type of the section which can be one of the following used exactly once in  the plan : introduction, body, example, common_mistakes, conclusion."        
        )
    brief: str = Field(..., description="A brief description of the task.") """

class Plan(BaseModel):
    blog_title: str
    audience: str = Field(..., description="The target audience for the blog post.")
    tone: str = Field(..., description="The tone of the blog post, e.g., informative, casual, professional.")
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = Field(..., description="The kind of blog post, e.g., how-to, listicle, case-study, opinion.")
    constraints: List[str] = Field(..., description="A list of constraints to follow when writing the blog post.")
    tasks: List[Task]

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str]

class EvidenceItem(BaseModel):
    title: str
    url: str    
    published_date: Optional[str]
    snippet: Optional[str]
    source: Optional[str]

class EvidencePack(BaseModel):
    evidence:  List[EvidenceItem] = Field(default_factory=list, description="A list of evidence items relevant to the research query.")
    
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="A placeholder string in the section content that indicates where the image should be inserted.eg., [IMAGE_1]")
    file_name: str = Field(..., description="The file name of the image to be inserted.")
    alt: str = Field(..., description="The alt text for the image, used for accessibility and SEO.")
    caption: str = Field(..., description="The caption for the image.")
    prompt: str = Field(..., description="A text prompt describing the content of the image for generation purposes.")
    size: Literal["1024 *1024", "1024*1536", "1536*1024"] = Field(..., description="The size of the image to be generated, e.g., small (400x300), medium (800x600), large (1200x900).")
    quality: Literal["low", "medium", "high"] = Field(..., description="The quality level of the image, e.g., draft for quick generation, final for high-quality output.")

class GlobalImagePlan(BaseModel):
    md_with_placeholders: str = Field(..., description="The markdown content of the blog post with placeholders indicating where images should be inserted.")
    images: List[ImageSpec] = Field(default_factory=list)





#======================================================================

# Define the state of the agent
class State(TypedDict):
    """Shared workflow state passed between LangGraph nodes."""
    topic: str
    
    #routing/reserch
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    
    #images
    merged_md: Optional[str]
    md_with_placeholders: Optional[str]
    image_specs: List[dict]
    
    
    #workers
    sections: Annotated[List[tuple[int,str]], operator.add]
    final: str
#========================================================================

#defin LLM
    
llm = ChatOpenAI(model="gpt-4.1-mini")
_tavily_client = TavilyClient()


#========================================================================

#define Router

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

    Decide whether web research is needed BEFORE planning.

    Modes:
    - closed_book (needs_research=false):
    Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
    - hybrid (needs_research=true):
    Mostly evergreen but needs up-to-date examples/tools/models to be useful.
    - open_book (needs_research=true):
    Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

    If needs_research=true:
    - Output 3–10 high-signal queries.
    - Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
    - If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
    """
    
def router_node(state: State) -> dict:
    """Classify the topic and decide if research should run before planning."""
    topic = state['topic']
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"User topic: {topic}")
    ])
    
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,  
        "queries": decision.queries}

def route_next(state: State) -> str:
    """Route into research branch only when router marks it as needed."""
    return "research" if state['needs_research'] else "orchestrator"
    

#========================================================================

#research node (Tavily Search)
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """Run Tavily search and normalize result keys for downstream models."""
    
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    """Collect and deduplicate evidence for planner/worker grounding."""
    queries = state['queries']
    max_results = 6
    raw_results: List[dict] = []
   
    for query in queries:
        raw_results.extend(_tavily_search(query=query, max_results=max_results))
    
    if not raw_results:
        return {"evidence": []}
    
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw search results: {raw_results}")
    ])
    
    return {"evidence": pack.evidence}
    
#========================================================================

#define orchestrator/planner

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    """Build a section-level plan from topic + optional evidence."""
    planner = llm.with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )

    return {"plan": plan}

#========================================================================

#define fanout 
def fanout(state: State):
    """Create one worker task payload per planned section."""
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

#========================================================================

#define worker
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def worker_node(payload: dict) -> dict:
    """Write one markdown section according to the planned task contract."""
    
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_date or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

#========================================================================
# REDUCER -SUBGRAPH with image handling (placeholder replacement and image generation)

#  merge_content -> decide_images -> generate_and_place_images

def merge_content(state: State) -> dict:
    """Combine ordered worker sections into a single markdown draft."""

    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    """Plan image placeholders and prompts; fallback safely on planner timeout/failure."""
    merged_md = state.get("merged_md") or ""
    topic = state.get("topic", "topic")

    def _fallback(reason: str) -> dict:
        # Always return 3 deterministic placeholders so downstream replacement can proceed.
        print(f"[DEBUG] decide_images fallback: {reason}")
        safe_topic = "".join(ch.lower() if ch.isalnum() else "_" for ch in topic).strip("_") or "topic"

        md = merged_md
        specs = []
        for i in range(1, 4):
            ph = f"[[IMAGE_{i}]]"
            if ph not in md:
                md += f"\n\n{ph}\n"
            specs.append(
                {
                    "placeholder": ph,
                    "prompt": f"Create a clear technical diagram about {topic}, figure {i}.",
                    "alt": f"{topic} diagram {i}",
                    "caption": f"Fallback diagram {i} for {topic}.",
                    "file_name": f"{safe_topic}_{i}.png",
                }
            )
        return {"md_with_placeholders": md, "image_specs": specs}

    planner = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        request_timeout=30,
        max_retries=1,
    ).with_structured_output(GlobalImagePlan)

    try:
        image_plan = planner.invoke(
            [
                SystemMessage(
                    content=(
                        "You plan images for a technical markdown blog. "
                        "Return markdown with placeholders like [[IMAGE_1]] and an images list."
                    )
                ),
                HumanMessage(content=f"Topic: {topic}\n\nMarkdown:\n{merged_md}"),
            ]
        )
    except KeyboardInterrupt:
        return _fallback("KeyboardInterrupt")
    except Exception as e:
        return _fallback(str(e))

    if not getattr(image_plan, "images", None):
        return _fallback("no images returned by planner")

    specs = [
        {
            "placeholder": img.placeholder,
            "prompt": img.prompt,
            "alt": img.alt,
            "caption": img.caption,
            "file_name": img.file_name,
        }
        for img in image_plan.images
    ]

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": specs,
    }
    
def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Generate image bytes from Gemini using robust response parsing.

    Notes:
    - SDK responses can vary by version, so we check multiple shapes.
    - Raises RuntimeError if no image bytes are found.
    """
    import google.genai as genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
        ),
    )

    # SDK shape 1: candidates -> content -> parts -> inline_data.data
    for cand in (getattr(resp, "candidates", None) or []):
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None)
            if data:
                return data

    # SDK shape 2: resp.contents[*].image_bytes
    for item in (getattr(resp, "contents", None) or []):
        data = getattr(item, "image_bytes", None)
        if data:
            return data

    raise RuntimeError(f"No image bytes in Gemini response: {resp}")

def _is_quota_error(exc: Exception) -> bool:
    """Best-effort detector for quota/rate-limit failures."""
    msg = str(exc).upper()
    return (
        "RESOURCE_EXHAUSTED" in msg
        or ("429" in msg and "QUOTA" in msg)
        or "RATE LIMIT" in msg
    )


def _write_local_svg_diagram(prompt: str, out_path: Path, title: str) -> Path:
    """Write a simple local SVG fallback diagram when remote image generation fails."""
    import html
    import textwrap

    svg_path = out_path.with_suffix(".svg")
    lines = textwrap.wrap((prompt or "No prompt provided").strip(), width=62)[:10] or ["No prompt provided"]

    tspans = []
    for i, line in enumerate(lines):
        dy = "0" if i == 0 else "1.35em"
        tspans.append(f'<tspan x="48" dy="{dy}">{html.escape(line)}</tspan>')

    height = 220 + max(0, len(lines) - 1) * 26
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">'
        '<rect x="0" y="0" width="1200" height="100%" fill="#0f172a"/>'
        f'<rect x="24" y="24" width="1152" height="{height - 48}" rx="14" fill="#111827" stroke="#334155" stroke-width="2"/>'
        f'<text x="48" y="76" fill="#e5e7eb" font-family="Segoe UI, Arial" font-size="28" font-weight="700">{html.escape(title)}</text>'
        f'<text x="48" y="124" fill="#cbd5e1" font-family="Consolas, Arial" font-size="20">{"".join(tspans)}</text>'
        "</svg>"
    )
    svg_path.write_text(svg, encoding="utf-8")
    return svg_path

# ...existing code...
def generate_and_place_images(state: State) -> dict:
    """Generate/resolve images and replace placeholders in markdown.

    Behavior:
    - Try Gemini image generation first.
    - If quota fails once, switch remaining images to local SVG fallback.
    - Always write final markdown to output.md.
    """
    print("[DEBUG] generate_and_place_images called")

    md = state.get("md_with_placeholders") or state.get("merged_md") or ""
    image_specs = state.get("image_specs", []) or []
    print(f"[DEBUG] image_specs count: {len(image_specs)}")

    images_dir = SCRIPT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    quota_exhausted = False

    for i, spec in enumerate(image_specs, start=1):
        placeholder = spec.get("placeholder", "")
        filename = spec.get("file_name") or spec.get("filename")
        if not filename:
            print(f"[DEBUG] missing filename in spec #{i}: {spec}")
            continue

        out_path = images_dir / filename
        print(f"[DEBUG] image #{i}: {out_path}")

        use_path: Path

        if quota_exhausted:
            # Fast path after first quota failure: avoid repeated remote calls.
            use_path = _write_local_svg_diagram(
                prompt=spec.get("prompt", ""),
                out_path=out_path,
                title=spec.get("alt", "Local fallback diagram"),
            )
            print(f"[DEBUG] fallback diagram written: {use_path}")
        else:
            try:
                if out_path.exists():
                    use_path = out_path
                else:
                    img_bytes = _gemini_generate_image_bytes(spec.get("prompt", ""))
                    out_path.write_bytes(img_bytes)
                    print(f"[DEBUG] wrote {out_path} ({len(img_bytes)} bytes)")
                    use_path = out_path
            except Exception as e:
                print(f"[DEBUG] FAILED {filename}: {e}")
                if _is_quota_error(e):
                    quota_exhausted = True
                    print("[DEBUG] Quota exhausted. Switching remaining images to local fallback.")

                # For non-quota failures we still produce a local diagram to avoid empty placeholders.
                use_path = _write_local_svg_diagram(
                    prompt=spec.get("prompt", ""),
                    out_path=out_path,
                    title=spec.get("alt", "Local fallback diagram"),
                )
                print(f"[DEBUG] fallback diagram written: {use_path}")

        rel = use_path.relative_to(SCRIPT_DIR).as_posix()
        caption = spec.get("caption", "")
        if use_path.suffix.lower() == ".svg":
            caption = (caption + " (local fallback)").strip()

        img_md = f"![{spec.get('alt', 'Generated image')}]({rel})\n*{caption}*"

        if placeholder and placeholder in md:
            md = md.replace(placeholder, img_md)
        else:
            md += f"\n\n{img_md}\n"

    (SCRIPT_DIR / "output.md").write_text(md, encoding="utf-8")
    return {"final": md}

#=================================================================
# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)


reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

#=================================================================
#define reducer
def reducer_node(state: State) -> dict:
    """Legacy reducer helper: writes merged markdown using blog title as filename.

    Note: the compiled reducer subgraph is wired into the graph, not this function.
    Kept for reference/backward compatibility.
    """

    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    filename = f"{plan.blog_title}.md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final": final_md}

#========================================================================


# Define the state graph
graph = StateGraph(State)

#======================================================================

#Define the nodes of the graph
graph.add_node('router', router_node)
graph.add_node('research', research_node)
graph.add_node('orchestrator', orchestrator_node)
graph.add_node('worker', worker_node)
graph.add_node('reducer', reducer_subgraph)

#======================================================================

#add edges between the nodes
graph.add_edge(START, 'router')
graph.add_conditional_edges('router', route_next, {'research': 'research', 'orchestrator': 'orchestrator'})
graph.add_edge('research', 'orchestrator')
graph.add_conditional_edges('orchestrator', fanout, ['worker'])
graph.add_edge('worker', 'reducer')
graph.add_edge('reducer', END)

#=====================================================================

#compile the graph
workflow = graph.compile()
print(' Workflow compiled successfully!')

#=====================================================================

# Runner fundtion
def run(topic: str):
    """Run the full graph once for a topic and return final state."""
    out = workflow.invoke(
        {
            "topic": topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "sections": [],
            "final": "",
        }
    )

    return out

#=====================================================================

#execute the graph with an initial state
invoke_result = run("Self Attention in Transformer Architecture")

