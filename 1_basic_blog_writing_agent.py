from typing import Literal, TypedDict, Annotated, List
import operator
import re
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

load_dotenv()
SCRIPT_DIR = Path(__file__).resolve().parent

class Task(BaseModel):
    id: str
    title: str
    goal: str = Field(
        ..., 
        description="The goal of the task describing what reader shpuld be able to understand")
    bullets: List[str] = Field(
        ..., 
        min_length=3,
        max_length=5,
        description="A list of 3 to 5 non-overlapping bullet points that should be covered in the section."
    )
    section_type: Literal[
        "introduction", "body", "example","common_mistakes","conclusion"
        ] = Field(
        ...,
        description="The type of the section which can be one of the following used exactly once in  the plan : introduction, body, example, common_mistakes, conclusion."        
        )
    target_words: int = Field(..., description="The target word count for the section. 120-150 words.")
    brief: str = Field(..., description="A brief description of the task.")

class Plan(BaseModel):
    blog_title: str
    audience: str = Field(..., description="The target audience for the blog post.")
    tone: str = Field(..., description="The tone of the blog post, e.g., informative, casual, professional.")
    tasks: List[Task]

# Define the state of the agent
class State(TypedDict):
    topic: str
    plan: Plan
    #reducer: str
    sections: Annotated[List[str], operator.add]
    final: str
    
llm = ChatOpenAI(model="gpt-4.1-mini")

def orchestrator(state: State) -> dict:
    plan = llm.with_structured_output(Plan).invoke(
            [
            SystemMessage(
                content=(
                    "You are a senior technical writer and developer advocate. Your job is to produce a "
                    "highly actionable outline for a technical blog post.\n\n"
                    "Hard requirements:\n"
                    "- Create 5–7 sections (tasks) that fit a technical blog.\n"
                    "- Each section must include:\n"
                    "  1) goal (1 sentence: what the reader can do/understand after the section)\n"
                    "  2) 3–5 bullets that are concrete, specific, and non-overlapping\n"
                    "  3) target word count (120–450)\n"
                    "- Include EXACTLY ONE section with section_type='common_mistakes'.\n\n"
                    "Make it technical (not generic):\n"
                    "- Assume the reader is a developer; use correct terminology.\n"
                    "- Prefer design/engineering structure: problem → intuition → approach → implementation → "
                    "trade-offs → testing/observability → conclusion.\n"
                    "- Bullets must be actionable and testable (e.g., 'Show a minimal code snippet for X', "
                    "'Explain why Y fails under Z condition', 'Add a checklist for production readiness').\n"
                    "- Explicitly include at least ONE of the following somewhere in the plan (as bullets):\n"
                    "  * a minimal working example (MWE) or code sketch\n"
                    "  * edge cases / failure modes\n"
                    "  * performance/cost considerations\n"
                    "  * security/privacy considerations (if relevant)\n"
                    "  * debugging tips / observability (logs, metrics, traces)\n"
                    "- Avoid vague bullets like 'Explain X' or 'Discuss Y'. Every bullet should state what "
                    "to build/compare/measure/verify.\n\n"
                    "Ordering guidance:\n"
                    "- Start with a crisp intro and problem framing.\n"
                    "- Build core concepts before advanced details.\n"
                    "- Include one section for common mistakes and how to avoid them.\n"
                    "- End with a practical summary/checklist and next steps.\n\n"
                    "Output must strictly match the Plan schema."
                )
            ),
            HumanMessage(content=f"Topic: {state['topic']}"),
        ]
    )    
    return {"plan": plan}

def fanout(state: State) -> List[str]:
    return [Send("worker", {"task": task, "topic": state['topic'], "plan": state['plan']}) 
            for task in state['plan'].tasks]

def worker(payload: dict) -> dict:
    
    task = payload['task']
    topic = payload['topic']
    plan = payload['plan']
    bullets_text = "\n- " + "\n- ".join(task.bullets)

    
    blog_title = plan.blog_title
    
    section_content = llm.invoke(
         [
            SystemMessage(
                content=(
                    "You are a senior technical writer and developer advocate. Write ONE section of a technical blog post in Markdown.\n\n"
                    "Hard constraints:\n"
                    "- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).\n"
                    "- Stay close to the Target words (±15%).\n"
                    "- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).\n\n"
                    "Technical quality bar:\n"
                    "- Be precise and implementation-oriented (developers should be able to apply it).\n"
                    "- Prefer concrete details over abstractions: APIs, data structures, protocols, and exact terms.\n"
                    "- When relevant, include at least one of:\n"
                    "  * a small code snippet (minimal, correct, and idiomatic)\n"
                    "  * a tiny example input/output\n"
                    "  * a checklist of steps\n"
                    "  * a diagram described in text (e.g., 'Flow: A -> B -> C')\n"
                    "- Explain trade-offs briefly (performance, cost, complexity, reliability).\n"
                    "- Call out edge cases / failure modes and what to do about them.\n"
                    "- If you mention a best practice, add the 'why' in one sentence.\n\n"
                    "Markdown style:\n"
                    "- Start with a '## <Section Title>' heading.\n"
                    "- Use short paragraphs, bullet lists where helpful, and code fences for code.\n"
                    "- Avoid fluff. Avoid marketing language.\n"
                    "- If you include code, keep it focused on the bullet being addressed.\n"
                )
            ),
            HumanMessage(
                content=(
                    f"Blog: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Topic: {topic}\n\n"
                    f"Section: {task.title}\n"
                    f"Section type: {task.section_type}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Bullets:{bullets_text}\n"
                )
            ),
        ]
    ).content.strip()
    
    return {"sections": [section_content]}

def reducer(state: State) -> dict:
    title = state['plan'].blog_title
    body = "\n\n".join(s for s in state['sections'] if s and s.strip())
    final_md = f"# {title}\n\n{body}"
    
    #SAVE final_md to a file
    safe_title = title.lower().strip()
    safe_title = re.sub(r"[<>:\"/\\|?*]", "_", safe_title)  # Windows-invalid chars
    safe_title = re.sub(r"\s+", "_", safe_title)              # whitespace -> underscore
    safe_title = re.sub(r"_+", "_", safe_title).strip("._ ")  # cleanup
    if not safe_title:
        safe_title = "generated_blog"
    file_name = f"{safe_title}.md"
    output_path = SCRIPT_DIR / file_name

    if not final_md.strip():
        raise ValueError("Generated markdown content is empty; nothing to write.")

    chars_written = output_path.write_text(final_md, encoding='utf-8')
    print(f"Blog written to: {output_path}")
    print(f"Characters written: {chars_written}")
    
    return {'final': final_md}


# Define the state graph
graph = StateGraph(State)

#Define the nodes of the graph
graph.add_node('orchestrator', orchestrator)
graph.add_node('worker', worker)
graph.add_node('reducer', reducer)

#add edges between the nodes
graph.add_edge(START, 'orchestrator')
graph.add_conditional_edges('orchestrator', fanout, ['worker'])
graph.add_edge('worker', 'reducer')
graph.add_edge('reducer', END)

#compile the graph
workflow = graph.compile()
print(' Workflow compiled successfully!')

#execute the graph with an initial state
invoke_result = workflow.invoke({
    'topic': 'Write a blog post about the benefits of using AI in education.',
    'sections': []})
print(f"Workflow execution result: {invoke_result}")

