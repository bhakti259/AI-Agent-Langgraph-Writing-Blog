from typing import TypedDict, Annotated, List
import operator
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

load_dotenv()

class Task(BaseModel):
    id: str
    title: str
    brief: str = Field(..., description="A brief description of the task.")

class Plan(BaseModel):
    blog_title: str
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
            SystemMessage(content="You are a helpful assistant that creates a plan for writing a blog post.Create a blog plan with 5-7 sections on following topic."),
            HumanMessage(content=f"Topic: {state['topic']}")
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
    
    blog_title = plan.blog_title
    
    section_content = llm.invoke(
        [
            SystemMessage(content=f"You are a helpful assistant that writes a section of a blog post. Write clean section"),
            HumanMessage(content=f"Blog Title: {blog_title}\n"
                         f"Section Title: {task.title}\n"
                         f"Brief: {task.brief}\n"
                         f"Topic: {topic}"
                         "return only the content of the section without any additional text or formatting.")
        ]
    ).content.strip()
    
    return {"sections": [section_content]}

def reducer(state: State) -> dict:
    title = state['plan'].blog_title
    body = "\n\n".join(state['sections'])   
    final_md = f"# {title}\n\n{body}"
    
    #SAVE final_md to a file
    file_name = title.lower().replace(" ", "_") + ".md"
    output_path = Path(file_name)
    output_path.write_text(final_md, encoding='utf-8')    
    
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
initial_state = State()
initial_state['topic'] = "Write a blog post about the benefits of using AI in education."
invoke_result = workflow.invoke(initial_state)
print(f"Workflow execution result: {invoke_result}")
