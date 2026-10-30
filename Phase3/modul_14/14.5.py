from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ── DEFINISI PENDUKUNG DARI MODUL SEBELUMNYA AGAR TIDAK ERROR ───────────────
class TokenCostCalculator(BaseTool):
    name: str = "Token Cost Calculator"
    description: str = "Calculate USD cost."
    def _run(self, model: str, input_tokens: int, output_tokens: int) -> str:
        return json.dumps({"cost_usd": 0.01})

class ModelContextLookup(BaseTool):
    name: str = "Model Context Window Lookup"
    description: str = "Look up context window."
    def _run(self, model: str) -> str:
        return json.dumps({"context_window_tokens": 128000})

cost_tool = TokenCostCalculator()
context_tool = ModelContextLookup()

def make_tasks(researcher, writer, reviewer):
    t1 = Task(description="Research AI topics.", expected_output="Report", agent=researcher)
    t2 = Task(description="Write guide.", expected_output="Guide", agent=writer, context=[t1])
    t3 = Task(description="Review guide.", expected_output="Review", agent=reviewer, context=[t2])
    return t1, t2, t3
# ─────────────────────────────────────────────────────────────────────────────

# PENYESUAIAN LOKAL: Mengganti ChatOpenAI dengan string LiteLLM Ollama
llm = "ollama/qwen2.5:3b"

# Recreate agents with tools
researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate information on AI topics.",
    backstory="Senior AI research analyst with expertise in LLM systems.",
    llm=llm,
    tools=[context_tool], # can look up context windows
    verbose=False,
    max_iter=5,
)

writer = Agent(
    role="Technical Content Writer",
    goal="Write clear technical content for developers.",
    backstory="Technical writer with a software engineering background.",
    llm=llm,
    tools=[cost_tool], # can look up costs for examples
    verbose=False,
    max_iter=5,
)

reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate and publication-ready.",
    backstory="Meticulous editor who reviews technical AI content.",
    llm=llm,
    verbose=False,
    max_iter=5,
)

research_task, writing_task, review_task = make_tasks(researcher, writer, reviewer)

crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential, # tasks run in order; use Process.hierarchical for manager-led
    verbose=False,
    memory=False, # set True to enable crosstask memory (requires embeddings)
)

# Kick off the crew
result = crew.kickoff()

print("=== CREW OUTPUT ===")
print(result)

# Access individual task outputs
print("\n=== TASK OUTPUTS ===")
for task in crew.tasks:
    print(f"\n--- {task.agent.role} ---") 
    if task.output:
        print(str(task.output)[:200], "...")