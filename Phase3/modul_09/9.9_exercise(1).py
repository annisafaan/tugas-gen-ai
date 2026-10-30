from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="qwen2.5:3b", temperature=0, api_key=os.environ["OPENAI_API_KEY"])

# Prompt 1: Ekstraksi fakta
facts_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract exactly 3 key facts from the text. Output them as a simple numbered list."),
    ("human", "{text}")
])

# Prompt 2: Generator kuis
quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", "Based on these facts, generate a quiz question for each fact. Output strictly as a JSON list of strings. Example: [\"Q1\", \"Q2\", \"Q3\"]"),
    ("human", "{facts}")
])

# Chain: Teks Mentah -> Ekstrak Fakta -> Buat Kuis -> Parse ke Python List
chain = (
    {"facts": facts_prompt | llm | StrOutputParser()}
    | quiz_prompt
    | llm
    | JsonOutputParser()
)

raw_text = "Python was created by Guido van Rossum in 1991. It emphasizes code readability with its use of significant indentation. Python is dynamically typed and garbage-collected."

questions_list = chain.invoke({"text": raw_text})
print(type(questions_list))
print(questions_list)