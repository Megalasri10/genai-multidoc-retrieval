## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

Research papers contain valuable information about research objectives, methodologies, findings, and conclusions. However, manually reading and analyzing multiple research papers is time-consuming and requires considerable effort to identify important information.
The proposed system uses LlamaIndex and Generative AI to develop an intelligent research paper analysis assistant. It processes PDF research papers, extracts relevant information, creates vector and summary indexes, and uses an AI-powered Function Agent to answer user questions.
The system aims to automate research paper summarization, identify research objectives, explain methodologies, highlight important findings, and provide clear answers through natural language queries.

### DESIGN STEPS:
## STEP 1:
Load the research papers using LlamaIndex.
Create vector and summary tools for each paper.
## STEP 2:
Combine the document tools.
Initialize the OpenAI language model.
## STEP 3:
Create the FunctionCallingAgentWorker.
Create the AgentRunner.
## STEP 4:
Give natural-language queries to the agent.
Retrieve relevant information from the papers.
## STEP 5:
Generate concise answers using the retrieved information.
Test the agent with different queries.
## STEP 6:
Evaluate the responses based on accuracy, relevance, and conciseness.
### PROGRAM:
## Name : Megala M S
## Register number : 212225040230
```
from helper import get_openai_api_key

OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

from pathlib import Path
```
```
import os

print("Current working directory:", os.getcwd())

print("\nAvailable PDF files:")

for file in os.listdir():
    if file.endswith(".pdf"):
        print(file)
```
```
papers = [
    "Sample_Research_Papers_for_LlamaIndex.pdf",
    "MachineLearning-Lecture01.pdf",
    "Generative_AI_Study_Guide_10_Pages(1).pdf"
]
```
```
from pathlib import Path

for paper in papers:
    if Path(paper).exists():
        print("Found:", paper)
    else:
        print("Missing:", paper)
```
```
papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "loftq.pdf"
]
```
```
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}

for paper in papers:
    print(f"Getting tools for paper: {paper}")

    vector_tool, summary_tool = get_doc_tools(
        paper,
        Path(paper).stem
    )

    paper_to_tools_dict[paper] = [
        vector_tool,
        summary_tool
    ]

print("All research papers loaded successfully!")
```
```
initial_tools = [
    tool
    for paper in papers
    for tool in paper_to_tools_dict[paper]
]

print("Number of tools:", len(initial_tools))
```
```
async def ask_question(question):

    response = await agent.run(
        user_msg=question
    )

    print("\nQuestion:", question)
    print("\nAnswer:")
    print(str(response))
```
```
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="gpt-4-turbo",
    temperature=0
)

print("OpenAI model initialized successfully!")
print("Function calling supported:", llm.metadata.is_function_calling_model)
```
```
import importlib.metadata

print("LlamaIndex OpenAI:",
      importlib.metadata.version("llama-index-llms-openai"))

print("LlamaIndex Core:",
      importlib.metadata.version("llama-index-core"))
```
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

print("API key configured:", bool(OPENAI_API_KEY))
```
```
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
    api_base="http://jupyter-api-proxy.internal.dlai/rev-proxy/full_power_standard_openai_for_lama_index"
)

print("Model:", llm.metadata.model_name)
print("Function calling:", llm.metadata.is_function_calling_model)
```
```
papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "loftq.pdf"
]

paper_names = [
    "MetaGPT",
    "LongLoRA",
    "LoftQ"
]

print("Selected research papers:")
for paper in papers:
    print(paper)

```
```
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("API key configured:", bool(OPENAI_API_KEY))
```
```
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
    api_base="http://jupyter-api-proxy.internal.dlai/rev-proxy/full_power_standard_openai_for_lama_index"
)

embedding = embed_model.get_text_embedding("Hello world")

print("Embedding test successful!")
print("Embedding dimension:", len(embedding))
```
```
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

# Configure embedding model
Settings.embed_model = embed_model

# Configure language model
llm = OpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    api_base="http://jupyter-api-proxy.internal.dlai/rev-proxy/full_power_standard_openai_for_lama_index",
    temperature=0.1
)

Settings.llm = llm

print("LlamaIndex configured successfully!")
print("Embedding model:", type(Settings.embed_model).__name__)
print("Language model:", type(Settings.llm).__name__)
```
```
from pathlib import Path
from utils import get_doc_tools

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "loftq.pdf"
]

paper_names = [
    "MetaGPT",
    "LongLoRA",
    "LoftQ"
]

# Check whether all PDF files exist
for paper in papers:
    print(paper, ":", Path(paper).exists())
```
```
initial_tools = []

for paper, name in zip(papers, paper_names):
    print("\nProcessing paper:", name)

    vector_tool, summary_tool = get_doc_tools(paper, name)

    initial_tools.append(vector_tool)
    initial_tools.append(summary_tool)

print("\nTotal tools created:", len(initial_tools))
```
```
from llama_index.core.agent.workflow import FunctionAgent

# Create the multidocument retrieval agent
agent = FunctionAgent(
    tools=initial_tools,
    llm=llm,
    system_prompt=(
        "You are a research assistant. "
        "Use the provided research paper tools to answer questions. "
        "Compare information across papers when required."
    )
)

print("Multidocument Retrieval Agent created successfully!")
```
```
response = llm.complete("Say Hello in one sentence.")
```
```
try:
    response = llm.complete("Say Hello")
    print(response)

except Exception as e:
    print("Error type:", type(e).__name__)
    print("Status code:", getattr(e, "status_code", None))
    print("Error details:", str(e))

    if hasattr(e, "body"):
        print("API response body:", e.body)
```
```
from openai import OpenAI

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="http://jupyter-api-proxy.internal.dlai/rev-proxy/full_power_standard_openai_for_lama_index"
)

try:
    models = client.models.list()

    print("Available models:")
    for model in models.data:
        print(model.id)

except Exception as e:
    print("Model-list error:", type(e).__name__, str(e))
```
```
await ask_question(
    "What is the main objective of the MetaGPT research paper?"
)
```
```
await ask_question(
    "Compare the main contributions of MetaGPT, LongLoRA, and LoftQ."
)
```
```
from llama_index.core.agent.workflow import FunctionAgent

agent = FunctionAgent(
    tools=initial_tools,
    llm=llm,
    system_prompt=(
        "You are a research assistant. "
        "Answer questions using the provided research papers. "
        "Compare information across papers when required."
    ),
    streaming=False
)

print("Agent recreated successfully without streaming!")
```
```
async def ask_question(question):
    response = await agent.run(user_msg=question)

    print("\nQuestion:", question)
    print("\nAnswer:")
    print(str(response))
```
```
await ask_question(
    "Explain the research methodologies used in MetaGPT, LongLoRA, and LoftQ."
)
```

### OUTPUT:


<img width="1206" height="195" alt="Screenshot 2026-11-24 221628" src="https://github.com/user-attachments/assets/b756ce98-2a96-44a3-ba40-339b181892d5" />

<img width="1325" height="832" alt="Screenshot 2026-11-24 221607" src="https://github.com/user-attachments/assets/e2b8d5f3-92eb-47dd-b851-ab999a97d40a" />


<img width="1328" height="545" alt="Screenshot 2026-11-24 221541" src="https://github.com/user-attachments/assets/a6576df4-6fda-466d-9c82-dd89323056ac" />

### RESULT:
The multidocument retrieval agent was successfully implemented using LlamaIndex. It retrieved relevant information from the research papers and generated concise and relevant responses to different queries. The system demonstrated effective document retrieval and information synthesis.
