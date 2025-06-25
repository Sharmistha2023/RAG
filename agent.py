from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama

from langchain.agents import AgentOutputParser
from langchain.agents.agent import OutputParserException
import re

# class SafeOutputParser(AgentOutputParser):
#     def parse(self, text: str):
#         # Fix invalid tool name formats like HelloTool()
#         cleaned_text = re.sub(r"HelloTool\(\)", "HelloTool", text)
#         try:
#             # Use the default MRKL parser on the cleaned text
#             from langchain.agents.mrkl.output_parser import MRKLOutputParser
#             return MRKLOutputParser().parse(cleaned_text)
#         except OutputParserException as e:
#             # Raise the same error if cleaning didn't help
#             raise OutputParserException(f"Fixed format still failed:\n{text}")



# Create a very simple text generator using GPT2 (runs locally)
generator = pipeline("text-generation", model="gpt2", max_new_tokens=20)
#llm = HuggingFacePipeline(pipeline=generator)

llm = Ollama(model="mistral")

def hello_tool_func(_):
    return "Hello, world!"
import subprocess

def run_d3x_help(_=None):
    try:
        result = subprocess.run(["d3x", "--help"], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"
    except FileNotFoundError:
        return "Error: 'd3x' command not found. Is it installed?"


def run_emb_list(_=None):
    try:
        result = subprocess.run(
            ["d3x", "emb", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error while running 'd3x emb list':\n{e.stderr}"
    except FileNotFoundError:
        return "Error: 'd3x' command not found. Is it installed and on PATH?"



hello_tool = Tool(
    name="HelloTool()",
    func=hello_tool_func,
    description="Returns 'Hello, world!' whenever used",
    handle_parsing_errors=True
)

d3x_tool = Tool(
    name="run_d3x_help()",
    func=run_d3x_help,
    description="Runs 'd3x --help' in the terminal and returns the output.",
    handle_parsing_errors=True
)



emb_list_tool = Tool(
    name="run_emb_list()",
    func=run_emb_list,
    description="Runs 'd3x emb list' in the terminal and returns the available embeddings list.",
    handle_parsing_errors=True
)


# Create the agent
# agent = initialize_agent(
#     tools=[hello_tool],
#     llm=llm,
#     agent="zero-shot-react-description",
#     verbose=True,
#     #output_parser=SafeOutputParser() 
# )

# Run the agent
# response = agent.run("Use the HelloTool to say hello.")
# print(response)


# agent = initialize_agent(
#     tools=[d3x_tool],
#     llm=llm,
#     #agent="ZERO_SHOT_REACT_DESCRIPTION",
#     verbose=True
# )

# agent.run("Can you show me what the d3x command does?")
# print(response)


agent = initialize_agent(
    tools=[emb_list_tool],
    llm=llm,
    #agent="ZERO_SHOT_REACT_DESCRIPTION",
    verbose=True
)

agent.run("Can you show me what the emb command does?")
