from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.agents import initialize_agent, Tool
from langchain.llms import Ollama

from langchain.agents import AgentOutputParser
from langchain.agents.agent import OutputParserException
import re
from langchain.chat_models import ChatOpenAI
#from llama_index.llms.openai_like import OpenAILike
import os
os.environ["CURL_CA_BUNDLE"] = ""
import urllib3
urllib3.disable_warnings()
import openai
openai.verify_ssl_certs = False

# Create a very simple text generator using GPT2 (runs locally)
generator = pipeline("text-generation", model="gpt2", max_new_tokens=20)
#llm = HuggingFacePipeline(pipeline=generator)

#llm = Ollama(model="mistral")
llm = ChatOpenAI(
    temperature=0,
    openai_api_base="http://a7d7dc9e697d843f780df5512f98a1df-adbef44e0ec59790.elb.us-east-2.amazonaws.com/deployment/sharmistha-choudhury/mistal/v1/",  # Replace with your URL
    openai_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoic2hhcm1pc3RoYS1jaG91ZGh1cnkiLCJ0eXBlIjoiYXBpIiwiaWQiOiIvZGVwbG95bWVudC9taXN0YWwvIn0.XFjzYbwGOZjxe52K31q47tpMbBurDvQOK-El28B_ubg",                 # Replace with your token
    model_name="mistralai/Mistral-7B-Instruct-v0.2",   
    verify_ssl=False # Or any supported model
)




import os
import shutil
import yaml

def update_input_dir_in_config(src_path, new_input_dir):
    try:
        # Step 1: Create .ingest dir if needed
        ingest_dir = os.path.join(os.getcwd(), ".ingest")
        os.makedirs(ingest_dir, exist_ok=True)

        # Step 2: Copy config.yaml to .ingest/
        dest_path = os.path.join(ingest_dir, "config.yaml")
        shutil.copy2(src_path, dest_path)

        # Step 3: Load YAML
        with open(dest_path, "r") as f:
            config = yaml.safe_load(f)

        # Step 4: Update input_dir value
        # Assuming: config['filereader'][0]['inputs']['loader_args']['input_dir']
        try:
            config['filereader'][0]['inputs']['loader_args']['input_dir'] = new_input_dir
        except (KeyError, IndexError, TypeError) as e:
            return f"Error navigating config structure: {e}"

        # Step 5: Save the updated YAML
        with open(dest_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        #return f"Updated input_dir in: {dest_path}"
        return dest_path

    
    except FileNotFoundError:
        return f"Error: Source file '{src_path}' not found."
    except yaml.YAMLError as e:
        return f"Error parsing YAML: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"



# Call the function
#print(update_input_dir_in_config("config.yaml", "/home/sharmista/sharmi.txt"))


def run_d3x_ingestion_delete_dataset(_=None):
    dest_path=  (update_input_dir_in_config("config.yaml", "/home/sharmistha-choudhury/hackathon/data"))
    dataset_name= "sharmi"
    try:
        result = subprocess.run(["d3x", "dataset", "ingest", "-d", dataset_name, "--config", dest_path], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"
    except FileNotFoundError:
        return "Error: 'd3x' command not found. Is it installed?"

ingest_tool = Tool(
    name="run_d3x_ingestion_delete_dataset()",
    func=run_d3x_ingestion_delete_dataset,
    description="Runs 'd3x ingest command ' in the terminal and returns the output",
    handle_parsing_errors=True
)

# agent.run("Can you show me what the emb command does?")
# print(response)

agent = initialize_agent(
    tools=[ingest_tool],
    llm=llm,
    #agent="ZERO_SHOT_REACT_DESCRIPTION",
    verbose=True
)

agent.run("Can you run ingestion command")
print(response)
