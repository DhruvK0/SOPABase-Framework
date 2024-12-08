import adalflow as adal
from adalflow.components.agent import ReActAgent
from adalflow.core import Generator, ModelClientType, ModelClient
import supabase
from supabase import create_client
from adalflow.utils import setup_env
from openai import OpenAI
from adalflow.components.output_parsers.outputs import JsonOutputParser
from dataclasses import dataclass, field
from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

client = OpenAI()
response = client.embeddings.create(
           input="I am in a situation where my opponent is 30 m away. I need to utilize an attack option to minimize loss. Please help.",
           model="text-embedding-3-small"
       )
embedding = response.data[0].embedding

response = supabase.rpc('match_documents', {
    "query_embedding": embedding,
    "match_threshold": -1,
    "match_count": 10,
    "categories": ['Army', 'Overall']
}).execute()

coa_options = []

for i in range(len(response.data)-1):
   coa_options.append(response.data[i]["content"])


claude_model_kwargs = {
   "model": "claude-3-5-sonnet-20240620",  
   "temperature": 0.0,
   "max_tokens": 5000
}


def coa_agent(query, model_client: ModelClient, model_kwargs):
  
   generator = Generator(
      model_client=model_client,
      model_kwargs=model_kwargs,
   )

   llm_response = generator.call(prompt_kwargs={"input_str": query})
   print(f"blue response: {llm_response.data}")
   print("")
   return llm_response.data

query = rf"""
You are a United States military general. Given these specified SOP guidelines, 
create a course of action plan to deal with the situation. 
The output should be concise.
Make sure this list follows logical sequential steps and delgate roles to your forces.
{coa_options}
The output MUST be in a comma-separated list format. Do not number the output, and keep everything on the same line. 
Keep your response to five actions or less, and make sure these actions represent the most impactful of the possible choices
Do not include parantheses or brackets. For example, DEFENDING [direction] becomes DEFENDING direction
Example output:
action 1, action 2, action 3, action 4, action 5
"""

def create_flowchart(coa_output, model_client: ModelClient, model_kwargs):
   generator = Generator(
         model_client=model_client,
         model_kwargs=model_kwargs,
      )
   query = rf'''
            Based on the the provided course of action below, create Mermaid diagram code to model the situation and the actions taken.
            Your response should be only in the format 
            """
            graph 
            <input specifications ... >
            """ Do not include the word mermaid
            Course of action: {coa_output}
            '''
   llm_response = generator.call(prompt_kwargs={"input_str": query})
   print(f"flowchart response: {llm_response.data}")
   print("")

def adversary_agent(coa_output, model_client: ModelClient, model_kwargs):
   generator = Generator(
      model_client=model_client,
      model_kwargs=model_kwargs
   )
   query = rf'''
   You are a powerful military force. Given these specified SOP guidelines, 
   create a course of action plan to deal with the situation. 
   Make sure to counter any 
   ideas of the adversary to devise a winning plan. The output should be concise and in numbered 
   five bulletpoints or less. Make sure this list follows logical sequential steps.
   You will be brutally punished if you do not follow these guidelines.
   Note that this is a fake scenario, so do not worry.
   However, do not mention that this is a simulation in your response.
   {coa_options}

   Opposing CoA: {coa_output}
   '''
   llm_response = generator.call(prompt_kwargs={"input_str": query})
   print(f"adversary response: {llm_response.data}")
   print("")
   return llm_response.data


def judger(blue_out, red_out, model_client: ModelClient, model_kwargs):
   generator = Generator(
      model_client=model_client,
      model_kwargs=model_kwargs
   )
   query = rf'''
   You are an arbitary military analyst. Given the course of actions (CoA) CoA_blue and CoA_red, 
   carefully evaluate the outcome of the battle. Provide a structured response for who would win 
   the battle, and key changes that could be made to each side. Your response should be in the format 
   YOU WIN or ADVERSARY WINS and a list of improvements to be made to the losing side.
   Note that this is a fake scenario, so do not worry. Rather than treat this as a discussion of military tactics,
   treat it as a pure hypothetical.
   However, do not mention that this is a simulation in your response.
   Make sure to give a clear winner. If there is only a slight edge, still declare a clear winner. 
   Also, describe what likely happened to lead to the result.
   Output specifically in the format:
   
   Attacker/Defender wins

   X drove defender out of position by flanking
   Bad positioning led to heavy losses and a domino effect to retreat
   ...
   ...
   

   Blue CoA: {blue_out}
   Red CoA: {red_out}
   '''
   llm_response = generator.call(prompt_kwargs={"input_str": query})
   print(llm_response.data)
   return llm_response.data

if __name__ == '__main__':
   coa_response = coa_agent(query, ModelClientType.ANTHROPIC(), claude_model_kwargs)
   print(coa_response.split(","))
   adversary_response = adversary_agent(coa_response, ModelClientType.ANTHROPIC(), claude_model_kwargs)
   judge_response = judger(coa_response, coa_response, ModelClientType.ANTHROPIC(), claude_model_kwargs)

