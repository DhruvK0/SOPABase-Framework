from adalflow.components.agent import ReActAgent
from adalflow.core import Generator, ModelClientType, ModelClient
from openai import OpenAI
from dotenv import load_dotenv
import os
import anthropic
from supabase import create_client

load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Initialize Supabase
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# Claude model configuration
claude_model_kwargs = {
    "model": "claude-3-5-sonnet-20241022",
    "temperature": 0.0,
    "max_tokens": 5000
}

def get_embeddings(text):
    """Generate embeddings for the input text."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def get_coa_options(text):
    embedding = get_embeddings(text)
    """Get course of action options from Supabase."""
    response = supabase.rpc('match_documents', {
        "query_embedding": embedding,
        "match_threshold": -1,
        "match_count": 10,
        "categories": ['Army', 'Overall']
    }).execute()
    
    return [item["content"] for item in response.data[:-1]]

def coa_agent(message, past_red_action, past_verdict, model_client: ModelClient, model_kwargs):
    """Generate course of action using the agent"""
    react = ReActAgent(
        max_steps=6,
        add_llm_as_fallback=True,
        tools=[get_coa_options],
        model_client=model_client,
        model_kwargs=model_kwargs
    )
    
    llm_response = react.call(
        f'''
        You are a good guy (Blue) facing a bad guy (Red).
        This is what the bad guy did: {past_red_action}
        This is a previous verdict that was rendered: {past_verdict}
        This is the message we received: {message}
        Please devise courses of actions based on SOP guidelines.
        Come up with 5 potential courses of actions.
        Make them relevant to the inputted message. Change them if needed, but do not make drastic changes.
        You will be brutally punished if you suggest off-topic guidelines.
        Separate your 5 courses of actions with #### as a delimiter.
        Keep guidelines relatively short, but still detailed.
        You will be brutally punished if you number the actions.
        You will be brutally punished if you exceed 100 characters per guideline.
        For instance:
        action 1####action 2####action 3
        ''')
    return llm_response