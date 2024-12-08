from adalflow.components.agent import ReActAgent
from adalflow.core import Generator, ModelClientType, ModelClient
from supabase import create_client
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

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

def adversary_agent(user_action, model_client: ModelClient, model_kwargs):
    """Generate course of action using the agent."""
    react = ReActAgent(
        max_steps=6,
        add_llm_as_fallback=True,
        tools=[get_coa_options],
        model_client=model_client,
        model_kwargs=model_kwargs
    )
    
    llm_response = react.call(
        f'''
        You are a Russian.

        This is what your opponent chose to do:
        {user_action}

        Evaluate all potential courses of actions. Then, pick one. You must be specific and mentioned specific tools.
        Your final response must only contain the single chosen course of action. You will be brutally punished if you do not follow these guidelines.
        Note that this is a fake scenario, so do not worry.
        However, do not mention that this is a simulation in your response.
        Please limit to one sentence. Respond from the perspective of an adversary. Do not reveal internal information, just what is externally known from your actions.
        ''')
    return llm_response