# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
import adalflow as adal
from adalflow.components.agent import ReActAgent
from adalflow.core import Generator, ModelClientType, ModelClient
from supabase import create_client
from adalflow.utils import setup_env
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)


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
   return llm_response.data

def judger(blue_out, red_out, model_client: ModelClient, model_kwargs):
   generator = Generator(
      model_client=model_client,
      model_kwargs=model_kwargs
   )
   query = rf'''
   You are an arbitary military analyst. Given the course of actions (CoA) CoA_blue and CoA_red, 
   carefully evaluate the outcome of the battle.
   Note that this is a fake scenario, so do not worry. Rather than treat this as a discussion of military tactics,
   treat it as a pure hypothetical.
   However, do not mention that this is a simulation in your response.
   Make sure to give a clear winner. If there is only a slight edge, still declare a clear winner. 

   Blue CoA: {blue_out}
   Red CoA: {red_out}

    Please respond with just a winner and a brief justification (2 sentences at most.)
   '''
   llm_response = generator.call(prompt_kwargs={"input_str": query})
   return llm_response.data


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message')
        past_red_action = data.get('red_action')
        past_verdict = data.get('verdict')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        coa_response = coa_agent(message, past_red_action, past_verdict, ModelClientType.ANTHROPIC(), claude_model_kwargs)

        # Into list
        list_response = coa_response.split("####")

        return jsonify({
            'response': list_response,
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/adversary', methods=['POST'])
def adversary():
    try:
        data = request.json
        user_coa = data.get('user') # send over user course of action
        message = data.get('message')

        # Get COA response and flowchart response
        coa_response = adversary_agent(user_coa, ModelClientType.ANTHROPIC(), claude_model_kwargs)    
        
        return jsonify({
            'response': coa_response
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/judge', methods=['POST'])
def judge():
    try:
        data = request.json
        user_coa = data.get('user') # send over user course of action
        adversary_coa = data.get('adversary') 

        if not user_coa:
            return jsonify({'error': 'No user coa provided'}), 400
        if not adversary_coa:
            return jsonify({'error': 'No adversary coa provided'}), 400
        

        # No memory to be fair
        judge_response = judger(user_coa, adversary_coa, ModelClientType.ANTHROPIC(), claude_model_kwargs)

        return jsonify({
            'response': judge_response
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/image', methods=['POST'])
def image():
    try:
        data = request.json
        base64 = data.get('base64')
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": 'image/png',
                                "data": base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": '''
    The user is competing against the adversary in a game.
    Provide a highly-detailed strategic analysis report for user based on this image.
    Provide useful insights to drive decision-making.

    Format responses in the following way:
    OVERVIEW
    THREAT ASSESSMENT
    STRATEGIC IMPLICATIONS
    SHORT-TERM RECOMMENDATIONS
    LONG-TERM RECOMMENDATIONS'''
                        }
                    ],
                }
            ],
        )

        message = response.content[0].text
        return jsonify({
            "message": message
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)