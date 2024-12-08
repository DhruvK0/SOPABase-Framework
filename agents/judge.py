from adalflow.core import Generator, ModelClientType, ModelClient

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