import adalflow as adal
from supabase import create_client
import numpy as np
from typing import Dict, Union
import os
from dotenv import load_dotenv

load_dotenv()

# Step 1: Supabase setup
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

# Define the AdalFlow Query Classifier component
class QueryClassifier(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()
        self.embedder = adal.Embedder(model_client=model_client, model_kwargs=model_kwargs)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def call(self, query: str, supabase_table: str) -> Union[adal.GeneratorOutput, str]:
        # Step 1: Generate embedding for the query
        query_embedding = self.embedder(query)

        # Step 2: Fetch embeddings and data from Supabase
        response = supabase.table(supabase_table).select("embeddings, category, content").execute()
        rows = response.data

        # Step 3: Find the best match
        best_match = None
        best_similarity = -1

        for row in rows:
            embedding = np.array(row["embeddings"])
            similarity = self.cosine_similarity(query_embedding, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = row

        # Step 4: Return the best category
        if best_match:
            return best_match["category"]
        else:
            return "No relevant category found"

# Define the AdalFlow Task Pipeline
def create_pipeline(model_client, model_kwargs):
    # Initialize the Query Classifier component
    query_classifier = QueryClassifier(model_client, model_kwargs)

    # Define the pipeline
    pipeline = adal.Pipeline([query_classifier])
    return pipeline

# Define the optimization process
def optimize_pipeline(model_client, model_kwargs, pipeline):
    # Define system prompt and few-shot examples
    system_prompt = adal.Parameter(
        data="You are a classifier. Given a query, classify it into one of the predefined categories.",
        role_desc="Task instruction for the language model",
        requires_opt=True,
        param_type=adal.ParameterType.PROMPT,
    )

    few_shot_demos = adal.Parameter(
        data=None,  # Add your few-shot examples here
        role_desc="Few-shot examples to guide the model",
        requires_opt=True,
        param_type=adal.ParameterType.DEMOS,
    )

    # Initialize the optimizer
    optimizer = adal.optim.TextOptimizer(
        parameters=[system_prompt, few_shot_demos],
        model_client=model_client,
        model_kwargs=model_kwargs,
        task_pipeline=pipeline,
    )

    # Run the optimization
    optimizer.optimize()

# Main function to classify a query
def classify_query(query, supabase_table, model_client, model_kwargs):
    # Create and optimize the pipeline
    pipeline = create_pipeline(model_client, model_kwargs)
    optimize_pipeline(model_client, model_kwargs, pipeline)

    # Classify the query
    category = pipeline.run(query=query, supabase_table=supabase_table)
    return category

# Example usage
if __name__ == "__main__":
    # Model client setup (Replace with actual model client and kwargs)
    model_client = adal.ModelClient(api_key="your-api-key", model_name="text-embedding-ada-002")
    model_kwargs = {"temperature": 0.2, "max_tokens": 500}

    # Query and Supabase table
    query = "What are best practices for data encryption?"
    supabase_table = "your_table_name"

    # Classify the query
    category = classify_query(query, supabase_table, model_client, model_kwargs)
    print(f"The query falls under the category: {category}")
