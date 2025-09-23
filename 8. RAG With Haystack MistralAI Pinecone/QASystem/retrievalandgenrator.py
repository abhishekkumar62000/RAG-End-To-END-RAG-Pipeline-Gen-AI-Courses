from haystack.utils import Secret
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.components.generators import OpenAIGenerator
import os
from dotenv import load_dotenv
from haystack import Pipeline
from QASystem.ingestion import ingest
from QASystem.utility import pinecone_config
import os
from dotenv import load_dotenv

prompt_template = """Answer the following query based on the provided context. If the context does
                     not include an answer, reply with 'I don't know'.\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}
                     Answer: 
                  """
 
def get_result(query):
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    query_pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=pinecone_config()))
    query_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    query_pipeline.add_component("llm", OpenAIGenerator(api_key=Secret.from_token(OPENAI_API_KEY), model="gpt-3.5-turbo"))

    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever.documents", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder", "llm")

    results = query_pipeline.run(
        {
            "text_embedder": {"text": query},
            "prompt_builder": {"query": query},
        }
    )
    # Debug: print retrieved documents
    docs = results.get('retriever', {}).get('documents', [])
    print(f"Retrieved {len(docs)} documents for query: '{query}'")
    for i, doc in enumerate(docs):
        print(f"Document {i+1}: {doc.content[:200]}...")
    print(f"LLM reply: {results['llm']['replies'][0]}")
    return results['llm']['replies'][0]

if __name__ == '__main__':
    #loading the environment variable
    '''load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
    
    print("Import Successfully")'''
    
    result=get_result("what is rag?")
    print(result)
