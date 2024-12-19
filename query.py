from dotenv import load_dotenv

from langchain import hub
# from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from colorama import Fore
from utils import clean_text
import warnings
import os
warnings.filterwarnings("ignore")


load_dotenv()


### BEDROCK LLM & EMBEDDINGS ###

# To get profile name, download AWS Cli and run "aws configure --profile bedrock", then provide credentials
# llm = BedrockLLM(
#     credentials_profile_name="bedrock", 
#     model_id="amazon.titan-text-express-v1"
# )

# def get_embedding_function():
#     embeddings = BedrockEmbeddings(
#         credentials_profile_name="bedrock", 
#         model_id="cohere.embed-multilingual-v3",
#         region_name="eu-west-3"
#     )
#     return embeddings

######

### OPENAI LLM & EMBEDDINGS ###

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in the environment.")


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)
 
def get_embedding_function():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
    )
    return embeddings

######


def load_documents():
    loader = PyPDFDirectoryLoader("docs")
    documents = loader.load()
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False
    )
    doc_splits = text_splitter.split_documents(documents)
    for doc in doc_splits:
        doc.page_content = clean_text(doc.page_content)
    return doc_splits


try:
    print("Creating embedding function...")
    embeddings = get_embedding_function()
    print("Embedding function created")

    print("Fetching vectorstore...")
    vectorstore = Chroma(
        persist_directory="data", 
        embedding_function=embeddings,
        collection_name="rag-chroma"
    )
    
    vectordict = vectorstore.get()
    existing_ids = set(vectordict['ids'])

    if len(existing_ids) == 0:
        print("Vectorstore not found, creating vectorstore...")
        if not os.path.exists("data"):
            os.makedirs("data")

        doc_splits = split_documents(load_documents())

        vectorstore = Chroma(
            embedding_function=embeddings, 
            persist_directory="data",
            collection_name="rag-chroma"
        )

        print("Adding documents to vectorstore...")
        vectorstore.add_documents(documents=doc_splits)
        print("Documents added to vectorstore.")
        vectorstore.persist()
    else:
        print("Vectorstore found.")

    retriever = vectorstore.as_retriever()
    print("retriever created successfully")
except Exception as e:
    print("An error occurred:", e)


### Retrieval Grader : Retrieval Evaluator ###
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

    def get_score(self) -> str:
        return self.binary_score


def get_score(self) -> str:
    return self.binary_score

# LLM with function call 
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt 
system_template = """You are an evaluator determining the relevance of a retrieved {documents} to a user's query {question}.If the document contains keyword(s) or semantic meaning related to the question, mark it as relevant.Assign a binary score of 'yes' or 'no' to indicate the document's relevance to the question."""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["documents", "question"],
    template="{question}",
)
grader_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

######


### Question Re-writer - Knowledge Refinement ###
# Prompt 
prompt_template = """Given a user input {question}, your task is re-write or rephrase the question to optimize the query in order to imprive the content generation"""

system_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
human_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["question"],
    template="{question}",
)
re_write_prompt = ChatPromptTemplate.from_messages(
    [system_prompt, human_prompt]
)

### Web Search Tool - Knowledge Searching ###
web_search_tool = TavilySearchResults(k=3) 


### Generate Answer  ###
# Prompt
# !!! Final prompt, telling the assistant to stay concise with 3 max sentences
prompt = hub.pull("rlm/rag-prompt")


# Retrieve and assess
def assess_retrieved_docs(query):
    retrieval_grader = grader_prompt | structured_llm_grader | get_score
    docs = retriever.get_relevant_documents(query) 
    doc_txt = docs[0].page_content
    binary_score = retrieval_grader.invoke({"question": query, "documents": doc_txt})
    return binary_score, docs


# Rewrite and optimize 
def rewrite_query(query):
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter.invoke({"question": query})


# Search the web
def search_web(query):
    docs = web_search_tool.invoke({"query": query})
    web_results = "\n".join([d["content"] for d in docs])
    return Document(page_content=web_results)


def generate_answer(docs, query):
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain.invoke({"context": docs, "question": query})


def query(query):
    # Grade the retrieved documents
    binary_score, docs = assess_retrieved_docs(query)
    print(f"Relevance score: {binary_score}")

    # Rewrite and optimize the query
    print(f"{Fore.YELLOW}Rewriting the query for content generation.{Fore.RESET}")
    optimized_query = rewrite_query(query)
    print(f"Optimized query: {optimized_query}")

    if binary_score == "no":
        print(f"{Fore.MAGENTA}Retrieved documents are irrelevant. Searching the web for additional information.{Fore.RESET}")
        docs = search_web(optimized_query)

    return generate_answer(docs, optimized_query)