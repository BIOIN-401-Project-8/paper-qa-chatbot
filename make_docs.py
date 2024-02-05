import pickle
import os

os.environ["OPENAI_API_KEY"] = "sk-XXXXXXXXX"  # isort:skip

from paperqa import Docs
from langchain_community.llms import LlamaCpp, HuggingFacePipeline

# from langchain import PromptTemplate, LLMChain
# from langchain.callbacks.manager import CallbackManager
from langchain_community.embeddings import LlamaCppEmbeddings, HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from glob import glob
from pathlib import Path
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
from tqdm import tqdm
# Make sure the model path is correct for your system!
# model_path = "../mistral-7b-v0.1.Q4_K_M.gguf"
n_batch = 4096
chunk_chars = 512
n_ctx = 4096
# llm = LlamaCpp(
#     model_path=model_path,
#     callbacks=[StreamingStdOutCallbackHandler()],
#     # n_gpu_layers=-1,
#     # b_batch=n_batch,
#     # n_ctx=n_ctx,
# )
# llm = HuggingFacePipeline(
#     model_id="TheBloke/Mistral-7B-Instruct-v0.2-AWQ", task="text-generation",
#     pipeline_kwargs={"max_new_tokens": 10},
# )
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings =  HuggingFaceHubEmbeddings(
#     model_id="TheBloke/Mistral-7B-Instruct-v0.2-AWQ", task="text-generation",
#     pipeline_kwargs={"max_new_tokens": 10},
# )

my_docs = glob("../../data/articles/*.xml")
assert len(my_docs)
docs = Docs(embeddings=embeddings)
for d in tqdm(my_docs):
    with open(d, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "xml")
    title = soup.find("article-title").text or "" if soup.find("article-title") is not None else "title"
    author = soup.find("surname").text or "" if soup.find("surname") is not None else "author"
    year = soup.find("year").text or "" if soup.find("year") is not None else "year"
    citation = f"{author} {year}, {title} ({Path(d).stem})"
    document = soup.find("abstract").text or "" if soup.find("abstract") is not None else ""
    document += soup.find("body").text or "" if soup.find("body") is not None else ""
    with open(Path(d).with_suffix(".txt"), "w") as f:
        f.write(document)
    print(citation)
    try:
        docs.add(Path(d).with_suffix(".txt"), chunk_chars=chunk_chars, citation=citation, docname=d)
    except ValueError as e:
        if "This does not look like a text document" in str(e):
            print(f"Skipping {d}")

with open("docs.pickle", "wb") as f:
    pickle.dump(docs, f)
