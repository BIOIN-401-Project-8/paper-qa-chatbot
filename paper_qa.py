import pickle
import os

os.environ["OPENAI_API_KEY"] = "sk-XXXXXXXXX"
# import paperqa


# from paperqa import Docs
from langchain_community.llms import LlamaCpp

# from langchain import PromptTemplate, LLMChain
# from langchain.callbacks.manager import CallbackManager
# from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from glob import glob

# Make sure the model path is correct for your system!
# n_gpu_layers = 16
# n_batch = 1
n_ctx = 4096

with open("docs.pickle", "rb") as f:
    docs = pickle.load(f)

llm = LlamaCpp(
    model_path="../mistral-7b-v0.1.Q4_K_M.gguf",
    callbacks=[StreamingStdOutCallbackHandler()],
    # n_gpu_layers=n_gpu_layers,
    # b_batch=n_batch,
    n_ctx=n_ctx,
)

docs.update_llm(llm)

answer = docs.query("What is Kallmann syndrome?", k=3, max_sources=3)
print("-- ANSWER --")
print(answer)
