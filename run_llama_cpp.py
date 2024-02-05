
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="llama-2-7b-chat.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    n_gpu_layers = -1,
    n_batch = 512,
    n_ctx = 512,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager

)

prompt = """
Question: A rap battle between Stephen Colbert and John Oliver
"""
llm.invoke(prompt)
