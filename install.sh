export CMAKE_ARGS=-DLLAMA_CUBLAS=on
export FORCE_CMAKE=1
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose
cp libllama.so /usr/local/lib/python3.10/dist-packages/llama_cpp/libllama.so 
