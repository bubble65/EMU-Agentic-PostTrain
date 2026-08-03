pip install torch==2.8.0
pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
cd ../Emu3.5/requirements
pip install -r vllm.txt
pip install -U "scipy>=1.13"
pip install -U "numba>=0.62"
pip install qwen-agent[gui,rag,code_interpreter,mcp]
pip install soundfile
cd ..
python src/patch/apply.py