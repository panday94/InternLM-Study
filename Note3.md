# 笔记
## 安装环境
- 安装conda环境
> 此步骤需要稍微等一会时间，请耐心等待
```
studio-conda -o internlm-base -t demo
# 上述命令中demo为coda环境名称方便切换coda环境，internlm-base应是包含了下面几行的命令，所以执行上述命令即可，如需想自己执行安装则可以执行如下命令（与 studio-conda 等效的配置方案）
# conda create -n demo python==3.10 -y
# conda activate demo
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- 进入刚才创建的为`demo`的conda环境
```
conda activate demo
```

- 继续安装环境包
```
pip install huggingface-hub==0.17.3
pip install transformers==4.34 
pip install psutil==5.9.8
pip install accelerate==0.24.1
pip install streamlit==1.32.2 
pip install matplotlib==3.8.3 
pip install modelscope==1.9.5
pip install sentencepiece==0.1.99
```

## 下载`InternLM2-Chat-1.8B`模型
- 按路径创建文件夹，并进入到对于文件目录
```
mkdir -p /root/demo
touch /root/demo/cli_demo.py
touch /root/demo/download_mini.py
cd /root/demo
```
- 下载模型
复制download_mini.py文件内容
```
import os
from modelscope.hub.snapshot_download import snapshot_download

# 创建保存模型目录
os.system("mkdir /root/models")

# save_dir是模型保存到本地的目录
save_dir="/root/models"

snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')

```

执行命令，下载模型参数文件

```
python /root/demo/download_mini.py
```
- 运行 cli_demo
复制cli_demo.py文件内容
```
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)

```

执行cli_demo.py

```
# 以上已在`demo`的conda环境中，可无需激活
conda activate demo
python /root/demo/cli_demo.py
```

等待运行完成即可看到下发作业内容

# 作业

- 使用 InternLM2-Chat-1.8B 模型生成 300 字的小故事
[](/docs/note2/work.jpg)