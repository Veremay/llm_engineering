import modal
from modal import App, Volume, Image
# Setup - define our infrastructure with code!

# 创建一个名为 "pricer-service" 的 Modal 应用实例
app = modal.App("pricer-service")
# 基于 Debian Slim 镜像创建一个新镜像，并安装所需的 Python 库
image = Image.debian_slim().pip_install("huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft")

# This collects the secret from Modal.
# Depending on your Modal configuration, you may need to replace "hf-secret" with "huggingface-secret"
secrets = [modal.Secret.from_name("hf-secret")]

# Constants
GPU = "T4"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
PROJECT_NAME = "pricer"
HF_USER = "ed-donner" # your HF name here! Or use mine if you just want to reproduce my results.
RUN_NAME = "2024-09-13_13.04.39"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "e8d637df551603dc86cd7a1598a8f44af4d7ae36"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
CACHE_DIR = "/cache"

# Change this to 1 if you want Modal to be always running, otherwise it will go cold after 2 mins
# 设置 Modal 容器的最小数量，0 表示在空闲 2 分钟后进入休眠状态，1 表示始终运行
MIN_CONTAINERS = 0

QUESTION = "How much does this cost to the nearest dollar?"
PREFIX = "Price is $"

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)  # 从 Modal 获取名为 "hf-hub-cache" 的volume，如果不存在则创建

# 定义一个 Modal 类，用于运行定价服务
@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets, 
    gpu=GPU, 
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume}
)
class Pricer:

    @modal.enter()
    def setup(self):
        """
        初始化方法，在进入类实例时调用，用于加载模型和分词器
        """
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel
        
        # Quant Config
        # 配置 4 位量化参数
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        # Load model and tokenizer
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        # 将填充标记设置为结束标记
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # 设置填充方向为右侧
        self.tokenizer.padding_side = "right"
        # 从预训练模型加载基础模型，并应用量化配置
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            quantization_config=quant_config,
            device_map="auto"
        )
        # 从预训练的微调模型加载到基础模型上
        self.fine_tuned_model = PeftModel.from_pretrained(self.base_model, FINETUNED_MODEL, revision=REVISION)


    @modal.method()
    def price(self, description: str) -> float:
        """
        根据物品描述预测价格的方法
        :param description: 物品的描述信息
        :return: 预测的价格
        """
        import os
        import re
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        from peft import PeftModel
    
        set_seed(42)
        # 构建输入提示
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        # 对提示进行编码，并转换为 PyTorch 张量，移动到 GPU 上
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        # 创建注意力掩码  （再去研究一下！！重要！！）
        attention_mask = torch.ones(inputs.shape, device="cuda")
        # 使用微调后的模型生成输出
        outputs = self.fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
        # 对生成的输出进行解码
        result = self.tokenizer.decode(outputs[0])
    
        # 提取价格部分
        contents = result.split("Price is $")[1]
        # 移除价格中的逗号
        contents = contents.replace(',','')
        # 使用正则表达式匹配价格数值
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        # 如果匹配到价格数值，则转换为浮点数返回，否则返回 0
        return float(match.group()) if match else 0