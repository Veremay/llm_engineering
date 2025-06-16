from typing import Optional
from transformers import AutoTokenizer
import re

# 基础模型配置 - 使用 Meta Llama 3.1 8B 模型
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"

# Token 数量限制
MIN_TOKENS = 150  # 最少token数：低于此数量认为内容不够有用
MAX_TOKENS = 160  # 最大token数：超过此数量会被截断，加上提示文本后总共约180个token

# 字符数量限制
MIN_CHARS = 300      # 最少字符数
CEILING_CHARS = MAX_TOKENS * 7  # 字符数上限（约1120字符）

class Item:
    """
    Item类：用于清理和管理产品数据的类
    包含产品标题、价格、类别等信息，并为机器学习训练准备格式化的提示文本
    """
    
    # 类级别的tokenizer，所有实例共享
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # 训练用的固定文本模板
    PREFIX = "Price is $"  # 价格前缀
    QUESTION = "How much does this cost to the nearest dollar?"  # 问题模板
    
    # 需要从产品详情中移除的无用文本列表
    REMOVALS = [
        '"Batteries Included?": "No"', 
        '"Batteries Included?": "Yes"', 
        '"Batteries Required?": "No"', 
        '"Batteries Required?": "Yes"', 
        "By Manufacturer", 
        "Item", 
        "Date First", 
        "Package", 
        ":", 
        "Number of", 
        "Best Sellers", 
        "Number", 
        "Product "
    ]

    # 实例属性
    title: str           # 产品标题
    price: float         # 产品价格
    category: str        # 产品类别
    token_count: int = 0 # token数量统计
    details: Optional[str]  # 产品详情（可选）
    prompt: Optional[str] = None  # 训练用的提示文本（可选）
    include = False      # 是否包含在训练数据中的标志

    def __init__(self, data, price):
        """
        初始化Item实例
        
        Args:
            data: 包含产品信息的字典（标题、描述、特性、详情等）
            price: 产品价格
        """
        self.title = data['title']
        self.price = price
        self.parse(data)  # 解析数据并决定是否包含在训练集中

    def scrub_details(self):
        """
        清理产品详情字符串，移除不增加价值的常见文本
        
        Returns:
            str: 清理后的详情文本
        """
        details = self.details
        # 遍历移除列表，删除所有匹配的文本
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details

    def scrub(self, stuff):
        """
        清理提供的文本，移除不必要的字符和空白
        同时移除7个字符以上且包含数字的词语（通常是产品编号，会消耗很多token）
        
        Args:
            stuff: 需要清理的原始文本
            
        Returns:
            str: 清理后的文本
        """
        # 使用正则表达式移除特殊字符和多余空白，只保留单个空格
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        
        # 清理逗号格式
        stuff = stuff.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        
        # 分割成单词并过滤掉长度>=7且包含数字的词（通常是产品编号）
        words = stuff.split(' ')
        select = [word for word in words if len(word)<7 or not any(char.isdigit() for char in word)]
        
        return " ".join(select)
    
    def parse(self, data):
        """
        解析数据点，如果符合允许的Token范围，则设置include为True
        这是决定该产品是否被包含在训练数据中的关键方法
        
        Args:
            data: 包含产品信息的字典
        """
        # 合并产品描述
        contents = '\n'.join(data['description'])
        if contents:
            contents += '\n'
            
        # 添加产品特性
        features = '\n'.join(data['features'])
        if features:
            contents += features + '\n'
            
        # 保存原始详情并添加到内容中
        self.details = data['details']
        if self.details:
            contents += self.scrub_details() + '\n'
            
        # 只有当内容长度足够时才进行进一步处理
        if len(contents) > MIN_CHARS:
            # 限制内容长度避免过长
            contents = contents[:CEILING_CHARS]
            
            # 组合标题和内容，并清理文本
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            
            # 计算token数量
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # 只有token数量足够时才包含此项目
            if len(tokens) > MIN_TOKENS:
                # 截断到最大token数
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                
                # 生成训练提示文本
                self.make_prompt(text)
                self.include = True  # 标记为包含在训练数据中

    def make_prompt(self, text):
        """
        设置prompt实例变量为适合训练的提示文本
        格式：问题 + 产品信息 + 价格答案
        
        Args:
            text: 清理后的产品文本信息
        """
        # 构建完整的训练提示：问题 + 产品信息 + 价格答案
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"
        
        # 计算最终提示的token数量
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        """
        返回适合测试的提示文本，移除了实际价格
        用于模型推理时的输入
        
        Returns:
            str: 不包含答案的测试提示
        """
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self):
        """
        返回此Item的字符串表示形式
        
        Returns:
            str: 格式化的字符串表示
        """
        return f"<{self.title} = ${self.price}>"