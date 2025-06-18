# imports

import os
import re
import math
import json
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from items import Item
from testing import Tester
from agents.agent import Agent


class FrontierAgent(Agent):

    name = "Frontier Agent"
    color = Agent.BLUE

    MODEL = "gpt-4o-mini"   # 默认使用的模型
    
    def __init__(self, collection):
        """
        Set up this instance by connecting to OpenAI or DeepSeek, to the Chroma Datastore,
        And setting up the vector encoding model
        初始化 FrontierAgent 实例，连接到 OpenAI 或 DeepSeek，连接到 Chroma 数据存储，
        并设置向量编码模型。

        :param collection: Chroma 数据集合对象
        """
        self.log("Initializing Frontier Agent")
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            self.client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
            self.MODEL = "deepseek-chat"
            self.log("Frontier Agent is set up with DeepSeek")
        else:
            self.client = OpenAI()
            self.MODEL = "gpt-4o-mini"
            self.log("Frontier Agent is setting up with OpenAI")
        self.collection = collection  # 存储 Chroma 数据集合对象（传入在deal_agent_framework.py）
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 初始化句子编码模型
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar products to the one being estimated
        :param prices: prices of the similar products
        :return: text to insert in the prompt that provides context

        创建可插入提示中的上下文信息。

        :param similars: 与待估算商品相似的商品列表
        :param prices: 相似商品的价格列表
        :return: 包含上下文信息的文本，可插入提示中
        """
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to OpenAI
        With the system and user prompt
        :param description: a description of the product
        :param similars: similar products to this one
        :param prices: prices of similar products
        :return: the list of messages in the format expected by OpenAI

        创建用于调用 OpenAI 的消息列表，包含系统提示和用户提示。

        :param description: 商品的描述
        :param similars: 相似商品列表
        :param prices: 相似商品的价格列表
        :return: 符合 OpenAI 要求格式的消息列表
        """
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + description
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]

    def find_similars(self, description: str):
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        通过在 Chroma 数据存储中查找，返回与给定商品描述相似的商品列表。

        :param description: 商品的描述
        :return: 相似商品列表和对应的价格列表
        """
        self.log("Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products")
        # 对商品描述进行编码
        vector = self.model.encode([description])
        # 在 Chroma 数据存储中查询相似商品
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        # 获取相似商品的文档信息
        documents = results['documents'][0][:]
        # 获取相似商品的价格信息
        prices = [m['price'] for m in results['metadatas'][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s) -> float:
        """
        A utility that plucks a floating point number out of a string

        从字符串中提取浮点数作为价格。

        :param s: 包含价格信息的字符串
        :return: 提取到的价格，如果未找到则返回 0.0
        """
        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        """
        Make a call to OpenAI or DeepSeek to estimate the price of the described product,
        by looking up 5 similar products and including them in the prompt to give context
        :param description: a description of the product
        :return: an estimate of the price

        调用 OpenAI 或 DeepSeek 模型估算给定商品描述的价格，
        通过查找 5 个相似商品并将其信息包含在提示中提供上下文。

        :param description: 商品的描述
        :return: 估算的价格
        """
        documents, prices = self.find_similars(description)
        self.log(f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products")
        response = self.client.chat.completions.create(
            model=self.MODEL, 
            messages=self.messages_for(description, documents, prices),
            seed=42,
            max_tokens=5
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
        