# imports

import os
import re
from typing import List
from sentence_transformers import SentenceTransformer
import joblib
from agents.agent import Agent



class RandomForestAgent(Agent):

    name = "Random Forest Agent"
    color = Agent.MAGENTA

    def __init__(self):
        """
        Initialize this object by loading in the saved model weights
        and the SentenceTransformer vector encoding model
        """
        self.log("Random Forest Agent is initializing")
        self.vectorizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 初始化 SentenceTransformer 向量编码器
        self.model = joblib.load('random_forest_model.pkl')  # 使用 joblib 加载预训练的随机森林模型
        self.log("Random Forest Agent is ready")

    def price(self, description: str) -> float:
        """
        Use a Random Forest model to estimate the price of the described item
        :param description: the product to be estimated
        :return: the price as a float

        使用随机森林模型估计所描述物品的价格
        :param description: 待估计价格的商品描述
        :return: 估计的价格，以浮点数形式返回
        """        
        self.log("Random Forest Agent is starting a prediction")
        vector = self.vectorizer.encode([description])
        result = max(0, self.model.predict(vector)[0])
        self.log(f"Random Forest Agent completed - predicting ${result:.2f}")
        return result