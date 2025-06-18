from pydantic import BaseModel
from typing import List, Dict, Self
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time

# 定义 RSS 订阅源列表，用于抓取商品交易信息
feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
        "https://www.dealnews.com/c39/Computers/?rss=1",
        "https://www.dealnews.com/c238/Automotive/?rss=1",
        "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
        "https://www.dealnews.com/c196/Home-Garden/?rss=1",
       ]

def extract(html_snippet: str) -> str:
    """
    Use Beautiful Soup to clean up this HTML snippet and extract useful text
    使用 Beautiful Soup 清理 HTML 片段并提取有用的文本

    :param html_snippet: 输入的 HTML 片段
    :return: 清理后的文本
    """
    soup = BeautifulSoup(html_snippet, 'html.parser')   # 创建 BeautifulSoup 对象，用于解析 HTML
    snippet_div = soup.find('div', class_='snippet summary')  # 查找 class 为 'snippet summary' 的 div 标签
    
    if snippet_div:
        # 提取 div 标签内的文本并去除首尾空格
        description = snippet_div.get_text(strip=True)
        # 再次使用 BeautifulSoup 清理文本
        description = BeautifulSoup(description, 'html.parser').get_text()
        # 使用正则表达式去除 HTML 标签
        description = re.sub('<[^<]+?>', '', description)
        result = description.strip()
    else:
        # 如果未找到指定 div 标签，则直接使用原始 HTML 片段
        result = html_snippet
    # 替换换行符为空格并返回结果
    return result.replace('\n', ' ')


class ScrapedDeal:
    """
    A class to represent a Deal retrieved from an RSS feed
    一个表示从 RSS 订阅源获取的交易信息的类
    """
    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str]):
        """
        Populate this instance based on the provided dict
        根据提供的字典填充该实例。

        :param entry: 包含交易信息的字典
        """
        # 从字典中获取交易标题
        self.title = entry['title']
        # 调用 extract 函数清理并提取交易摘要
        self.summary = extract(entry['summary'])
        # 从字典中获取交易链接
        self.url = entry['links'][0]['href']
        # 发送 HTTP 请求获取交易页面内容
        stuff = requests.get(self.url).content
        # 创建 BeautifulSoup 对象，用于解析交易页面内容
        soup = BeautifulSoup(stuff, 'html.parser')
        # 查找 class 为 'content-section' 的 div 标签并提取文本
        content = soup.find('div', class_='content-section').get_text()
        # 替换特定文本并去除换行符
        content = content.replace('\nmore', '').replace('\n', ' ')
        if "Features" in content:
            # 如果内容中包含 "Features"，则分割内容为详情和特性两部分
            self.details, self.features = content.split("Features")
        else:
            # 否则，将全部内容作为详情，特性部分为空
            self.details = content
            self.features = ""

    def __repr__(self):
        """
        Return a string to describe this deal
        返回一个描述此交易的字符串
        """
        return f"<{self.title}>"

    def describe(self):
        """
        Return a longer string to describe this deal for use in calling a model
        返回一个描述此交易的更长字符串，用于在调用模型时使用
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress : bool = False) -> List[Self]:
        """
        Retrieve all deals from the selected RSS feeds
        从选定的 RSS 订阅源中检索所有交易信息。

        :param show_progress: 是否显示进度条，默认为 False
        :return: 包含 ScrapedDeal 实例的列表
        """
        deals = []
        # 根据 show_progress 参数决定是否使用进度条
        feed_iter = tqdm(feeds) if show_progress else feeds
        for feed_url in feed_iter:
            # 解析 RSS 订阅源
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                # 创建 ScrapedDeal 实例并添加到列表中
                deals.append(cls(entry))
                # 每次请求间隔 0.5 秒，避免给服务器造成过大压力
                time.sleep(0.5)
        return deals

class Deal(BaseModel):
    """
    A class to Represent a Deal with a summary description
    一个表示带有摘要描述的交易的类，使用 pydantic 定义
    """
    product_description: str
    price: float
    url: str

class DealSelection(BaseModel):
    """
    A class to Represent a list of Deals
    一个表示交易列表的类，使用 pydantic 定义
    """
    deals: List[Deal]

class Opportunity(BaseModel):
    """
    A class to represent a possible opportunity: a Deal where we estimate
    it should cost more than it's being offered
    一个表示可能的交易机会的类，即我们估计其价值高于售价的交易，使用 pydantic 定义
    """
    deal: Deal
    estimate: float
    discount: float