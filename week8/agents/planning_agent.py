from typing import Optional, List
from agents.agent import Agent
from agents.deals import ScrapedDeal, DealSelection, Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent


class PlanningAgent(Agent):

    name = "Planning Agent"
    color = Agent.GREEN  # agent的颜色，继承自 Agent
    DEAL_THRESHOLD = 50    # 交易阈值，当折扣超过该值时会触发通知

    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across  创建此planner协调的 3 个agent的实例
        """
        self.log("Planning Agent is initializing")
        self.scanner = ScannerAgent()  # 初始化scanner agent，用于从 RSS 源中查找交易
        self.ensemble = EnsembleAgent(collection)   # 初始化ensemble agent，用于对商品进行价格估算
        self.messenger = MessagingAgent()   # 初始化messaging agent，用于发送交易通知
        self.log("Planning Agent is ready")

    def run(self, deal: Deal) -> Opportunity:
        """
        Run the workflow for a particular deal  为特定交易运行工作流
        :param deal: the deal, summarized from an RSS scrape 从 RSS 抓取汇总得到的交易信息
        :returns: an opportunity including the discount  包含折扣信息的交易机会
        """
        self.log("Planning Agent is pricing up a potential deal")
        estimate = self.ensemble.price(deal.product_description) # 使用ensemble agent对商品进行价格估算
        discount = estimate - deal.price  # 计算折扣，即估算价格减去交易价格
        self.log(f"Planning Agent has processed a deal with discount ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)   # 返回包含交易、估算价格和折扣的交易机会对象

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        """
        Run the full workflow:
        1. Use the ScannerAgent to find deals from RSS feeds
        2. Use the EnsembleAgent to estimate them
        3. Use the MessagingAgent to send a notification of deals

        运行完整的工作流：
        1. 使用 ScannerAgent 从 RSS 源中查找交易
        2. 使用 EnsembleAgent 对交易进行价格估算
        3. 使用 MessagingAgent 发送符合条件的交易通知

        :param memory: a list of URLs that have been surfaced in the past  过去已出现过的交易 URL 列表
        :return: an Opportunity if one was surfaced, otherwise None  如果找到符合条件的交易机会则返回该对象，否则返回 None
        """
        self.log("Planning Agent is kicking off a run")
        selection = self.scanner.scan(memory=memory)  # 使用scanner agent从 RSS 源中查找交易
        if selection:
            opportunities = [self.run(deal) for deal in selection.deals[:5]]  # 对前5个交易分别运行工作流，得到交易机会列表
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)  # 按折扣从高到低对交易机会列表进行排序
            best = opportunities[0]  # 获取折扣最高的交易机会
            self.log(f"Planning Agent has identified the best deal has discount ${best.discount:.2f}")
            if best.discount > self.DEAL_THRESHOLD:  # 如果最佳交易的折扣超过阈值，则使用消息代理发送通知
                self.messenger.alert(best)
            self.log("Planning Agent has completed a run")
            return best if best.discount > self.DEAL_THRESHOLD else None    # 如果最佳交易的折扣超过阈值则返回该交易机会，否则返回 None
        return None