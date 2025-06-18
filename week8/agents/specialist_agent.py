import modal
from agents.agent import Agent


class SpecialistAgent(Agent):
    """
    An Agent that runs our fine-tuned LLM that's running remotely on Modal
     一个运行已经在 Modal 上远程部署的微调大语言模型的agent
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        """
        Set up this Agent by creating an instance of the modal class
        通过创建 modal 类的实例来初始化此agent
        """
        self.log("Specialist Agent is initializing - connecting to modal")
        Pricer = modal.Cls.from_name("pricer-service", "Pricer") # 从 Modal 中获取名为 'pricer-service' 应用下的 'Pricer' 类
        self.pricer = Pricer() # 创建 Pricer 类的实例
        self.log("Specialist Agent is ready")
        
    def price(self, description: str) -> float:
        """
        Make a remote call to return the estimate of the price of this item
        进行远程调用，返回此物品的价格估计值
        """
        self.log("Specialist Agent is calling remote fine-tuned model")
        result = self.pricer.price.remote(description)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result
