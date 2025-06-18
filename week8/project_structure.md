graph TD
subgraph User Interface
price_is_right_final.py[Gradio UI] -->|显示日志, 机会, 图表| deal_agent_framework.py
price_is_right_final.py -->|获取图表数据| deal_agent_framework.py
price_is_right_final.py -->|格式化日志| log_utils.py
end

    subgraph Core Framework
        deal_agent_framework.py[DealAgentFramework] -->|协调| planning_agent.py
        deal_agent_framework.py -->|管理内存| memory.json(本地文件)
        deal_agent_framework.py -->|与向量数据库交互| ChromaDB(products_vectorstore)
    end

    subgraph Agents
        planning_agent.py[PlanningAgent] -->|扫描优惠| scanner_agent.py
        planning_agent.py -->|评估优惠价格| ensemble_agent.py
        planning_agent.py -->|发送通知| messaging_agent.py

        scanner_agent.py[ScannerAgent] -->|抓取RSS源| deals.py
        scanner_agent.py -->|使用LLM进行筛选和总结| OpenAI/DeepSeek LLM(外部服务)

        ensemble_agent.py[EnsembleAgent] -->|价格预测| specialist_agent.py
        ensemble_agent.py -->|价格预测 (RAG)| frontier_agent.py
        ensemble_agent.py -->|价格预测 (ML)| random_forest_agent.py
        ensemble_agent.py -->|加载集成模型| ensemble_model.pkl(本地文件)

        specialist_agent.py[SpecialistAgent] -->|调用远程LLM服务| Modal (pricer-service)(外部服务)

        frontier_agent.py[FrontierAgent] -->|文本编码| SentenceTransformer(外部库)
        frontier_agent.py -->|查询相似商品| ChromaDB(products_vectorstore)
        frontier_agent.py -->|使用LLM进行定价| OpenAI/DeepSeek LLM(外部服务)
        frontier_agent.py -->|使用Item类| items.py
        frontier_agent.py -->|使用Tester类| testing.py

        random_forest_agent.py[RandomForestAgent] -->|文本编码| SentenceTransformer(外部库)
        random_forest_agent.py -->|加载随机森林模型| random_forest_model.pkl(本地文件)

        messaging_agent.py[MessagingAgent] -->|发送短信 (可选)| Twilio(外部服务, 通过.env配置)
        messaging_agent.py -->|发送推送 (可选)| Pushover(外部服务, 通过.env配置)
    end

    subgraph Data Models & Utilities
        deals.py[deals.py] -->|定义数据模型 (ScrapedDeal, Deal, Opportunity)| planning_agent.py, scanner_agent.py, messaging_agent.py
        agent.py[agent.py] -->|代理基类| planning_agent.py, scanner_agent.py, ensemble_agent.py, specialist_agent.py, frontier_agent.py, random_forest_agent.py, messaging_agent.py
        items.py[items.py] -->|商品数据结构, 文本处理| frontier_agent.py
        log_utils.py[log_utils.py] -->|日志格式化| price_is_right_final.py
        testing.py[testing.py] -->|模型测试工具| frontier_agent.py
    end

    subgraph Configuration
        .env[环境变量文件.env] -->|提供API密钥等| messaging_agent.py, frontier_agent.py
    end
