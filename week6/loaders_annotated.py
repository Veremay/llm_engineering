'''
核心功能：

数据加载 - 从Hugging Face的Amazon Reviews 2023数据集加载特定类别的商品数据
数据过滤 - 根据价格范围（0.5-999.49）和其他条件筛选有效商品
并行处理 - 使用多进程提高大数据集的处理速度
批量处理 - 将大数据集分块处理，优化内存使用

'''



# 导入必要的库
from datetime import datetime      # 用于记录时间和计算处理耗时
from tqdm import tqdm             # 用于显示进度条
from datasets import load_dataset # Hugging Face datasets库，用于加载数据集
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # 并发处理库
from items import Item            # 自定义的Item类

# 配置常量
CHUNK_SIZE = 1000    # 每个数据块的大小，用于批量处理
MIN_PRICE = 0.5      # 商品价格的最小值过滤条件
MAX_PRICE = 999.49   # 商品价格的最大值过滤条件

class ItemLoader:
    """
    商品数据加载器类
    用于从Amazon Reviews 2023数据集中加载和处理商品数据
    """

    def __init__(self, name):
        """
        初始化加载器
        
        Args:
            name (str): 数据集的名称/类别（如"Electronics", "Books"等）
        """
        self.name = name        # 数据集名称
        self.dataset = None     # 将要加载的数据集对象

    def from_datapoint(self, datapoint):
        """
        从单个数据点创建Item对象
        
        Args:
            datapoint (dict): 包含商品信息的字典
            
        Returns:
            Item: 如果数据有效且符合条件，返回Item对象；否则返回None
        """
        try:
            # 尝试获取价格字符串
            price_str = datapoint['price']
            if price_str:  # 如果价格字段不为空
                # 将价格字符串转换为浮点数
                price = float(price_str)
                # 检查价格是否在有效范围内
                if MIN_PRICE <= price <= MAX_PRICE:
                    # 创建Item对象
                    item = Item(datapoint, price)
                    # 只有当Item的include属性为True时才返回，否则返回None
                    return item if item.include else None
        except ValueError:
            # 如果价格转换失败，返回None（跳过这个数据点）
            return None

    def from_chunk(self, chunk):
        """
        从数据块中创建Item列表
        
        Args:
            chunk: 数据集的一个子集（包含多个数据点）
            
        Returns:
            list: 包含有效Item对象的列表
        """
        batch = []  # 用于存储处理后的Item对象
        # 遍历数据块中的每个数据点
        for datapoint in chunk:
            result = self.from_datapoint(datapoint)
            if result:  # 如果成功创建了Item对象
                batch.append(result)
        return batch

    def chunk_generator(self):
        """
        数据块生成器
        将大数据集分割成小块，便于批量处理和内存管理
        
        Yields:
            Dataset: 每次产出一个包含CHUNK_SIZE个数据点的数据集子集
        """
        size = len(self.dataset)  # 获取数据集总大小
        # 按照CHUNK_SIZE的步长遍历数据集
        for i in range(0, size, CHUNK_SIZE):
            # 选择从i到i+CHUNK_SIZE的数据点（确保不超出数据集边界）
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers):
        """
        使用多进程并行处理数据块
        显著提高处理速度，但会占用较多计算资源
        
        Args:
            workers (int): 并行工作进程的数量
            
        Returns:
            list: 包含所有处理后Item对象的列表
        """
        results = []  # 存储所有处理结果
        
        # 计算总的数据块数量（向上取整）
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        
        # 创建进程池执行器
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # 使用进程池并行处理所有数据块，并显示进度条
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), 
                            total=chunk_count):
                results.extend(batch)  # 将每个批次的结果添加到总结果中
        
        # 为所有Item对象设置类别名称
        for result in results:
            result.category = self.name
            
        return results
            
    def load(self, workers=8):
        """
        加载数据集的主方法
        
        Args:
            workers (int): 并行处理的工作进程数，默认为8
            
        Returns:
            list: 包含所有处理后Item对象的列表
        """
        start = datetime.now()  # 记录开始时间
        
        print(f"Loading dataset {self.name}", flush=True)
        
        # 从Hugging Face加载Amazon Reviews 2023数据集的特定类别
        # raw_meta_{self.name}表示加载该类别的原始元数据
        self.dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", 
            f"raw_meta_{self.name}", 
            split="full", 
            trust_remote_code=True
        )
        
        # 使用并行处理加载数据
        results = self.load_in_parallel(workers)
        
        finish = datetime.now()  # 记录结束时间
        
        # 输出处理结果统计
        elapsed_minutes = (finish - start).total_seconds() / 60
        print(f"Completed {self.name} with {len(results):,} datapoints in {elapsed_minutes:.1f} mins", 
              flush=True)
        
        return results