## LangChain Expression Language

**LCEL is adeclarativelanguage that can be used as an alternative to the code approach**

- Describe what you want to achieve in a YAML file
- Arguably not much easier than coding directly

## Behind the curtain

Understanding how LangChain works, and identifying & fixing common problems

![LangChain.png](HowLangChainWorks.png)

## `as_retriever` 方法

`as_retriever` 方法是一个用于将 `VectorStore` 对象转换为 `VectorStoreRetriever` 对象的便捷方法。`VectorStoreRetriever` 是一个检索类，用于从向量存储中查找和检索最相关的文档。这个方法接受多个可选参数来配置检索器的行为。

### 用法介绍

### 参数

- **search_type (Optional[str])**:

  - 定义检索器应该执行的搜索类型。
  - 选项包括:
    - `"similarity"`: 基于相似度的搜索。
    - `"mmr"`: 基于最大边际相关性 (Maximal Marginal Relevance) 的搜索。
    - `"similarity_score_threshold"`: 基于相似度分数阈值的搜索。

- **search_kwargs (Optional[Dict])**:
  - 传递给搜索函数的关键字参数，可能包括:
    - `k`: 要返回的文档数量 (默认值: 4)。
    - `score_threshold`: 用于 `similarity_score_threshold` 的最低相关性阈值。
    - `fetch_k`: 传递给 MMR 算法的文档数量 (默认值: 20)。
    - `lambda_mult`: MMR 返回的多样性: 1 为最小多样性，0 为最大多样性 (默认值: 0.5)。
    - `filter`: 根据文档元数据进行过滤。

### 返回值

- `VectorStoreRetriever`: 为 `VectorStore` 初始化的检索类。
