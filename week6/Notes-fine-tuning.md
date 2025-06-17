## Digging into the data

six steps to process data roughly:

1. Investigate
2. Parse
3. Visualize
4. Assess Data Quality
5. Curate
6. Save

## 5 step strategy

To selecting, training, and applying an LLM to a commercial problem

### 1. Understand

- Gather business requirements for the task
- Identify performance criteria
  - Particularly the Business Centric metrics
- Understand the data: quantity, quality, format
- Determine non-functionals
  - Cost constraints, scalability, latency
  - R&D/build budget and implementation timeline

### 2. Prepare

- Research existing/non-LLM solutions
  - Potential baseline model
- Compare relevant LLMs
  - The basics,including context length, price and license
  - Benchmarks, Leaderboards and Arenas
  - Specialist scores for the task at hand
- Curate data:clean, preprocess and split

### 3. Select

- Choose LLM(s)
- Experiment
- Train and validate with curated data

### 4. Customize

- Prompting, multi-shot, chaining and tools
- RAG
- Fine-tuning

|             | Pros                                                                                                                             | Cons                                                                                                                       |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Prompting   | 1. Fast to implement <br>2. Low cost <br>3. Often immediate improvement                                                          | 1. Limited by context length <br> 2. Diminishing returns <br> 3. Slower, more expensive inference                          |
| RAG         | 1. Accuracy improvement with low data needs <br>2. Scalable <br> 3. Efficient                                                    | 1. Harder to implement <br> 2. Requires up-to-date accurate data <br> 3. Lacks nuance                                      |
| Fine-tuning | 1. Deep expertise & specialist knowledge <br>2. Nuance <br> 3. Learn a different tone/style <br> 4. Faster and cheaper inference | 1. Significant effort to implement <br> 2. High data needs <br> 3. Training cost <br> 4. Risk of "catastrophic forgetting" |

### 5. Productionize

- Determine API between model and platform(s)
- Identify model hosting and deployment architecture
- Address scaling, monitoring, security and compliance
- Measure the Business-Focused Metrics identified in step 1
- Continuously retrain and measure performance
