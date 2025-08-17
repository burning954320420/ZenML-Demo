### Decision Drivers

- 易用性与上手门槛
- 可扩展性与生产部署能力
- 与现有基础设施的兼容性与集成能力



###  Context

- **Kubeflow OSS** 是一个 Kubernetes 原生的开源平台，提供从模型开发、训练（Kubeflow Pipelines、Training Operator）、自动化调参（Katib）、Serving（KServe）等完整 MLOps 功能
- **ZenML OSS** 则是一个轻量级、用户友好的 MLOps 框架，重点在简化 ML pipeline 创建与管理，可在本地、云端或 Kubernetes 上部署，同时支持与 Kubeflow、Airflow、SageMaker 等后端集成
- ZenML 提供简洁的 Python API，可以与其它实验跟踪工具进行整合（如 MLflow），并支持多种执行后端，但本身不原生实现训练功能
- Kubeflow 在 Kubernetes 环境中功能丰富但上手成本高、维护复杂；ZenML 更适合中小团队或希望快速搭建 MLOps Pipelines 的组织



### Options

#### A: 使用 **Kubeflow OSS** 全栈平台

**优点**

- 完整 MLOps 功能覆盖：Notebook、训练、自动调参、Serving、调度器等
- Kubernetes 原生支持，适合大规模、容器化部署场景

**缺点**

- 高学习曲线，需要显著的 Kubernetes 和 infra 管理经验
- 维护复杂，尤其在小团队或资源有限的场景下操作成本高

------

#### B: 采用 **ZenML**

**优点**

- 用户友好：Python API，易上手
- 灵活：可以在不同后端之间切换，如本地、Kubernetes、Kubeflow、Airflow 等
- 易集成：支持实验跟踪工具（MLflow 等） ([DagsHub](https://dagshub.com/blog/best-machine-learning-workflow-and-pipeline-orchestration-tools/?utm_source=chatgpt.com))
- 更快实现生产化，与多种工具协同更自然

**缺点**

- 本身不提供完整训练服务、Serving 功能，依赖底层平台
- 分层结构稍复杂，需要额外维护 ZenML 与后端的整合



### Decision Outcome

在团队需要快速构建可复用、易修改的 ML pipelines，并希望保持对底层执行基础设施的灵活选择能力时，ZenML  是更合适的方案。

