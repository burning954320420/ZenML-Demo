# ZenML-Demo: CPU异常检测系统

基于ZenML构建的CPU使用率异常检测系统，实现了完整的机器学习工作流程，包括数据生成、特征工程、模型训练、推理和可视化。

## 项目结构

```
ZenML-Demo/
├── configs/                # 配置文件
│   └── config.py           # 全局配置
├── create_data/            # 数据生成模块
│   └── create_cpu_timestamp_data.py  # CPU数据生成器
├── data/                   # 数据目录
├── models/                 # 模型保存目录
├── pipelines/              # ZenML管道定义
│   ├── training_pipeline.py  # 训练管道
│   └── inference_pipeline.py # 推理管道
├── steps/                  # ZenML步骤定义
│   ├── data_loader.py      # 数据加载
│   ├── data_validation.py  # 数据验证
│   ├── feature_engineering.py  # 特征工程
│   ├── model_trainer.py    # 模型训练
│   ├── model_saver.py      # 模型保存
│   └── model_predictor.py  # 模型预测
├── run.py                  # 训练执行脚本
```

## 功能特点

- **数据生成**：创建带有时间戳的CPU使用率模拟数据，包含多种类型的异常模式
- **数据验证**：自动检查数据质量，确保训练数据符合要求
- **特征工程**：构建丰富的时间序列特征，包括统计特征、滑动窗口特征和算法特征
- **模型训练**：使用随机森林算法训练异常检测模型，包含超参数调优
- **模型评估**：全面评估模型性能，包括混淆矩阵、ROC曲线和PR曲线
- **模型推理**：使用训练好的模型进行异常检测

## 快速开始

### 环境准备

1. 安装依赖：

```bash
pip install zenml pandas numpy scikit-learn matplotlib seaborn joblib
```

2. 初始化ZenML（可选）：

```bash
zenml init
```

这将依次执行：
1. 生成模拟CPU数据
2. 训练异常检测模型
3. 使用模型进行推理

### 单独执行各个步骤

#### 1. 生成模拟数据

```bash
python -c "from create_data.create_cpu_timestamp_data import CPUDataGenerator; CPUDataGenerator().generate_dataset('data/cpu_data.csv', 'data/cpu_data.txt')"
```

#### 2. 训练模型

```bash
python run.py
```


## 自定义配置

- 修改 `configs/config.py` 更改默认数据路径和模型保存路径
- 在 `run.py` 中可以传入自定义参数

## 扩展功能

### 添加新的异常类型

修改 `create_data/create_cpu_timestamp_data.py` 中的 `CPUDataGenerator` 类，添加新的异常模式生成方法。

### 添加新的特征

修改 `steps/feature_engineering.py` 中的 `CPUFeatureEngineer` 类，添加新的特征创建方法。

### 使用不同的模型

修改 `steps/model_trainer.py`，替换 `RandomForestClassifier` 为其他模型。

