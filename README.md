# 当代人工智能实验一：文本分类

## 实验简介
本实验实现新闻文本 10 分类任务，使用 TF-IDF 特征提取，并对比朴素贝叶斯、逻辑回归、LinearSVC 三种模型在验证集上的表现，最终选择最优模型对测试集进行预测。

## 数据说明
- `train_data.csv`：训练集（7368 条，含标签）
- `test_data_unlabeled.csv`：测试集（2457 条，无标签）

## 运行环境
- Python 3.10
- numpy / pandas / scikit-learn

## 运行方法

```bash
conda create -n project1 python=3.10 -y
conda activate project1
conda install numpy pandas scikit-learn -y
python main.py
```

运行完成后会自动生成 `predictions.csv`（无表头，共 2457 条预测结果）。

## 实验设置
- 验证集比例：15%（分层抽样，随机种子 42）
- TF-IDF：max_features=20000，ngram_range=(1,2)，英文停用词，sublinear_tf
- 最终选择模型：LinearSVC (C=0.5)，验证集准确率 0.9421
