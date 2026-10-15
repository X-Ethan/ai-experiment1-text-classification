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
