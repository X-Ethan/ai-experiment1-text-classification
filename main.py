import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data():
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X = train_df['text'].astype(str).tolist()
    y = train_df['target'].values
    X_test = test_df['text'].astype(str).tolist()
    return X, y, X_test

print("=== 加载数据 ===")
X, y, X_test = load_data()
print(f"训练集: {len(X)}, 测试集: {len(X_test)}")

# 划分验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
)
print(f"训练子集: {len(X_train)}, 验证集: {len(X_val)}")

print("\n=== TF-IDF 特征提取 ===")
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    stop_words='english',
    sublinear_tf=True
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)
print(f"特征维度: {X_train_tfidf.shape[1]}")

results = {}

# 1. 朴素贝叶斯
print("\n=== MultinomialNB ===")
for alpha in [0.1, 0.5, 1.0]:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train_tfidf, y_train)
    acc = accuracy_score(y_val, model.predict(X_val_tfidf))
    results[f'NB_alpha={alpha}'] = acc
    print(f"  alpha={alpha}: {acc:.4f}")

# 2. 逻辑回归
print("\n=== Logistic Regression ===")
for C in [1.0, 5.0, 10.0]:
    model = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train_tfidf, y_train)
    acc = accuracy_score(y_val, model.predict(X_val_tfidf))
    results[f'LR_C={C}'] = acc
    print(f"  C={C}: {acc:.4f}")

# 3. LinearSVC
print("\n=== LinearSVC ===")
for C in [0.5, 1.0, 2.0]:
    model = LinearSVC(C=C, random_state=RANDOM_SEED, max_iter=2000)
    model.fit(X_train_tfidf, y_train)
    acc = accuracy_score(y_val, model.predict(X_val_tfidf))
    results[f'LinearSVC_C={C}'] = acc
    print(f"  C={C}: {acc:.4f}")

print("\n=== 验证集结果汇总 ===")
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{k}: {v:.4f}")

# 选最优模型，在全部训练数据上重新训练
best_name = max(results, key=results.get)
print(f"\n最优配置: {best_name}")

print("用全部训练数据重新训练最优模型...")
vectorizer_full = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    stop_words='english',
    sublinear_tf=True
)
X_all_tfidf = vectorizer_full.fit_transform(X)
X_test_final = vectorizer_full.transform(X_test)

if best_name.startswith('NB'):
    alpha = float(best_name.split('=')[1])
    final_model = MultinomialNB(alpha=alpha)
elif best_name.startswith('LR'):
    C = float(best_name.split('=')[1])
    final_model = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_SEED, n_jobs=-1)
else:
    C = float(best_name.split('=')[1])
    final_model = LinearSVC(C=C, random_state=RANDOM_SEED, max_iter=2000)

final_model.fit(X_all_tfidf, y)
predictions = final_model.predict(X_test_final)

pd.DataFrame(predictions).to_csv('predictions.csv', index=False, header=False)
print(f"已保存 predictions.csv，共 {len(predictions)} 条预测")
print("前10条预测:", predictions[:10])
