import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Загрузка ресурсов nltk (если не установлены)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# ---------------------------
# 1. Гибкая загрузка датасета
# ---------------------------
def load_dataset(filepath):
    # Пробуем стандартную загрузку (разделитель ',')
    df = pd.read_csv(filepath, encoding='utf-8')
    if df.shape[1] > 1:
        return df
    # Если один столбец, пробуем другие разделители
    for sep in ['\t', ';', '|']:
        df = pd.read_csv(filepath, sep=sep, encoding='utf-8')
        if df.shape[1] > 1:
            print(f"Успешно загружено с разделителем '{sep}'")
            return df
    # Если всё ещё один столбец, возможно, файл без заголовков
    df = pd.read_csv(filepath, header=None, encoding='utf-8')
    if df.shape[1] == 2:
        df.columns = ['text', 'label']
        return df
    else:
        raise ValueError("Не удалось определить структуру файла. Проверьте разделитель.")

# Загружаем
df = load_dataset('medical_llm_dataset.csv')
print("Размер датасета:", df.shape)
print("Столбцы:", df.columns.tolist())
print("Первые 5 строк:")
print(df.head())

# Если в данных есть лишние столбцы, оставляем только text и label
if 'text' not in df.columns or 'label' not in df.columns:
    # предположим, что первый столбец — текст, второй — метка
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
        df.columns = ['text', 'label']
    else:
        raise KeyError("В данных нет столбцов 'text' и 'label'")

# Удаляем возможные пропуски
df = df.dropna(subset=['text', 'label']).reset_index(drop=True)

print("Распределение классов:\n", df['label'].value_counts())

# ---------------------------
# 1b. Распределение классов (текстом, по убыванию)
# ---------------------------
dist_col = 'label_enc' if 'label_enc' in df.columns else 'label'
class_counts = df[dist_col].astype(str).value_counts().sort_values(ascending=False)
print(f"\nРаспределение классов по убыванию ({dist_col}):")
print(class_counts.to_string())

# ---------------------------
# 2. Предобработка текста
# ---------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)   # удаление пунктуации
    text = re.sub(r'\d+', ' ', text)       # удаление цифр
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(preprocess_text)

# ---------------------------
# 2b. Выбор и кодирование ЦЕЛЕВОЙ метки
# ---------------------------
# В этом датасете "настоящая" метка (название диагноза) находится в label_enc.
# Столбец label часто бывает просто числовым id класса, служебным индексом и т.п.
TARGET_COL = 'label_enc' if 'label_enc' in df.columns else 'label'
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df[TARGET_COL].astype(str))
num_classes = len(le.classes_)
print(f"Целевая метка: {TARGET_COL}")
print(f"Количество классов: {num_classes}")

# ---------------------------
# 3. ML модель (Random Forest)
# ---------------------------
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest ---")
print(f"Precision (macro): {precision_score(y_test, y_pred_rf, average='macro', zero_division=0):.4f}")
print(f"Recall (macro):    {recall_score(y_test, y_pred_rf, average='macro', zero_division=0):.4f}")
print(f"F1 (macro):        {f1_score(y_test, y_pred_rf, average='macro', zero_division=0):.4f}")

# Матрица ошибок
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show(block=False)
plt.pause(0.1)

# ---------------------------
# 4. Нейросетевая модель
# ---------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TextVectorization, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

MAX_WORDS = 50000

X_text = df['cleaned_text'].astype(str).values
y_arr = y.values if hasattr(y, "values") else np.asarray(y)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y_arr, test_size=0.2, random_state=42, stratify=y_arr
)

text_vectorizer = TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode='tf-idf',
    ngrams=2,
)
text_vectorizer.adapt(X_train_text)

model = Sequential([
    text_vectorizer,
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train_text, y_train,
                    validation_split=0.2,
                    epochs=5,
                    batch_size=128,
                    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                    verbose=1)

y_pred_nn = np.argmax(model.predict(X_test_text), axis=1)
print("\n--- Neural Network ---")
print(f"Precision (macro): {precision_score(y_test, y_pred_nn, average='macro', zero_division=0):.4f}")
print(f"Recall (macro):    {recall_score(y_test, y_pred_nn, average='macro', zero_division=0):.4f}")
print(f"F1 (macro):        {f1_score(y_test, y_pred_nn, average='macro', zero_division=0):.4f}")

# Матрица ошибок
cm_nn = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Neural Network')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show(block=False)
plt.pause(0.1)

# ---------------------------
# 5. Инференс: классификация своего текста
# ---------------------------
def predict_symptom_text(user_text, top_k = 3):
    cleaned = preprocess_text(user_text)

    # NN prediction
    probs = model.predict(tf.constant([cleaned], dtype=tf.string), verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    top_labels = le.inverse_transform(top_idx)
    top_probs = probs[top_idx]


    return {
        "cleaned_text": cleaned,
        "nn_top": list(zip(top_labels.tolist(), top_probs.tolist())),
    }

print("\nВведи текст симптомов. Для выхода введи :q или :exit")
while True:
    user_text = input("\nСимптомы: ").strip()
    if user_text.lower() in {":q", ":exit"}:
        break
    result = predict_symptom_text(user_text, top_k=3)
    print(f"Cleaned: {result['cleaned_text']}")
    print("NN top-3:")
    for label_name, p in result["nn_top"]:
        print(f"  - {label_name}: {p:.4f}")