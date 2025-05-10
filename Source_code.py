import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/content/fake_news_dataset.csv")

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("/content/fake_news_dataset.csv")

# Show basic info
print("🔍 Dataset Info:")
print(df.info())

# Show first 5 rows
print("\n📝 First 5 Records:")
print(df.head())

# Show unique label values
print("\n🧾 Unique label values (before cleaning):")
print(df['label'].unique())

# Clean label values (normalize)
df['label'] = df['label'].astype(str).str.strip().str.upper()

# Show cleaned unique label values
print("\n✅ Unique label values (after cleaning):")
print(df['label'].unique())

# Check label distribution
print("\n📊 Label Distribution:")
print(df['label'].value_counts())

# Plot label distribution
sns.countplot(x='label', data=df)
plt.title("Real vs Fake News Count")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()
# Check for missing values
print("\n❓ Missing Values:")
print(df.isnull().sum())

# Add column: article text length
df['text_length'] = df['text'].astype(str).apply(len)

# Plot text length distribution
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='text_length', hue='label', bins=50, kde=True)
plt.title("Article Length Distribution by Label")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.show()

# Word Cloud function with safety check
def show_wordcloud(data, title):
    if data.empty:
        print(f"⚠️ Skipping word cloud: No data available for '{title}'")
        return
    text = ' '.join(data.astype(str))
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Generate word clouds if data is available
print("\n☁️ Word Cloud Generation:")
show_wordcloud(df[df['label'] == 'FAKE']['text'], "Fake News Word Cloud")
show_wordcloud(df[df['label'] == 'REAL']['text'], "Real News Word Cloud")

print("📌 Missing Values in Each Column:\n")
print(df.isnull().sum())

# 📌 Total number of missing values
print("\n🧮 Total Missing Values:", df.isnull().sum().sum())

# 🔍 Check for duplicate rows
duplicate_rows = df.duplicated()
num_duplicates = duplicate_rows.sum()

print("\n📌 Total Duplicate Rows:", num_duplicates)

# Optional: View duplicate rows (uncomment if needed)
# print(df[df.duplicated()])

df['label'] = df['label'].astype(str).str.strip().str.upper()

# 📊 1. Label distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df, palette='Set2')
plt.title("Distribution of Fake vs Real News")
plt.xlabel("News Label")
plt.ylabel("Count")
plt.show()

# 📏 2. Article text length by label
df['text_length'] = df['text'].astype(str).apply(len)

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='text_length', hue='label', bins=50, kde=True, palette='husl')
plt.title("Text Length Distribution by Label")
plt.xlabel("Text Length (number of characters)")
plt.ylabel("Frequency")
plt.show()

# 🧾 3. (Optional) Top 10 frequent words in all text
from sklearn.feature_extraction.text import CountVectorizer

# Basic preprocessing: drop missing text values
df = df.dropna(subset=['text'])

vectorizer = CountVectorizer(stop_words='english', max_features=10)
X = vectorizer.fit_transform(df['text'].astype(str))

word_freq = X.toarray().sum(axis=0)
words = vectorizer.get_feature_names_out()

plt.figure(figsize=(8, 4))
sns.barplot(x=word_freq, y=words, palette='coolwarm')
plt.title("Top 10 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()

df['label'] = df['label'].astype(str).str.strip().str.upper()

# 📊 1. Label distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df, palette='Set2')
plt.title("Distribution of Fake vs Real News")
plt.xlabel("News Label")
plt.ylabel("Count")
plt.show()

# 📏 2. Article text length by label
df['text_length'] = df['text'].astype(str).apply(len)

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='text_length', hue='label', bins=50, kde=True, palette='husl')
plt.title("Text Length Distribution by Label")
plt.xlabel("Text Length (number of characters)")
plt.ylabel("Frequency")

plt Show all column names
print("🧾 Column Names:")
print(df.columns.tolist())

# Identify target column
target_col = 'label'
print("\n🎯 Target Variable:")
print(target_col)

# Identify feature columns (drop target)
feature_cols = df.drop(columns=[target_col]).columns.tolist()
print("\n🧰 Feature Columns:")
print(feature_cols).show()


print("📌 Original Column Types:\n")
print(df.dtypes)

# Convert 'label' to numeric if needed
df['label'] = df['label'].astype(str).str.strip().str.upper()
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# Find object (categorical) columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\n🔍 Categorical Columns to Convert:", cat_cols)

# Drop 'text' and 'title' from encoding (they are too long to encode directly)
cat_cols = [col for col in cat_cols if col not in ['text', 'title']]

# Convert remaining categorical columns using one-hot encoding
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Show final structure
print("\n✅ Data After Encoding:")
print(df_encoded.head())


print("Dataset Head:\n", df.head())
print("\nColumns:", df.columns.tolist())

# 3. Optional: Drop unused or irrelevant columns if needed
# Example: if you have an unnamed index column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# 4. Add derived features
df['char_count'] = df['text'].astype(str).apply(len)

# 5. Handle missing values
df.fillna("unknown", inplace=True)

# 6. Encode target label if it's categorical
if df['label'].dtype == 'object':
    df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# Show final preview
print("\nProcessed DataFrame sample:")
print(df[['text', 'char_count', 'label']].head())


f = pd.read_csv('/content/fake_news_dataset.csv')

# 2. Preprocessing
# Add a feature for character count in the text
df['char_count'] = df['text'].apply(len)

# Handle missing values
df.fillna("unknown", inplace=True)

# Encode the target label 'label' (FAKE -> 0, REAL -> 1)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Split the data into features (X) and target (y)
X = df[['text', 'char_count']]  # Text and character count
y = df['label']

# 3. Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Text Vectorization and Model Pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),   # Converts text to word count vectors
    ('classifier', MultinomialNB())      # Naive Bayes classifier
])

# Train the model using the pipeline
pipeline.fit(X_train['text'], y_train)

# 5. Model Evaluation
def evaluate_model(X_test, y_test, model_pipeline):
    # Predict the labels on the test set
    y_pred = model_pipeline.predict(X_test['text'])

    # Print evaluation metrics
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Evaluate the model
evaluate_model(X_test, y_test, pipeline)
