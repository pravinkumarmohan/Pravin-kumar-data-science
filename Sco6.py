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
