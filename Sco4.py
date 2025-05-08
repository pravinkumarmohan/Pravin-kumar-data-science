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
