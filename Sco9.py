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
