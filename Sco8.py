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
