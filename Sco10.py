df = pd.read_csv('/content/fake_news_dataset.csv')

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
