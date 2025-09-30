# predict_with_keywords.py

import pandas as pd

# Load dataset
df = pd.read_csv('data.csv')
df.columns = ['description', 'label']
df.dropna(inplace=True)

# Preprocess into a keyword-label dictionary
keyword_map = {}

for _, row in df.iterrows():
    words = str(row['description']).lower().split()
    for word in words:
        if word not in keyword_map:
            keyword_map[word] = set()
        keyword_map[word].add(row['label'])

# Prediction function
def keyword_match_predict(input_text: str):
    input_words = set(input_text.lower().split())
    matched_labels = set()

    for word in input_words:
        if word in keyword_map:
            matched_labels.update(keyword_map[word])

    if matched_labels:
        return matched_labels
    return {"unknown"}

# Interactive test
if __name__ == "__main__":
    while True:
        desc = input("Enter a food item description (or 'exit'): ").strip()
        if desc.lower() == 'exit':
            break
        labels = keyword_match_predict(desc)
        print(f"Possible Categories Based on Keywords: {', '.join(labels)}\n")
