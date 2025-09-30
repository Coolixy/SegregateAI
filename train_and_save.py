import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import os

# --- 1) Load CSVs -------------------------------------------------
def load_data(train_path='waste_train.csv', test_path='waste_test.csv'):
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)
    df_train.columns = df_train.columns.str.strip().str.lower()
    df_test.columns  = df_test.columns.str.strip().str.lower()
    X_train = df_train['text'].astype(str).tolist()
    y_train = df_train['label'].tolist()
    X_val   = df_test['text'].astype(str).tolist()
    y_val   = df_test['label'].tolist()
    return (X_train, y_train), (X_val, y_val)

# --- 2) Encode labels ---------------------------------------------
def encode_labels(labels):
    mapping = {'biodegradable':0, 'nonbiodegradable':1, 'other':2}
    out = []
    for lab in labels:
        clean = lab.strip().lower().replace('_','').replace('-','')
        if clean not in mapping:
            raise ValueError(f"Unknown label {lab!r}")
        out.append(mapping[clean])
    return np.array(out, dtype=np.int32)

# --- 3) Prepare tokenizer & sequences ----------------------------
def prepare_tokenizer(texts, num_words=10000, oov_token="<UNK>"):
    tok = Tokenizer(num_words=num_words, oov_token=oov_token)
    tok.fit_on_texts(texts)
    return tok

def texts_to_padded_sequences(tok, texts, max_len=20):
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')

# --- 4) Build the model -------------------------------------------
def build_model(vocab_size, embed_dim=16, max_len=20):
    inp = Input(shape=(max_len,), dtype='int32', name='input_seq')
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,
                         input_length=max_len, mask_zero=True,
                         name='embedding')(inp)
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(16, activation='relu', name='dense_relu')(x)
    out = layers.Dense(3, activation='softmax', name='predictions')(x)
    model = Model(inputs=inp, outputs=out, name='waste_classifier')
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 5) Train, plot, save model + tokenizer ----------------------
def train_plot_save(epochs=10,
                    batch_size=32,
                    num_words=10000,
                    max_len=20,
                    tokenizer_path='tokenizer.pkl',
                    model_path='waste_classifier_model.keras'):

    # Load & encode
    (X_train, y_train_text), (X_val, y_val_text) = load_data()
    y_train = encode_labels(y_train_text)
    y_val   = encode_labels(y_val_text)

    # Tokenizer
    tokenizer = prepare_tokenizer(X_train, num_words=num_words)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Saved tokenizer to {tokenizer_path}")

    # Sequences + padding
    X_train_seq = texts_to_padded_sequences(tokenizer, X_train, max_len=max_len)
    X_val_seq   = texts_to_padded_sequences(tokenizer, X_val,   max_len=max_len)

    # Build & train model
    model = build_model(vocab_size=num_words, embed_dim=16, max_len=max_len)
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    # Plot
    plt.plot(history.history['accuracy'],    label='train_acc')
    plt.plot(history.history['val_accuracy'],label='val_acc')
    plt.plot(history.history['loss'],        label='train_loss')
    plt.plot(history.history['val_loss'],    label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Value')
    plt.legend(); plt.title('Training & Validation Metrics')
    plt.show()

    # Save model in SavedModel format
    model.save("waste_classifier_model.keras", save_format="keras")

  # This creates a folder
    print(f"✅ Saved model in SavedModel format to {model_path}/")

# --- Entry point --------------------------------------------------
if __name__ == '__main__':
    train_plot_save()