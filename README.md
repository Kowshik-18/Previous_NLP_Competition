# ETE-ML-Competition

This project addresses a classification problem where the dataset provided is highly diverse. Given a book summary, the goal is to classify the genre of the book. We have primarily fine-tuned the `csebuetnlp/banglabert` model for classification.

## Data Preprocessing

After performing exploratory data analysis, we found that:
- The maximum number of words in a sentence is **10,192**.
- The average number of words in a sentence is **733**.

In the preprocessing step, the following actions were performed:
- **Removed**: hashtags, digits (both Bangla and English), occurrences of ZWNJ or ZWJ, all types of punctuations, and emojis.
- **Normalized**: Used `bnunicodenormalizer` and `unicodedata` library for normalization.
- Replaced **'ঃ'** with a space as it concatenated many words.
- **Lowercased** all English words.
- **Removed stopwords**: Collected from the `bltk` and `nltk` libraries for both Bangla and English text.
- **Excluded frequent words**: Removed the 10 most frequent words from the corpus.

After preprocessing, the average word count in the dataset was reduced to **542**.

## Fine-tuning Process

We fine-tuned the `csebuetnlp/banglabert` model using HuggingFace's `Trainer` API in Google Colab. The dataset was split into 90% for training and 10% for validation. The hyperparameters used during training are:

- `learning_rate`: `6e-5`
- `weight_decay`: `5e-7`
- `num_train_epochs`: `5`
- `evaluation_strategy`: `'epoch'`
- `optim`: `'adafactor'`

The model with the lowest validation loss was saved as the best checkpoint. The fine-tuned model is available on HuggingFace Hub at:  
`Udoy/test-trainer-withoutMF-0-2`. (Udoy Das => Teammate)

## Inference

The fine-tuned model is accessible via the HuggingFace Hub and was used to make predictions on the provided test set.

## Results

- **Public Score**: `0.87607`
- **Private Score**: `0.83187`
