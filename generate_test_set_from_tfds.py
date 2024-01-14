import pandas as pd
import numpy as np

from vnlp import SentenceSplitter
import tensorflow_datasets as tfds

from utils_noise_generator import noise_generator

dataset_name = 'book_corpus_v2'
ds = tfds.load(name = dataset_name, data_dir = f'{dataset_name}')

texts = []
for split in ['train', 'validation']:
    for item in ds[split]:
        article = item['article'].numpy().decode('utf-8').strip('.txt')
        text = item['text'].numpy().decode('utf-8')

        texts.append((article, text))

print('Number of documents:', len(texts))

ss = SentenceSplitter()

lines = []
for text in texts[:5]: # use the first 5 books only
    lines += ss.split_sentences(text[1])
print(len(lines))


np.random.seed(43)
sample_indices = np.random.choice(range(len(lines)), size = 5_000, replace = False)
sample_lines = [line for idx, line in enumerate(lines) if idx in sample_indices]

ds = []
for clean_sentence in sample_lines:
    noised_sentence, noise_type = noise_generator(clean_sentence, return_noise_type = True)
    
    # since this is the TEST dataset, we want to make sure that there IS some noise
    # in every sample, contrary to the train set.
    # so make sure clean and noised sentences are not the same.
    while clean_sentence == noised_sentence:
        noised_sentence, noise_type = noise_generator(clean_sentence, return_noise_type = True)
        
    ds.append((clean_sentence, noised_sentence, noise_type))

df = pd.DataFrame(ds, columns = ['clean', 'noised', 'noise_type'])
df['ds'] = 'book'
df.to_parquet('test_set_books_5k.prq')
