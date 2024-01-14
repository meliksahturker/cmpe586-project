import json

import numpy as np
import tensorflow as tf
import sentencepiece as spm

from datasets import load_from_disk

from utils_noise_generator import noise_generator

# 0.97 quantile: 252, 0.80 quantile: 132
# Read Config
with open('config.json', 'r') as openfile:
    config = json.load(openfile)

SOURCE_LEN = config['SOURCE_LEN']
TARGET_LEN = config['TARGET_LEN']


# Tokenizer
tokenizer = spm.SentencePieceProcessor('char_tokenizer/char_tokenizer.model')
sp_key_to_index = {tokenizer.id_to_piece(id): id for id in range(tokenizer.get_piece_size())}
sp_index_to_key = {id: tokenizer.id_to_piece(id) for id in range(tokenizer.get_piece_size())}


# Data
"""
lines = []
for file in ['0_boun_nlp_project', '2_bilkent_turkish_writings', '4_my_dear_watson_turkish_sentences',
             '5_ted_talks_tr', '6_tr2en_mt_machine_translation', '7_old_newspaper']:
    with open(f'data/{file}.txt', encoding = 'utf-8') as f:
        for line in f:
            line_strip = line.strip()
            if line_strip:
           print('Number of lines:', len(lines))ines.append(line_strip)
print('Number of lines:', len(lines))
"""

dataset = load_from_disk('book_corpus_v2')
lines = []
for idx in range(len(dataset['train'])):
    document = dataset['train'][idx]['text']
    # excluding first and last sentences as they may contain noise, author name, etc.
    lines += document.split('\n')[5:-5]
print('Number of lines:', len(lines))

def data_generator():
    while True:
        idx = np.random.randint(0, len(lines))
        
        sentence = lines[idx]
        corrupted_sentence = noise_generator(sentence)
        
        sentence_tokenized = tokenizer.encode_as_ids(sentence)
        corrupted_sentence_tokenized = tokenizer.encode_as_ids(corrupted_sentence)
        sentence_tokenized.insert(0, 2)
        sentence_tokenized.append(3)
        
        encoder_inputs = tf.keras.preprocessing.sequence.pad_sequences([corrupted_sentence_tokenized], maxlen = SOURCE_LEN, 
                                                                         padding = 'post', truncating = 'post')[0]
        decoder_inputs_outputs = tf.keras.preprocessing.sequence.pad_sequences([sentence_tokenized],
                                                                                  maxlen = TARGET_LEN + 1, 
                                                                                  padding = 'post', truncating = 'post')[0]
        
        decoder_inputs = decoder_inputs_outputs[:-1]
        decoder_outputs = decoder_inputs_outputs[1:]

        yield (encoder_inputs, decoder_inputs), decoder_outputs


def get_tf_data_generator(batch_size, num_parallel_calls):
    tf_dataset = tf.data.Dataset.from_generator(data_generator, 
                                            output_types = ((tf.int32, tf.int32), tf.int32),
                                            output_shapes = ((SOURCE_LEN, TARGET_LEN), TARGET_LEN))

    tf_dataset = tf_dataset.batch(batch_size, num_parallel_calls = num_parallel_calls).prefetch(num_parallel_calls)

    return tf_dataset