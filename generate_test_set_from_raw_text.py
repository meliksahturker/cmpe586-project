import pandas as pd
import numpy as np

from utils_noise_generator import noise_generator


lines = []
for file in ['0_boun_nlp_project', '2_bilkent_turkish_writings', '4_my_dear_watson_turkish_sentences',
             '5_ted_talks_tr', '6_tr2en_mt_machine_translation', '7_old_newspaper']:
    with open(f'data/{file}.txt', encoding = 'utf-8') as f:
        for line in f:
            line_strip = line.strip()
            if line_strip:
                lines.append(line_strip)
num_lines = len(lines)
print(f'Number of lines: {num_lines:,}')

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
df['ds'] = 'generic'
df.to_parquet('test_set_generic_5k.prq')