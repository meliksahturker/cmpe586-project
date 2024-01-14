import string
import random

import numpy as np

import nlpaug.augmenter.char as nac


# Noise Generator Config
# Meliksah
unchanged_ratio = 0.10 # to prevent over-pessimistic model that always to changes and corrects everything.
all_lowercase_ratio = 0.10
remove_punctuations_ratio = 0.10
stt_output_ratio = 0.10 # Simulating STT output: both all_lowercase and remove punctuations. Note: this may also help sentence segmentation.
shuffle_chars_ratio = 0.10 # Randomly select a word and shuffle its characters

# NLPAUG
keyboard_errors_ratio = 0.15
ocr_errors_ratio = 0.15
random_error_ratio = 0.15

# ByT5
drop_random_char_ratio = random_error_ratio # Implemented by NLPAUG Random Error action delete.
repeat_ratio = 0.15
antspeak_ratio = 0.15
all_uppercase_ratio = 0.15
random_case_ratio = 0.15

# Class Instances
tokenizer = lambda x: x.split() # NLPAUG's default tokenizer acts like WordPunctTokenizer.
# Keyboard Errors
keyboard_aug = nac.KeyboardAug(lang = 'tr', tokenizer = tokenizer, aug_char_p = keyboard_errors_ratio, 
                               aug_word_p = keyboard_errors_ratio)
# OCR Errors
ocr_aug = nac.OcrAug(tokenizer = tokenizer, aug_char_p = ocr_errors_ratio,
                     aug_word_p = ocr_errors_ratio)

# Random Errors
random_aug_insert = nac.RandomCharAug(action='insert', tokenizer = tokenizer, aug_char_p = random_error_ratio,
                                      aug_word_p = random_error_ratio)
random_aug_substitute = nac.RandomCharAug(action='substitute', tokenizer = tokenizer, aug_char_p = random_error_ratio,
                                          aug_word_p = random_error_ratio)
random_aug_swap = nac.RandomCharAug(action='swap', tokenizer = tokenizer, aug_char_p = random_error_ratio,
                                    aug_word_p = random_error_ratio)

# ByT5 Errors
random_aug_delete = nac.RandomCharAug(action='delete', tokenizer = tokenizer, aug_char_p = random_error_ratio,
                                      aug_word_p = random_error_ratio)


def turkish_upper(text: str):
    return text.replace('i', 'İ').upper()

turkish_lowercase_dict = {"İ": "i", "I": "ı", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ş": "ş", "Ç": "ç"}
def turkish_lower(text):
    for k, v in turkish_lowercase_dict.items():
        text = text.replace(k, v)
    return text.lower()

def remove_punctuations(text):
    return "".join([char for char in text if not char in string.punctuation])

def shuffle_chars(word):
    return "".join(random.sample(word, len(word)))

# Split Word Randomly Noise
def _split_by_indices(word:str, num_splits:int, split_by:str):
    """
    Helper function used in split_noise().
    """
    word_len = len(word)
    
    split_locations = sorted([0] + random.sample(range(1, word_len), num_splits) + [word_len])
    splitted_word_parts = [word[split_locations[i]: split_locations[i+1]] for i in range(len(split_locations) - 1)]
    splitted_word = split_by.join(splitted_word_parts)
    
    return splitted_word

def split_noise(text:str, split_by:str):
    """
    text: input text
    split_by: the string to be used while splitting.
    """
    words = text.split()
    word_lens = [len(word) for word in words]
    word_len_greater_than_one_indices = np.where([word_len > 1 for word_len in word_lens])[0].tolist()
    
    if word_len_greater_than_one_indices:
        word_idx_to_noise = random.choice(word_len_greater_than_one_indices)
        word_to_noise = words[word_idx_to_noise]
        word_len = word_lens[word_idx_to_noise]

        if split_by == "-\n":
            num_splits = 1
        else:
            num_potential_splits = word_len // 2
            num_splits = random.randint(1, num_potential_splits)

        splitted_word = _split_by_indices(word_to_noise, num_splits, split_by)
        words[word_idx_to_noise] = splitted_word
        noised_text = " ".join(words)
        return noised_text
    
    else:
        return text
    
asciify_map = {'ç': 'c',
               'Ç': 'C',
               'ğ': 'g',
               'Ğ': 'G',
               'ı': 'i', # ı/I is an exceptional case for Turkish
               'ö': 'o',
               'Ö': 'O',
               'ş': 's',
               'Ş': 'S',
               'ü': 'u',
               'Ü': 'U'}
def turkish_asciify(text: str):
    for k,v in asciify_map.items():
        text = text.replace(k, v)
    return text

# ByT5 Errors
def generate_repeat_error(text: str):
    # I do not explicitly repeat more than once but it is left to probabilities.
    # The probability is consecutive repetition is p**2.
    text = list(text) # Convert to list for insertion
    text_len = len(text)
    probs = np.random.random(text_len)
    boolean_mask_to_repeat_indices = probs < repeat_ratio
    indices_to_repeat = np.where(boolean_mask_to_repeat_indices)[0]
    shifted_indices_to_repeat = indices_to_repeat + np.arange(len(indices_to_repeat)) # This is needed as repeating shifts the indices
    for index in shifted_indices_to_repeat: # TODO: can this be done without a for loop?
        text.insert(index, text[index])
    repeated_text = "".join(text)
    
    return repeated_text

def generate_antspeak_error(text: str):
    text = generate_all_uppercase_error(text)
    text = list(text) # Convert to list for insertion
    text_len = len(text)

    positions_to_insert_spaces = np.arange(1, text_len * 2 - 1, 2)
    for index in positions_to_insert_spaces:
        text.insert(index, ' ')
    antspeak_text = "".join(text)
    
    return antspeak_text

def generate_all_uppercase_error(text: str): # Python converts i to I so this is needed for Turkish.
    return turkish_upper(text)

def generate_random_case_error(text: str):
    mask = np.random.randint(0, 2, size = len(text))
    return "".join([turkish_upper(char) if mask_ == 1 else turkish_lower(char) for char, mask_ in zip(text, mask)])


byt5_noise_prob = 0.4
def generate_byt5_errors(text: str):
    """
    This schema allows multiple noises simultaneously
    or no noise at all with a probability of (1 - 0.4) ** 5 = 0.0777
    """
    # Make sure that the input is always noised.
    noised_text = None
    while not noised_text:
        if np.random.rand() < byt5_noise_prob:
            noise_type = 'byt5:deletion'
            noised_text = random_aug_delete.augment(text)[0]
        
        if np.random.rand() < byt5_noise_prob:
            noise_type = 'byt5:repeat'
            noised_text = generate_repeat_error(text)
            
        if np.random.rand() < byt5_noise_prob:
            noise_type = 'byt5:antspeak'
            noised_text = generate_antspeak_error(text)
        
        if np.random.rand() < byt5_noise_prob:
            noise_type = 'byt5:uppercase_all'
            noised_text = generate_all_uppercase_error(text)
            
        if np.random.rand() < byt5_noise_prob:
            noise_type = 'byt5:randomcase'
            noised_text = generate_random_case_error(text)
        
    return noised_text, noise_type


def generate_random_errors(text):
    # Make sure that the input is always noised.
    noised_text = None
    while not noised_text:
        if np.random.rand() < 0.5:
            noise_type = 'random:insertion'
            noised_text = random_aug_insert.augment(text, n = 1)[0]
        
        if np.random.rand() < 0.5:
            noise_type = 'random:substitution'
            noised_text = random_aug_substitute.augment(text, n = 1)[0]
            
        if np.random.rand() < 0.5:
            noise_type = 'random:transposition'
            noised_text = random_aug_swap.augment(text, n = 1)[0]
        
    return noised_text, noise_type


num_melik_noises = 7
def generate_melik_noise(text):
    rand_number = np.random.rand()
    prob_per_noise = 1 / num_melik_noises

    # Unchanged
    if rand_number < prob_per_noise:
        noise_type = 'melik:unchanged'
        noised_text = text
    
    # Lowercase
    elif rand_number >= prob_per_noise and rand_number < prob_per_noise * 2:
        noise_type = 'melik:lowercase'
        noised_text = turkish_lower(text)
    
    # Remove punctuations
    elif rand_number >= prob_per_noise * 2 and rand_number < prob_per_noise * 3:
        noise_type = 'melik:remove_punct'
        noised_text = remove_punctuations(text)
    
    # STT Output
    elif rand_number >= prob_per_noise * 3 and rand_number < prob_per_noise * 4:
        noise_type = 'melik:stt_output'
        noised_text = turkish_lower(remove_punctuations(text))
    
    # Split a word randomly
    elif rand_number >= prob_per_noise * 4 and rand_number < prob_per_noise * 5:
        rand_number_for_split_noise = np.random.rand()

        # Split by whitespace half of the time
        if rand_number_for_split_noise < 0.50:
            noise_type = 'melik:split_whitespace'
            noised_text = split_noise(text, " ")
        
        # Split by newline half of the time
        else:
            noise_type = 'melik:split_newline'
            noised_text = split_noise(text, '-\n')
        
    # Asciify
    elif rand_number >= prob_per_noise * 5 and rand_number < prob_per_noise * 6:
        noise_type = 'melik:asciify'
        noised_text = turkish_asciify(text)
    
    # Shuffle Chars of a random word
    else:
        noise_type = 'melik:shuffle_chars'
        words = text.split()
        
        new_words = []
        for word in words:
            if np.random.rand() < 0.10:
                word = shuffle_chars(word)
            new_words.append(word)
            
        noised_text =  " ".join(new_words)

    return noised_text, noise_type

def noise_generator(text, return_noise_type = False):
    rand_number = np.random.rand()
    
    # Keyboard Errors
    if rand_number < 0.20:
        noise_type = 'keyboard'
        noised_text = keyboard_aug.augment(text, n = 1)[0]
    
    # OCR Errors
    elif rand_number >= 0.20 and rand_number < 0.40:
        noise_type = 'ocr'
        noised_text = ocr_aug.augment(text, n = 1)[0]
    
    # Random Errors
    elif rand_number >= 0.40 and rand_number < 0.60:
        noised_text, noise_type = generate_random_errors(text)
    
    # ByT5 Errors
    elif rand_number >= 0.60 and rand_number < 0.80:
        noised_text, noise_type = generate_byt5_errors(text)
    
    # Melik Noise
    else:
        noised_text, noise_type = generate_melik_noise(text)

    if return_noise_type:
        return noised_text, noise_type
    
    else:
        return noised_text
    
