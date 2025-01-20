import re


def parse_text(text_path, language='en', remove_spaces=False):
    with open(text_path, mode='r', encoding='utf-8') as f:
        raw_text = f.read()  # read text, cast to lowercase and remove repeated spaces
    
    raw_text = raw_text.lower()  # cast to lower case
    
    if language == 'en':
        pattern = "[a-z]+"
    elif language == 'ru':
        pattern = "[а-яё]+"
    else:
        raise ValueError(f'Unsupported language: {language}')
    
    words = re.findall(pattern, raw_text)
    if not remove_spaces:
        text = " ".join(words)
    else:
        text = "".join(words)
    
    return text, words


def split_into_n_grams(text, N=2, overlap=False):
    res = []
    
    if not overlap:
        n_gram = ""
        for i, c in enumerate(text):
            n_gram += c
            if (i + 1) % N == 0:
                res.append(n_gram)
                n_gram = ""
    else:
        for i in range(N, len(text)):
            res.append(text[i-N:i])
            
            
    return res