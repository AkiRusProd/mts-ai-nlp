import re
import nltk

from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

from datetime import datetime
from dateutil.parser import parse
from bert_ner import BERTNER
from functools import wraps
from termcolor import colored

nltk.download('punkt') if not nltk.data.find('tokenizers/punkt') else None
nltk.download('averaged_perceptron_tagger') if not nltk.data.find('taggers/averaged_perceptron_tagger') else None
nltk.download('maxent_ne_chunker') if not nltk.data.find('chunkers/maxent_ne_chunker') else None
nltk.download('words') if not nltk.data.find('corpora/words') else None

def extract_date(text, valid_dates):
    try:
        dt = parse(text, fuzzy=True)
        
        if str(dt) in valid_dates:
            return str(dt)
        
        return None
    except ValueError:
        return None



def extract_birth_date(text):
    try:
        dt = parse(text, fuzzy=True)
        if (datetime.now() - dt).days >= 18 * 365:  # check if the date is at least 18 years old
            return str(dt)
        else:
            return None
    except ValueError:
        return None

# print(extract_birthday("1358"))
# text = "I have a meeting on 30/03/2023 at 10:30"
# date = extract_date(text)
# print(date)


def extract_email(text):
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_regex, text)
    return matches[0] if matches else None


def extract_name(text):
    nltk_results = ne_chunk(pos_tag(word_tokenize(text)))
    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '
            # print ('Type: ', nltk_result.label(), 'Name: ', name)
            if nltk_result.label() == 'PERSON':
                return name
            

print(extract_name("Oleg Akimov"))

def extract_numbers(text):
    # Ищем группу из 4 цифр, возможно разделенных пробелами, затем ищем 6 цифр
    match = re.search(r'(\d\s*\d\s*\d\s*\d)\s*(\d{6})', text)
    if match:
        # Удаляем пробелы и объединяем числа
        numbers = ''.join(match.group(1).split()) + match.group(2)
        return numbers
    else:
        return None

# text = "1234 567890"
# numbers = extract_numbers(text)
# print(numbers)  # Вывод: 1234567890




cities = ["Volgograd", "Saint Petersburg", "Novosibirsk", "Yekaterinburg", "Kazan", "Chelyabinsk", "Rostov", "Ufa", "Krasnoyarsk", "Perm"]

def extract_classes_of_service(text):
    classes_of_service = ["economy", "business", "first"]

    if any(word in text.lower() for word in classes_of_service):
        return next(service for service in classes_of_service if service in text.lower())
    else:
        return None

def extract_gender(text):
    male_synonyms = ["male", "man", "guy", "gentleman", "he-man", "masculine", "manful", "manlike", "virile", "androcentric", "androcratic", "androgenous", "staminate", "anthropoidal"]
    female_synonyms = ["female", "woman", "lady", "ladylike", "she-woman", "feminine", "womanful", "womanlike", "girl", "dame", "feminine", "distaff", "heroine", "broad" ]

    if any(word in text.lower() for word in male_synonyms):
        return "male"
    elif any(word in text.lower() for word in female_synonyms):
        return "female"
    else:
        return None

def extract_city(text, cities):
    for city in cities:
        if city in text:
            return city
    return None



bert_ner = BERTNER()

def bert_extract_name(text):
    entities = bert_ner(text)
    names = []
    name_tokens = []
    for entity in entities:
        if 'PER' in entity['entity']:
            if entity['entity'] == 'B-PER' and name_tokens:
                names.append(' '.join(name_tokens).replace(' ##', ''))
                name_tokens = []
            name_tokens.append(entity['word'])
    if name_tokens:
        names.append(' '.join(name_tokens).replace(' ##', ''))
    
    if len(names) > 0:
        return names[0]
    else:
        return None



# print(bert_extract_name("John and Valentin are going to the park. They will meet with Rustam Akimov there."))

def extract_value(text, values):
    if not values or not text:
        return None

    for price in values:
        if str(price) in text:
            return price
    return None


def logging(enabled = True, message = "", color = "yellow"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enabled:
                print(f"LOG: {colored(message, color = color)}")
            return func(*args, **kwargs)
        return wrapper
    return decorator