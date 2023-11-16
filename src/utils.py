import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree


from dateutil.parser import parse
from ner import BERTNER
from functools import wraps
from termcolor import colored

def extract_date(text):
    try:
        dt = parse(text, fuzzy=True)
        return dt
    except ValueError:
        return None

# text = "I have a meeting on 30/03/2023 at 10:30"
# date = extract_date(text)
# print(date)


def extract_email(text):
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_regex, text)
    return matches[0] if matches else None

# text = "My email addresses are example@example.com and another_example@domain.com"
# emails = extract_emails(text)
# print(emails)

# import spacy

# def extract_names(text):
#     nlp = spacy.load("en_core_web_sm")
#     doc = nlp(text)
#     names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
#     return names

# text = "John and Valentin are going to the park. They will meet with Michael there."
# names = extract_names(text)
# print(names)


# text = '''
# This is a sample text that contains the name Akimov Rustam who is one of the developers of this project.
# You can also find the surname Jones here.
# '''

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

male_synonyms = ["male", "man", "guy", "gentleman", "he-man", "masculine", "manful", "manlike", "virile", "androcentric", "androcratic", "androgenous", "staminate", "anthropoidal"]
female_synonyms = ["female", "woman", "lady", "ladylike", "she-woman", "feminine", "womanful", "womanlike", "girl", "dame", "feminine", "distaff", "heroine", "broad" ]
classes_of_service = ["economy", "business", "first"]
cities = ["Volgograd", "Saint Petersburg", "Novosibirsk", "Yekaterinburg", "Kazan", "Chelyabinsk", "Rostov", "Ufa", "Krasnoyarsk", "Perm"]

def extract_classes_of_service(text):
    if any(word in text.lower() for word in classes_of_service):
        return next(service for service in classes_of_service if service in text.lower())
    else:
        return None

def extract_gender(text):
    if any(word in text.lower() for word in male_synonyms):
        return "male"
    elif any(word in text.lower() for word in female_synonyms):
        return "female"
    else:
        return None

def extract_city(text):
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


def logging(enabled = True, message = "", color = "yellow"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enabled:
                print(f"LOG: {colored(message, color = color)}")
            return func(*args, **kwargs)
        return wrapper
    return decorator