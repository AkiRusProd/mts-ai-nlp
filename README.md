# mts-ai-nlp
NLP project for MTS AI       

## Description
This system, designed for efficient airline ticket booking, consists of three main components: the Q4 quantized [Mistral-7B-Instruct-v0.1](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) language model, the [ChromaDB](https://github.com/chroma-core/chroma) database system, and a user name extraction module powered by [bert-large-NER](https://huggingface.co/dslim/bert-large-NER). The language model processes and responds to user requests, while the user name extraction module, utilizing a fine-tuned BERT model, accurately identifies user names from inputs. The ChromaDB system stores and retrieves flight data, initially held in a pandas DataFrame for efficient manipulation. These components work together to automate ticket booking, providing a personalized user experience.

### Scripts:
1. [bert_ner.py](src/bert_ner.py) - fine-tuned BERT model for entity recognition       
2. [chat.py](src/chat.py) - runs the chat      
3. [embedder.py](src/embedder.py) - embedding [sup-simcse-roberta-large](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large) model to operate with text in vector db        
4. [flight_db_filler.py](src/flight_db_filler.py) - fills the database with synthetic data     
5. [flights_db.py](src/flights_db.py) - class to operate on the pandas flights dataframe       
6. [llm.py](src/llm.py) - class to interact with the language model        
7. [tickets_db.py](src/tickets_db.py) - class to operate on chroma db database     
8. [utils.py](src/utils.py) - utility functions for features extraction from text      


### Example:

TODO: YT link

### Notes
This system uses Q4 version of LLM via [LLAMA_cpp_python](https://github.com/abetlen/llama-cpp-python) binding.

### Usage:
- Install requirements.txt
- Download [Mistral-7B-Instruct-v0.1 Q4 version](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
- Specify variables in .env
- Run [flight_db_filler.py](src/flight_db_filler.py) to fill the database with synthetic data
- Run [chat.py](src/chat.py)