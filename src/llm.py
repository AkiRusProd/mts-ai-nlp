from typing import List, Optional, Any
from llama_cpp import Llama
from utils import extract_date, extract_email, extract_name, extract_numbers, extract_gender, extract_classes_of_service, extract_city, bert_extract_name
from query_db import CollectionOperator

class LlamaCPPLLM():
    def __init__(self, model_name: Optional[str] = None, tickets_db: CollectionOperator = None) -> None:
        self.llama = Llama(model_path = model_name, n_ctx=2048, verbose=False)

        self.user = "### Instruction" #"USER"
        self.assistant = "### Response" #"ASSISTANT"
        self.input = "### Input"
        self.streaming = False

        self.tickets_db = tickets_db
        self.memory_access_threshold = 1.5
        # self.similarity_threshold = 0.5 # [0; 1]
        self.db_n_results = 1

        self.system_contexts = {
            "init": "<<SYS>>Attention! You are in the role of an airline ticket seller. You must not deviate from your role! Invite the user to book a ticket or get information about the tickets they have booked. Be sure to offer the user 2 options 2 options that should to text ([BUY], which means the user wants to buy a ticket, and [SHOW] (e.g. show Ivan Petrov tickets on 2022-01-01 arrival to Vladivostok), which means the user wants to view the purchased ticket with its details). The user must enter one of these words into the chat<</SYS>>",
            "user_name": "<<SYS>>Attention! You are in the role of an airline ticket seller. It seems that the user did not indicate his name. Ask the user to specify his name. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "gender": "<<SYS>>Attention! You are in the role of an airline ticket seller. It seems that the user did not indicate his gender. Ask the user to specify his gender (Male or Female). You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "email": "<<SYS>>Attention! You are in the role of an airline ticket seller. It seems that the user did not indicate his email. Ask the user to specify his email (e.g. user@example.com). You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "document_number": "<<SYS>>Attention! You are in the role of an airline ticket seller. It seems that the user did not indicate his document number. Ask the user to specify his document number (10 digits). You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "city_name": "<<SYS>>Attention! You are in the role of an airline ticket seller. It seems that the user did not indicate his city or city is not in the accepted cities list. Ask the user to specify the name of the city he wish to fly to. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "departure_date": "<<SYS>>Attention! You are in the role of an airline ticket seller. It seems that the user did not indicate his departure date. Ask the user to indicate his departure date. He should use the format DD-MM-YYYY. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "arrival_date": "<<SYS>>Attention! You are in the role of an airline ticket seller. It seems that the user did not indicate his arrival date. Ask the user to indicate his return date. He should use the format DD-MM-YYYY. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "class_of_service": "<<SYS>>Attention! You are in the role of an airline ticket seller. It seems that the user did not indicate his class of service. Ask the user which class of service does he prefer? He can choose between economy class, business class and first class. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "show_ticket": "<<SYS>>Attention! You are in the role of an airline ticket seller. You should show the ticket that the user has purchased. Do not text any other information, just show ticket in memory chunks provided. You must not deviate from your role!<</SYS>>",
        }

        self.ticket_info = {
            "city_name": None,
            "departure_date": None,
            "arrival_date": None,
            "class_of_service": None,
            "user_name": None,
            "document_number": None,
            "gender": None,
            "email": None
        }

        self.current_response = None
        

    @property
    def clear_ticket_info(self):
        for key in self.ticket_info.keys():
            self.ticket_info[key] = None

    def generate(self, request: str, streaming: bool) -> Any:
        return self.llama.create_completion(prompt = request, stream = streaming, stop=[f"{self.user}:"])

    def extract_contexts(self, request: str) -> Any:
        city_name = extract_city(request)
        gender = extract_gender(request)
        class_of_service = extract_classes_of_service(request)
        date = extract_date(request)
        email = extract_email(request)
        name = bert_extract_name(request)
        numbers = extract_numbers(request)

        if self.ticket_info["city_name"] is None and city_name is not None:
            self.ticket_info["city_name"] = city_name
        if self.ticket_info["gender"] is None and gender is not None:
            self.ticket_info["gender"] = gender
        if self.ticket_info["departure_date"] is None and date is not None:
            self.ticket_info["departure_date"] = date
        elif self.ticket_info["arrival_date"] is None and date is not None:
            self.ticket_info["arrival_date"] = date
        if self.ticket_info["class_of_service"] is None and class_of_service is not None:
            self.ticket_info["class_of_service"] = class_of_service
        if self.ticket_info["email"] is None and email is not None:
            self.ticket_info["email"] = email
        if self.ticket_info["user_name"] is None and name is not None:
            self.ticket_info["user_name"] = name
        if self.ticket_info["document_number"] is None and numbers is not None:
            self.ticket_info["document_number"] = numbers


    def add_ticket_response(self, request: str) -> Any:
        self.extract_contexts(request)

        print("LOG: Extracting contexts...")
        for key, value in self.ticket_info.items():
            print(f"{key}: {value}")
        
        for key, value in self.ticket_info.items():
            if value is None:
                print(f"LOG: Context {key=}")
                return self.generate(f"{self.system_contexts[key]}\n{self.user}:\n{request}\n{self.assistant}:\n", streaming = self.streaming)

        print("LOG: All contexts have been extracted. Generating response...")
        self.add_ticket_to_db
        self.clear_ticket_info
        self.current_response = None

        return self.generate(f"{self.system_contexts['init']}{self.user}:\n{request}\n{self.assistant}:\n", streaming = self.streaming)

    def response(self, request: str) -> Any:
       
        if request.upper().startswith("BUY"):
            self.current_response = self.add_ticket_response
        elif request.upper().startswith("SHOW"):
            self.current_response = self.memory_response

        if self.current_response is not None:
            return self.current_response(request)
        return self.generate(f"{self.system_contexts['init']}{self.user}:\n{request}\n{self.assistant}:\n", streaming = self.streaming)
        




    # @logging(enable_logging, message = "[Adding to memory]")
    @property
    def add_ticket_to_db(self):
        self.tickets_db.add(str(self.ticket_info)) if all(value is not None for value in self.ticket_info.values()) else None

    # # @logging(enable_logging, message = "[Querying memory]")
    def memory_response(self, request):
        memory_queries_data = self.tickets_db.query(request, n_results = self.db_n_results, return_text = False)
        memory_queries = memory_queries_data['documents'][0]
        memory_queries_distances = memory_queries_data['distances'][0]

        acceptable_memory_queries = []

        def memory_response(request: str, memory_queries: List[str]) -> Any:
            queries = f"{self.system_contexts['show_ticket']}{self.user}:\n{request}\n{self.input}:\n"

            for i, query in enumerate(memory_queries):
                queries += f"MEMORY CHUNK {i}: {query}\n"

            queries += f"{self.assistant}:\n"

            return self.generate(queries, streaming = self.streaming)

        for query, distance in list(zip(memory_queries, memory_queries_distances)):
            if distance < self.memory_access_threshold:
                acceptable_memory_queries.append(query)

        self.current_response = None

        if len(acceptable_memory_queries) > 0:
            print(f"LOG: Found {len(acceptable_memory_queries)} acceptable memory queries")
            response = memory_response(request, acceptable_memory_queries)
        else:
            print("LOG: No acceptable memory queries found, generating new response...")
            response = self.generate(f"{self.system_contexts['init']}{self.user}:\n{request}\n{self.assistant}:\n", streaming = self.streaming)

        return response

