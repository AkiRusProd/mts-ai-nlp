from typing import List, Optional, Any
from llama_cpp import Llama
from utils import extract_date, extract_email, extract_birth_date, extract_numbers, extract_gender, extract_classes_of_service, extract_city, bert_extract_name, extract_value
from utils import cities
from tickets_db import TicketsDB
from flights_db import FlightsDB

class LlamaCPPLLM():
    def __init__(self, model_name: Optional[str] = None, tickets_db: TicketsDB = None, flights_db: FlightsDB = None) -> None:
        self.llama = Llama(model_path = model_name, n_ctx=2048, verbose=False)

        self.user = "### Instruction"
        self.assistant = "### Response"
        self.input = "### Input"
        self.streaming = False

        self.tickets_db = tickets_db
        self.flights_db = flights_db
        self.memory_access_threshold = 1.5
        # self.similarity_threshold = 0.5 # [0; 1]
        self.db_n_results = 1

        self.system_contexts = {
            "init": "<<SYS>>Keep in mind!, you are in the role of an airline ticket seller. Stick to your role! Prompt the user to book a ticket or inquire about their bookings. Make sure to mention the two options: [BUY] to purchase a ticket, and [SHOW] to view ticket details. The user must use these terms in the chat!<</SYS>>",
            "buy": "<<SYS>>Remember, you are in the role of an airline ticket seller. Stick to your role! The user has booked a ticket. Wish the user a happy flight; be sure to ask this question!<</SYS>>",
            "user_name": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his name. Ask the user to specify his name. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "gender": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his gender. Ask the user to specify his gender (Male or Female). You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "birth_date": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his birth date. Ask the user to specify his birth date. He should use the format DD-MM-YYYY. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "email": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his email. Ask the user to specify his email (e.g. user@example.com). You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "document_number": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his document number. Ask the user to specify his document number (10 digits). You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "city_name": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his city or city is not in the accepted cities list. Accepted cities are {cities}: Ask the user to specify the name of the city he wish to fly to. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            # "departure_date": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his departure date or the date is not in the accepted dates. Accepted dates for {city} are {dates}. Ask the user to indicate his departure date. He should use the format DD-MM-YYYY. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "class_of_service": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his class of service. Ask the user which class of service does he prefer? He can choose between economy class, business class and first class. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "show_ticket": "<<SYS>>You should show the ticket as json object in tickets infos that the user has purchased. Do not text any other information, just show ticket as json object in memory chunks provided. You must not deviate from your role!<</SYS>>",
            # "price": "<<SYS>>Remember, you are in the role of an airline ticket seller. The user has purchased a ticket. The prices of suitable tickets is {prices}. You must not deviate from your role!<</SYS>>",
            "ticket_id": "<<SYS>>Remember, you are in the role of an airline ticket seller. The user has a choice of several tickets. The ticket ids are {ticket_ids}. You must not deviate from your role!<</SYS>>",
        }

        self.ticket_info = {
            "city_name": None,
            "departure_date": None,
            "arrival_date": None,
            "price": None,
            "ticket_id": None,
            "class_of_service": None,
            "user_name": None,
            "document_number": None,
            "gender": None,
            "birth_date": None,
            "email": None,
        }

        self.current_response = None
        

    @property
    def clear_ticket_info(self):
        for key in self.ticket_info.keys():
            self.ticket_info[key] = None

    def response(self, request: str, streaming: bool) -> Any:
        return self.llama.create_completion(prompt = request, stream = streaming, stop=[f"{self.user}:"])

    def extract_contexts(self, request: str) -> Any:
        city_name = extract_city(request, self.flights_db.get_cities())
        gender = extract_gender(request)
        class_of_service = extract_classes_of_service(request)
        birth_date = extract_birth_date(request)
        # departure_date = extract_date(request, self.flights_db.get_departure_dates(self.ticket_info["city_name"]))
        ticket_id = extract_value(request, self.flights_db.get_ticket_ids(self.ticket_info["city_name"]))
        email = extract_email(request)
        name = bert_extract_name(request)
        numbers = extract_numbers(request)
        # price = extract_value(request, self.flights_db.get_prices(self.ticket_info["city_name"], self.ticket_info["departure_date"]))




        if self.ticket_info["city_name"] is None and city_name is not None:
            self.ticket_info["city_name"] = city_name
        if self.ticket_info["gender"] is None and gender is not None:
            self.ticket_info["gender"] = gender
        # if self.ticket_info["departure_date"] is None and departure_date is not None:
        #     self.ticket_info["departure_date"] = departure_date
            # self.ticket_info["arrival_date"] = self.flights_db.get_arrival_date(city_name, departure_date)
        # elif self.ticket_info["price"] is None and price is not None:
        #     self.ticket_info["price"] = price
        if self.ticket_info["ticket_id"] is None and ticket_id is not None:
            self.ticket_info["ticket_id"] = ticket_id
            self.ticket_info["departure_date"] = self.flights_db.get_ticket(ticket_id)["departure_date"]
            self.ticket_info["arrival_date"] = self.flights_db.get_ticket(ticket_id)["arrival_date"]
            self.ticket_info["price"] = self.flights_db.get_ticket(ticket_id)["price"]

        elif self.ticket_info["birth_date"] is None and birth_date is not None:
            self.ticket_info["birth_date"] = birth_date
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
            if key in ["departure_date", "arrival_date", "price"]:
                continue

            if value is None:
                print(f"LOG: Context {key=}")
                system_context = self.system_contexts[key]

                if key == "city_name":
                    system_context = system_context.format(cities = self.flights_db.get_cities())
                elif key == "ticket_id":
                    print(f"Suitable flights:\n{self.flights_db.get_flights(self.ticket_info['city_name'])}")

                    system_context = system_context.format(ticket_ids = self.flights_db.get_ticket_ids(self.ticket_info['city_name']))
                # elif key == "departure_date":
                #     print(f"Suitable flights:\n{self.flights_db.get_flights(self.ticket_info['city_name'])}")

                #     system_context = system_context.format(city = self.ticket_info["city_name"], dates = self.flights_db.get_departure_dates(self.ticket_info["city_name"]))
                # elif key == "price":
                #     print(f"Suitable flights:\n{self.flights_db.get_flights(self.ticket_info['city_name'], self.ticket_info['departure_date'])}")

                    # system_context = system_context.format(prices = self.flights_db.get_prices(self.ticket_info["city_name"], self.ticket_info["departure_date"]))

                return self.response(f"{system_context}\n{self.user}:\n{request}\n{self.assistant}:\n", streaming = self.streaming)

        print("LOG: All contexts have been extracted. Generating response...")
        self.add_ticket_to_db
        self.clear_ticket_info
        self.current_response = None

        return self.response(f"{self.system_contexts['buy']}{self.user}:\n{request}\n{self.assistant}:\n", streaming = self.streaming)

    def generate(self, request: str) -> Any:
       
        if request.upper().startswith("BUY"):
            self.current_response = self.add_ticket_response
        elif request.upper().startswith("SHOW"):
            self.current_response = self.memory_response

        if self.current_response is not None:
            return self.current_response(request)
        return self.response(f"{self.system_contexts['init']}{self.user}:\n{request}\n{self.assistant}:\n", streaming = self.streaming)
        


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
                queries += f"Ticket info {i}: {query}\n"

            queries += f"{self.assistant}:\n"

            return self.response(queries, streaming = self.streaming)

        for query, distance in list(zip(memory_queries, memory_queries_distances)):
            if distance < self.memory_access_threshold:
                acceptable_memory_queries.append(query)

        self.current_response = None

        if len(acceptable_memory_queries) > 0:
            print(f"LOG: Found {len(acceptable_memory_queries)} acceptable memory queries")
            response = memory_response(request, acceptable_memory_queries)
        else:
            print("LOG: No acceptable memory queries found, generating new response...")
            response = self.response(f"{self.system_contexts['init']}{self.user}:\n{request}\n{self.assistant}:\n", streaming = self.streaming)

        return response

