import ast
from prettytable import PrettyTable
from termcolor import colored
from typing import List, Optional, Any
from llama_cpp import Llama
from tickets_db import TicketsDB
from flights_db import FlightsDB
from utils import (
    extract_email,
    extract_birth_date,
    extract_numbers,
    extract_gender,
    extract_classes_of_service,
    extract_city,
    extract_name,
    extract_value,
)

from utils import logging

enable_logging = True


class LlamaCPPLLM:
    def __init__(
        self,
        model_name: Optional[str] = None,
        tickets_db: TicketsDB = None,
        flights_db: FlightsDB = None,
    ) -> None:
        self.llama = Llama(model_path=model_name, n_ctx=2048, verbose=False)

        self.user = "### Instruction"
        self.assistant = "### Response"
        self.input = "### Input"
        self.streaming = False

        self.tickets_db = tickets_db
        self.flights_db = flights_db
        self.memory_access_threshold = 1.5

        self.db_n_results = 1

        self.system_contexts = {
            "init": "<<SYS>>Keep in mind!, you are in the role of an airline ticket seller. Stick to your role!  It seems that the user did not indicate his choise (BUY or SHOW). Prompt the user to book a ticket or inquire about their bookings. Make sure to mention the two options: [BUY] to purchase a ticket, and [SHOW] to view ticket details. The user must use these terms in the chat!<</SYS>>",
            "buy": "<<SYS>>Remember, you are in the role of an airline ticket seller. Stick to your role! The user {user_name} has booked a ticket with departure date {departure_date}, arrival date {arrival_date} and seat_place {seat_place}. The price is {price}. Wish the user a happy flight; be sure to ask this question!<</SYS>>",
            "user_name": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his name. Ask the user to specify his full name. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "gender": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his gender. Ask the user to specify his gender (Male or Female). You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "birth_date": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his birth date. Ask the user to specify his birth date. He should use the format DD-MM-YYYY. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "email": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his email. Ask the user to specify his email (e.g. user@example.com). You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "document_number": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his document number. Ask the user to specify his document number (10 digits). You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "city_name": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his city or city is not in the accepted cities list. Accepted cities are {cities}: Ask the user to specify the name of the city he wish to fly to. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "class_of_service": "<<SYS>>Keep in mind! You are in the role of an airline ticket seller. It seems that the user did not indicate his class of service. Ask the user which class of service does he prefer? He can choose between economy class, business class and first class. You must not deviate from your role; be sure to ask this question!<</SYS>>",
            "show_ticket": "<<SYS>>You should extract base information from the flight ticket in provided TICKET_INFO ant tell the summary of the ticket. Just one sentence about the ticket. Please don't write anything else<</SYS>>",
            "ticket_id": "<<SYS>>Remember, you are in the role of an airline ticket seller. The user has a choice of several tickets. It seems that the user did not indicate his id or id is not in the accepted ids. Accepted ticket ids are {ticket_ids}. Say user he should choose a ticket by its id. Don't come up with anything that isn't listed in SYS!!! You must not deviate from your role!<</SYS>>",
        }

        self.ticket_info = {
            "city_name": None,
            "ticket_id": None,
            "departure_date": None,
            "arrival_date": None,
            "seat_place": None,
            "price": None,
            "class_of_service": None,
            "user_name": None,
            "document_number": None,
            "gender": None,
            "birth_date": None,
            "email": None,
        }

        self.current_response = None

    @property
    @logging(enable_logging, message="[Resetting ticket info]")
    def clear_ticket_info(self):
        for key in self.ticket_info.keys():
            self.ticket_info[key] = None

    @logging(enable_logging, message="[Printing ticket info]")
    def print_ticket_info(self, ticket_info=None):
        ticket_info = self.ticket_info if ticket_info is None else ticket_info
        table = PrettyTable()
        table.field_names = [colored("Ticket Info", "blue"), colored("Value", "blue")]
        for key, value in ticket_info.items():
            table.add_row(
                [colored(key, "green"), colored(value, "yellow" if value else "red")]
            )
        print(table)

    @property
    @logging(enable_logging, message="[Printing flights info]")
    def print_flights(self):
        flights = self.flights_db.get_flights(self.ticket_info["city_name"])
        table = PrettyTable()
        table.field_names = [
            colored("ID", "blue"),
            colored("City Name", "blue"),
            colored("Departure Date", "blue"),
            colored("Arrival Date", "blue"),
            colored("Seat Place", "blue"),
            colored("Price", "blue"),
        ]

        for id, flight in flights.iterrows():
            table.add_row(
                [
                    colored(id, "magenta"),
                    colored(flight["city_name"], "green"),
                    colored(flight["departure_date"], "yellow"),
                    colored(flight["arrival_date"], "yellow"),
                    colored(flight["seat_place"], "cyan"),
                    colored(flight["price"], "red"),
                ]
            )

        print(table)

    @property
    @logging(enable_logging, message="[Adding info to tickets db]")
    def add_ticket_to_db(self):
        self.tickets_db.add(str(self.ticket_info)) if all(
            value is not None for value in self.ticket_info.values()
        ) else None

    def response(self, request: str, streaming: bool) -> Any:
        return self.llama.create_completion(
            prompt=request, stream=streaming, max_tokens=512, stop=[f"{self.user}:"]
        )

    def extract_contexts(self, request: str) -> Any:
        extraction_mapping = {
            "city_name": (extract_city, [request, self.flights_db.get_cities()]),
            "gender": (extract_gender, [request]),
            "class_of_service": (extract_classes_of_service, [request]),
            "birth_date": (extract_birth_date, [request]),
            "ticket_id": (extract_value, [request, self.flights_db.get_ticket_ids(self.ticket_info["city_name"])]),
            "email": (extract_email, [request]),
            "user_name": (extract_name, [request]),
            "document_number": (extract_numbers, [request])
        }

        for field, (extractor, args) in extraction_mapping.items():
            if self.ticket_info[field] is None:
                value = extractor(*args)
                if value is not None:
                    self.ticket_info[field] = value

        if self.ticket_info["ticket_id"] is not None:
            ticket = self.flights_db.get_ticket(self.ticket_info["ticket_id"])
            self.ticket_info.update({
                "departure_date": ticket["departure_date"],
                "arrival_date": ticket["arrival_date"],
                "price": ticket["price"],
                "seat_place": ticket["seat_place"]
            })

    def add_ticket_response(self, request: str) -> Any:
        self.extract_contexts(request)

        self.print_ticket_info()

        for key, value in self.ticket_info.items():
            if key in ["departure_date", "arrival_date", "seat_place", "price"]:
                continue

            if value is None:
                system_context = self.system_contexts[key]

                if key == "city_name":
                    system_context = system_context.format(
                        cities=self.flights_db.get_cities()
                    )
                elif key == "ticket_id":
                    self.print_flights
                    system_context = system_context.format(
                        ticket_ids=self.flights_db.get_ticket_ids(
                            self.ticket_info["city_name"]
                        )
                    )

                return self.response(
                    f"{system_context}\n{self.user}:\n{request}\n{self.assistant}:\n",
                    streaming=self.streaming,
                )

        system_context = self.system_contexts["buy"].format(
            user_name=self.ticket_info["user_name"],
            departure_date=self.ticket_info["departure_date"],
            arrival_date=self.ticket_info["arrival_date"],
            seat_place=self.ticket_info["seat_place"],
            price=self.ticket_info["price"],
        )
        self.add_ticket_to_db
        self.clear_ticket_info
        self.current_response = None

        return self.response(
            f"{system_context}{self.user}:\n{request}\n{self.assistant}:\n",
            streaming=self.streaming,
        )

    def generate(self, request: str) -> Any:
        if request.upper().startswith("BUY"):
            self.current_response = self.add_ticket_response
        elif request.upper().startswith("SHOW"):
            self.current_response = self.memory_response

        if self.current_response is not None:
            return self.current_response(request)
        return self.response(
            f"{self.system_contexts['init']}{self.user}:\n{request}\n{self.assistant}:\n",
            streaming=self.streaming,
        )

    @logging(enable_logging, message="[Querying ticket info from tickets db]")
    def memory_response(self, request):
        memory_queries_data = self.tickets_db.query(
            request, n_results=self.db_n_results, return_text=False
        )
        memory_queries = memory_queries_data["documents"][0]
        memory_queries_distances = memory_queries_data["distances"][0]

        acceptable_memory_queries = []

        def memory_response(request: str, memory_queries: List[str]) -> Any:
            queries = f"{self.system_contexts['show_ticket']}{self.user}:\n{request}\n{self.input}:\n"

            for i, query in enumerate(memory_queries):
                queries += f"TICKET_INFO {i}: {query}\n"
                self.print_ticket_info(ast.literal_eval(query))

            queries += f"{self.assistant}:\n"

            return self.response(queries, streaming=self.streaming)

        for query, distance in list(zip(memory_queries, memory_queries_distances)):
            if distance < self.memory_access_threshold:
                acceptable_memory_queries.append(query)

        self.current_response = None

        if len(acceptable_memory_queries) > 0:
            response = memory_response(request, acceptable_memory_queries)
        else:
            response = self.response(
                f"{self.system_contexts['init']}{self.user}:\n{request}\n{self.assistant}:\n",
                streaming=self.streaming,
            )

        return response
