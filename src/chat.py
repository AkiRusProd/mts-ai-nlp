from llm import LlamaCPPLLM
from embedder import HFEmbedder
from tickets_db import TicketsDB
from flights_db import FlightsDB
from dotenv import dotenv_values

env = dotenv_values(".env")


def chat() -> None:
    llm_agent.streaming = True

    while True:
        user_text_request = input("You > ")

        bot_text_response = llm_agent.generate(user_text_request)
        
        if llm_agent.streaming:
            print(f"Bot <", end = ' ')
            for token in bot_text_response:
                print(token['choices'][0]['text'], end = '')
            print()
        else:
            print(f"Bot < {bot_text_response['choices'][0]['text']}")


if __name__ == "__main__":
    embedder = HFEmbedder()

    tickets_db = TicketsDB("total-memory", embedder = embedder)
    flights_db = FlightsDB("flights.csv")
    
    llm_agent = LlamaCPPLLM(env['LLM_PATH'], tickets_db, flights_db)
    llm_agent.user = "### Instructions" #"USER", ### Human
    llm_agent.assistant = "### Response" #"ASSISTANT"

    chat()