from llm import LlamaCPPLLM
from embedder import HFEmbedder
from query_db import CollectionOperator
from dotenv import dotenv_values

env = dotenv_values(".env")


def chat():
    llm_agent.streaming = True

    while True:
        user_text_request = input("You > ")

        bot_text_response = llm_agent.response(user_text_request)
        
        if llm_agent.streaming:
            print(f"Bot <", end = ' ')
            for token in bot_text_response:
                print(token['choices'][0]['text'], end = '')
            print()
        else:
            print(f"Bot < {bot_text_response['choices'][0]['text']}")


if __name__ == "__main__":

    
    embedder = HFEmbedder()

    tickets_db_operator = CollectionOperator("total-memory", embedder = embedder)
    
    llm_agent = LlamaCPPLLM(env['LLM_PATH'], tickets_db_operator)
    llm_agent.user = "### Instructions" #"USER", ### Human
    llm_agent.assistant = "### Response" #"ASSISTANT"

    # llm_agent = LLMAgent(llm, total_memory_co)

    chat()