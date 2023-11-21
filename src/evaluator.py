import os
import numpy as np
from evaluate import load
from llm import LlamaCPPLLM
from typing import Union, List
from tqdm import tqdm
from dotenv import dotenv_values

env = dotenv_values(".env")
os.environ["HUGGINGFACE_HUB_CACHE"] = env["HUGGINGFACE_HUB_CACHE"]


class LLMEvaluator:
    def __init__(self, model: LlamaCPPLLM):
        self.model = model
        self.bleurt = load("bleurt", module_type="metric")
        self.rouge = load("rouge")
        self.bertscore = load("bertscore")

    def evaluate(
        self, questions: List[str], answers: List[str], filename: str = None
    ) -> None:
        assert (
            isinstance(questions, list)
            and isinstance(answers, list)
            and len(questions) == len(answers)
        )

        predictions = [
            self.model.llama.create_completion(
                f"### Instruction:\n{q}\n### Response:\n", max_tokens=512
            )["choices"][0]["text"]
            for q in tqdm(questions, total=len(questions))
        ]

        bert_scores = self.bertscore.compute(
            predictions=predictions,
            references=answers,
            lang="en",
            model_type="distilbert-base-uncased",
        )

        bleurt_results = self.bleurt.compute(
            predictions=predictions, references=answers
        )
        bleurt_scores = bleurt_results["scores"]

        average_rouge_scores = self.rouge.compute(
            predictions=predictions, references=answers
        )
        print(f"Rouge scores: {average_rouge_scores}")

        average_bert_scores = {
            "precision": np.mean(bert_scores["precision"]),
            "recall": np.mean(bert_scores["recall"]),
            "f1": np.mean(bert_scores["f1"]),
        }
        print(f"Bert scores: {average_bert_scores}")

        average_bleurt_score = np.mean(bleurt_scores)
        print(f"Bleurt score: {average_bleurt_score}")

        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(
                    f"Rouge scores: {average_rouge_scores}\nBert scores: {average_bert_scores}\nBleurt score: {average_bleurt_score}\n\n"
                )
                for q, a, p in zip(questions, answers, predictions):
                    f.write(
                        f"Question:\n{q}\nAnswer:\n{a}\nPrediction:\n{p}\n\n".encode(
                            "utf-8"
                        ).decode()
                    )


if __name__ == "__main__":
    questions = [
        "What is the capital of France?",
        "Who wrote the novel '1984'?",
        "What is the square root of 64?",
        "Who painted the Mona Lisa?",
        "What is the distance from Earth to the Moon?",
        "Who is the current president of the United States?",
        "What is the chemical symbol for gold?",
        "What is the highest mountain in the world?",
        "Who discovered penicillin?",
        "What is the largest ocean on Earth?",
        "What is the capital of Germany?",
        "Who wrote the novel 'To Kill a Mockingbird'?",
        "What is the square root of 144?",
        "Who painted 'The Starry Night'?",
        "What is the distance from Earth to Mars?",
        "Who is the current Prime Minister of the United Kingdom?",
        "What is the chemical symbol for silver?",
        "What is the longest river in the world?",
        "Who invented the telephone?",
        "What is the largest continent on Earth?",
        "What is the speed of light?",
        "Who composed the music for 'The Magic Flute'?",
        "What is the boiling point of water at sea level?",
        "Who is the author of 'Pride and Prejudice'?",
        "What is the currency of Japan?",
        "What is the tallest building in the world?",
        "Who discovered gravity?",
        "What is the largest planet in our solar system?",
        "Who won the Nobel Prize in Literature in 2020?",
        "What is the national bird of the United States?",
    ]

    answers = [  # reference full-precision model answers
        "The capital of France is Paris.",
        "The novel '1984' was written by George Orwell.",
        "The square root of 64 is 8.",
        "The Mona Lisa was painted by Leonardo da Vinci.",
        "The average distance from Earth to the Moon is about 238,900 miles (384,400 kilometers). However, this distance can vary slightly due to the elliptical shape of the Moon's orbit around Earth.",
        "The current President of the United States is Joe Biden.",
        "The chemical symbol for gold is Au.",
        "The highest mountain in the world is Mount Everest, which stands at 8,848 meters (29,029 feet) tall.",
        "Penicillin was discovered by Sir Alexander Fleming in 1928.",
        "The largest ocean on Earth is the Pacific Ocean.",
        "The capital of Germany is Berlin.",
        "The novel 'To Kill a Mockingbird' was written by Harper Lee.",
        "The square root of 144 is 12.",
        "'The Starry Night' was painted by Vincent van Gogh.",
        "The average distance from Earth to Mars is approximately 225 million miles (361 million kilometers). However, this distance can vary due to the elliptical shape of the planets' orbits.",
        "As of my knowledge up to August 2021, the current Prime Minister of the United Kingdom is Boris Johnson.",
        "The chemical symbol for silver is Ag.",
        "The longest river in the world is the Nile River, which is approximately 4,135 miles or 6,650 kilometers long.",
        "Alexander Graham Bell invented the telephone.",
        "The largest continent on Earth is Asia, covering an area of approximately 44.58 million km2 (17.17 million sq mi).",
        "The speed of light is approximately 299,792 kilometers per second or 186,282 miles per second. It is denoted by the universal physical constant 'c'.",
        "The music for 'The Magic Flute' was composed by Wolfgang Amadeus Mozart.",
        "The boiling point of water at sea level is 100 degrees Celsius or 212 degrees Fahrenheit.",
        "The author of 'Pride and Prejudice' is Jane Austen.",
        "The currency of Japan is the Japanese Yen (JPY).",
        "The tallest building in the world is the Burj Khalifa in Dubai, United Arab Emirates. It stands at 828 meters (2,717 feet) tall with 163 floors.",
        "Gravity is a fundamental force of nature that affects all objects with mass, and it was discovered by Sir Isaac Newton.",
        "The largest planet in our solar system is Jupiter.",
        "Louis Glück won the Nobel Prize in Literature in 2020.",
        "The national bird of the United States is the Bald Eagle.",
    ]

    evaluator = LLMEvaluator(LlamaCPPLLM(model_name=env["LLM_PATH"]))
    evaluator.evaluate(questions, answers, filename="Q-model-eval.txt")
