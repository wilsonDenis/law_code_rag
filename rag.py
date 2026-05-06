
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from dotenv import load_dotenv
from groq import Groq

from vector_db import VectorDB
from config import LLM_MODEL_NAME, VECTOR_DB_NAME


class RAG:


    def __init__(self) -> None:
        load_dotenv()
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.vector_db = VectorDB(VECTOR_DB_NAME)



    @staticmethod
    def _read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_context(self, docs: list[str], metadatas: list[dict]) -> str:
        template = self._read_file("context.txt")
        chunks_text = ""
        for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
            chunks_text += (
                f"[{i}] Article {meta['article']} — {meta['titre']}\n"
                f"    Section : {meta['section']}\n"
                f"    Texte : {doc}\n\n"
            )
        return template.replace("{{Chuncks}}", chunks_text)



    def answer_question(self, question: str) -> tuple[str, list[str]]:

        docs, metadatas = self.vector_db.retrieve(question, k=4)
        context = self._build_context(docs, metadatas)

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": question},
            ],
            model=LLM_MODEL_NAME,
        )
        answer = response.choices[0].message.content

        sources = sorted({meta["article"] for meta in metadatas})
        return answer, sources




def main() -> None:
    print("Chargement de la base de connaissances...")
    rag = RAG()
    print("\nSystème RAG prêt. Tapez 'quit' pour quitter.\n")

    while True:
        question = input("\nVotre question : ").strip()

        if question.lower() in {"quit", "exit", "q"}:
            print("Au revoir !")
            break

        if not question:
            continue

        print("\nRecherche en cours...\n")
        answer, sources = rag.answer_question(question)

        print(answer)
        print(f"\nSources : Articles {', '.join(sources)} du Code du travail")
    


if __name__ == "__main__":
    main()
