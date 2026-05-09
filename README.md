# Assistant Code du Travail RAG avec Python et Groq

Système RAG (Retrieval-Augmented Generation) qui répond à des questions sur le droit du travail français en citant les articles du Code du travail.

## Lancer le projet

```bash
# 1. Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate   # macOS / Linux

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Configurer la clé API Groq dans .env
echo "GROQ_API_KEY=votre_clé_ici" > .env

# 4. Construire la base vectorielle (une seule fois)
python indexation.py

# 5. Lancer l'assistant
python rag.py
```

## Structure du projet

```
.
├── corpus/
│   └── code_travail.json   # 48 articles du Code du travail (8 thèmes)
├── indexation.py            # Chunking + création de la base FAISS
├── vector_db.py             # Classe VectorDB (FAISS + sentence-transformers)
├── rag.py                   # Classe RAG + interface CLI interactive
├── config.py                # Paramètres centralisés (modèles, chemins)
├── context.txt              # Prompt système du LLM
├── parse_corpus.py          # (optionnel) Extraction depuis l'archive LEGI XML
├── requirements.txt
└── .env                     # Clé API Groq (non commitée)
```

## Thèmes couverts

| Thème | Articles |
|---|---|
| Durée du travail et heures supplémentaires | L3121-1, L3121-27, L3121-28, L3121-33, L3121-20, L3121-22, L3121-36, L3131-1, L3132-2 |
| Congés payés | L3141-1, L3141-3, L3141-4, L3141-12, L3141-16, L3141-24 |
| Contrat de travail (CDI, CDD) | L1221-1, L1221-2, L1242-1, L1242-2, L1243-1, L1243-3, L1245-1 |
| Licenciement | L1232-1, L1232-2, L1234-1, L1234-9, L1235-3, L1237-1, L1251-1 |
| Rupture conventionnelle | L1237-11, L1237-12, L1237-13, L1237-14, L1237-19 |
| Salaire minimum (SMIC) | L3231-2, L3231-4, L3231-6, L3232-1 |
| Représentation du personnel | L2311-1, L2311-2, L2312-1, L2312-8, L2314-1 |
| Harcèlement et discrimination | L1152-1, L1152-2, L1152-3, L1153-1, L1132-1 |

## Exemple de sortie

```
■ Votre question : Combien de jours de congés payés ai-je droit par an ?

Selon l'article L3141-3 du Code du travail, tout salarié a droit à un
congé de deux jours et demi ouvrables par mois de travail effectif chez
le même employeur, soit 30 jours ouvrables (5 semaines) par an.

Précisions importantes :
- Le calcul se base sur les mois de travail effectif (article L3141-4)
- Des jours supplémentaires peuvent s'ajouter selon la convention collective

⚠️  Cet assistant ne fournit pas de conseil juridique. Consultez un avocat
ou l'inspection du travail pour votre situation personnelle.

■ Sources : Articles L3141-3, L3141-4 du Code du travail
```

---

## Questions de réflexion (Sujet C)

**Q1. Indexer chaque article séparément ou les regrouper par section ?**

Nous indexons chaque article séparément. Les articles du Code du travail sont courts (100–600 caractères) et traitent chacun d'un point de droit précis. Les regrouper par section produirait des chunks trop longs et diluerait la pertinence sémantique lors de la recherche vectorielle. L'inconvénient de l'approche par article est que certaines questions nécessitent plusieurs articles complémentaires (ex : L1234-1 pour la durée du préavis et L1234-9 pour l'indemnité) — on compense en récupérant k=4 chunks par requête.

**Q2. Comment intégrer le numéro d'article dans les métadonnées et dans la réponse ?**

Le numéro d'article est stocké dans les métadonnées FAISS (`{"article": "L3121-27", ...}`). Il est également intégré dans le texte embeddé : `"Article L3121-27 — Durée légale du travail [section] : texte"`. Cela permet de retrouver un article même si l'utilisateur cite son numéro dans sa question. Dans le prompt système, on impose au LLM de citer le numéro exact dans chaque réponse.

**Q3. Comment signaler que l'information peut être obsolète ?**

Le prompt système précise que les articles cités datent du corpus constitué. Le disclaimer final invite systématiquement à consulter un avocat ou l'inspection du travail pour toute situation personnelle. Pour une version de production, on horodaterait le corpus et on afficherait la date de dernière mise à jour.

**Q4. Comment gérer les questions dont la réponse dépend du secteur ou de la taille de l'entreprise ?**

Le prompt système demande au LLM de signaler explicitement lorsqu'une réponse dépend de critères variables (effectif, convention collective, ancienneté). Exemple : l'obligation de CSE varie selon que l'entreprise dépasse 11 ou 50 salariés — le LLM le précise dans sa réponse.

**Q5. Comment distinguer une question factuelle d'une question nécessitant une interprétation juridique ?**

Le prompt système impose au LLM de ne répondre que sur la base des articles fournis en contexte, et de rediriger vers un juriste dès que la question dépasse la simple lecture d'un article (litiges, cas particuliers, cumul de règles complexes). Le disclaimer légal est présent sur chaque réponse sans exception.

---

## Choix techniques

| Composant | Choix | Justification |
|---|---|---|
| Embeddings | `distiluse-base-multilingual-cased-v2` | Modèle multilingue performant sur le français juridique |
| Index vectoriel | FAISS `IndexFlatIP` | Similarité cosinus exacte via produit interne sur vecteurs normalisés |
| Persistance | `.faiss` + `.json` | Index rechargeable sans réindexation (`faiss.write_index` / `faiss.read_index`) |
| Chunking | 1 article = 1 chunk (taille_max=500) | Articles courts et atomiques ; la fonction `chunker()` gère les cas longs |
| LLM | `llama-3.3-70b-versatile` via Groq | Meilleure compréhension juridique que le 8b, latence acceptable |
| Texte embeddé | `Article L3121-27 — titre [section] : texte` | Enrichissement du vecteur avec le numéro et le contexte thématique |
