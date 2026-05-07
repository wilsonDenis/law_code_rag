# Compte rendu  Assistant Code du Travail (RAG)

## Décisions de conception

**Constitution du corpus**
Le Code du travail complet est disponible via l'archive LEGI sur data.gouv.fr, mais le fichier téléchargé s'est révélé être un delta de mise à jour (archive horodatée ne contenant que les articles récemment modifiés, essentiellement des articles réglementaires `D`). Faute de fond complet, nous avons constitué un corpus manuel de 48 articles législatifs (`L`) soigneusement choisis sur legifrance.gouv.fr, couvrant les 8 thèmes prioritaires du sujet. Un script `parse_corpus.py` est fourni pour régénérer le corpus depuis un fond LEGI complet lorsque disponible.

**Stratégie de chunking**
Les articles du Code du travail sont naturellement courts (100–600 caractères), ce qui les rend atomiques par construction. La fonction `chunker()` implémentée dans `indexation.py` retourne donc chaque article comme un chunk unique (taille_max=500). Pour les rares articles plus longs contenant des listes numérotées (`1°, 2°, 3°…`), la fonction segmente sur ces séparateurs légaux et applique un chevauchement de 50 caractères. Ce choix préserve la cohérence juridique : découper un article au milieu d'une énumération de cas légaux priverait chaque chunk de son contexte.

**Enrichissement des vecteurs**
Chaque article est embeddé sous la forme `"Article L3121-27 — Durée légale du travail [Durée du travail] : texte"` plutôt que le texte seul. Cela permet à la recherche vectorielle de trouver un article même si l'utilisateur formule sa question en citant un numéro d'article ou un thème.

**Choix de l'index FAISS**
Nous utilisons `IndexFlatIP` (produit interne) avec des vecteurs normalisés L2, ce qui est équivalent à la similarité cosinus. Ce choix garantit une recherche exacte (pas approximative), suffisante pour un corpus de 48 articles. L'index est sauvegardé via `faiss.write_index()` et rechargé via `faiss.read_index()` sans réindexation.

## Difficultés rencontrées

**Segfault FAISS sur macOS Apple Silicon**
L'appel à `model.encode()` de sentence-transformers causait un segfault (exit code 139) sur macOS avec PyTorch 2.11 et le backend MPS activé. Le problème venait du parallélisme des tokenizers OpenMP conflictant avec MPS. Solution : définir `TOKENIZERS_PARALLELISM=false` et `OMP_NUM_THREADS=1` avant tout import numpy/torch, en tête de chaque script.

**Archive LEGI delta vs fond complet**
L'archive téléchargée depuis data.gouv.fr était un fichier de mise à jour incrémentale contenant seulement 396 articles (dont 73 en vigueur), majoritairement des articles `D` hors de nos thèmes. Le parser XML `parse_corpus.py` fonctionne correctement mais nécessite le fond complet LEGI pour produire un corpus exhaustif.

**Compatibilité ChromaDB → FAISS**
Le projet a d'abord été développé avec ChromaDB (approche du cours de démonstration), puis migré vers FAISS conformément aux exigences du TP. La migration a nécessité de réécrire la classe `VectorDB`, d'implémenter manuellement la sauvegarde/chargement de l'index, et de gérer le format numpy float32 contiguous pour la compatibilité FAISS.
