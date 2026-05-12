Dossier des textes sources pour LexMaroc
========================================

Placez ici vos fichiers PDF des codes juridiques marocains (en français),
ou indiquez un autre dossier via la variable d'environnement LEXMAROC_DATA_DIR
(chemin absolu ou relatif à la racine du dossier lexmaroc).

Conseils de nommage pour la reconnaissance automatique du code :
- code_travail.pdf          -> Code du Travail
- code_penal.pdf            -> Code Pénal
- code_famille.pdf          -> Code de la Famille
- code_commerce.pdf         -> Code de Commerce
- code_obligations.pdf      -> Code des Obligations et Contrats
- code_procedure_civile.pdf -> Code de Procédure Civile

Les articles doivent commencer dans le PDF par des formes telles que :
« Article 1 », « Art. 1 », « ARTICLE 1 » ou « Article premier ».

Après avoir copié les PDF, placez-vous dans le dossier lexmaroc (celui qui
contient app.py et le dossier backend), puis lancez :
python -m backend.ingestion.ingest
