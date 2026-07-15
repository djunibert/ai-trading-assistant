# Chaine CI/CD

## CI

Le workflow `.github/workflows/ci.yml` s'execute sur les pull requests et les pushes vers `main` ou `develop`.

Il effectue les controles suivants :

- installation des dependances Python apres normalisation de `requirements.txt` en UTF-8 ;
- compilation de `backend`, `src` et `tests` ;
- execution des tests unitaires stables ;
- build Docker de verification.

## CD

Le workflow `.github/workflows/cd.yml` s'execute sur chaque push vers `main` et peut aussi etre lance manuellement.

Il publie l'image Docker dans GitHub Container Registry :

```text
ghcr.io/<owner>/<repository>:latest
ghcr.io/<owner>/<repository>:<commit-sha>
```

Le deploiement SSH est optionnel. Pour l'activer, ajouter la variable de depot :

```text
DEPLOY_ENABLED=true
```

Puis ajouter les secrets GitHub suivants :

```text
DEPLOY_HOST
DEPLOY_USER
DEPLOY_SSH_KEY
```

Le serveur cible doit disposer de Docker et des dossiers suivants :

```text
/opt/ai-trading-system/data
/opt/ai-trading-system/models
```

Le modele attendu par defaut est :

```text
/opt/ai-trading-system/models/random_forest_v1.pkl
```

## Execution locale

```bash
docker compose up --build
```

L'API sera disponible sur :

```text
http://localhost:8000
```
