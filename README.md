# rag

## Getting started

Clone the repository, set up the virtual environment, and install the required packages

```
cd <project-name>

python3 -m venv .venv

. .venv/bin/activate

pip install -r requirements.txt
```

## Store your OpenAI API key

Copy the example env file

`cp .env.example .env`

## Data indexing

`python3 -m importer.load_and_process`

## Development

`python3 -m app.server`
