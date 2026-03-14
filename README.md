# NLP-Powered Urdu Articles Search Engine

An information retrieval system for Urdu-language articles that implements three classical scoring models to rank documents against user queries.

![Search Engine Overview](https://github.com/zainali89/NLP-Powered-Urdu-Articles-Search-Engine/assets/75775907/ef07e696-e5e3-43ef-9f87-dc6d777c9869)

## Features

- **TF-IDF** -- Term Frequency-Inverse Document Frequency scoring with logarithmic term weighting.
- **Okapi BM25** -- Probabilistic retrieval model with document-length normalisation (k1=1.2, b=0.75).
- **Dirichlet Smoothing** -- Language-model approach that smooths document term probabilities with a collection-level background model.
- Handles Urdu script tokenisation and stopword removal.
- Pre-built inverted index for fast lookup (`term_info.txt`, `term_index.txt`).
- Ten sample Urdu queries covering topics such as crime, protests, smuggling, and labour rights.

## Tech Stack

| Component | Library |
|-----------|---------|
| HTML parsing | BeautifulSoup 4 |
| Tokenisation | NLTK (`word_tokenize`) |
| Numeric computation | NumPy |
| Language | Python 3.8+ |

## Project Structure

```
.
├── main.py                  # Search engine (TF-IDF, BM25, Dirichlet)
├── requirements.txt         # Python dependencies
├── Urdu stopwords.txt       # Urdu stopword list
├── term_info.txt            # Term metadata index
├── term_index.txt           # Inverted index postings
├── Documents/               # Directory of Urdu HTML articles
├── tf_idf_score.txt         # Output: TF-IDF ranked results
├── okapi_bm25_score.txt     # Output: BM25 ranked results
└── dirichlet_smoothing_score.txt  # Output: Dirichlet ranked results
```

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/zainali89/NLP-Powered-Urdu-Articles-Search-Engine.git
   cd NLP-Powered-Urdu-Articles-Search-Engine
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (first time only)

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

5. **Prepare data files**

   Make sure the following are in the project root:
   - `Documents/` -- a directory containing Urdu HTML article files
   - `term_info.txt` -- term metadata index
   - `term_index.txt` -- inverted index postings
   - `Urdu stopwords.txt` -- Urdu stopwords, one per line

## Usage

Run the search engine:

```bash
python main.py
```

This will score all ten built-in queries using all three models and write ranked results to:

- `tf_idf_score.txt`
- `okapi_bm25_score.txt`
- `dirichlet_smoothing_score.txt`

### Configuration via environment variables

You can override default paths:

```bash
export DOCUMENTS_DIR=path/to/articles
export TERM_INFO_FILE=path/to/term_info.txt
export TERM_INDEX_FILE=path/to/term_index.txt
export STOPWORDS_FILE=path/to/stopwords.txt
python main.py
```

## Output Format

Each output line follows the format:

```
query_id  document_name  rank  score  run_id
```

Example:

```
1  article_042.html  1  3.8721  run 1
1  article_017.html  2  2.5134  run 1
```
