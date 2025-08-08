# Install dependencies

``` pip install -r requirements.txt ```

# Usage

``` python main.py ```

main.py contains fastapi server with 3 endpoints
engine.py contains bert model for text anonymizing
text_extractor.py is wrapper for text extraction from files
logger_conf.py is logging config for all project, logging module must import from this file
test.py contains some test for basic project checking

# Docker
- build ```docker build -t ner-anonymizer:cpu .```
- run ```docker run --rm -p 8000:8000 ner-anonymizer:cpu```
