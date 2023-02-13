# labelling_explanation

## Installation
Following the installation tips in Snorkel tutorial and Anchor

```
# [OPTIONAL] Activate a virtual environment
pip3 install --upgrade virtualenv
virtualenv -p python3 labelling
source labelling/bin/activate

# Install requirements
pip3 install -r snorkel/requirements.txt
pip3 install -r snorkel/spam/requirements.txt

# Install anchor
# If you use pip to install anchor package, there will be errors when using BERT
python anchor/setup.py install

python -m spacy download en_core_web_lg
pip install torch transformers spacy && python -m spacy download en_core_web_sm


# Launch the Jupyter notebook interface
jupyter notebook snorkel/spam
```
Open `01_spam_tutorial.ipynb`, then just run all cells and go directly to the last section `Explaining a labelling prediction using Anchor`.
