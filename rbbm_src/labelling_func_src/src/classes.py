from snorkel.labeling import (
    LabelingFunction, 
    labeling_function
    )
from snorkel.preprocess import BasePreprocessor,preprocessor
from typing import *
import re
from dataclasses import dataclass
from typing import *
import pandas as pd 

ABSTAIN = -1
HAM = 0
SPAM = 1

class lfunc(LabelingFunction):
    
    def __init__(self,
        name: str,
        f: Callable[..., int],
        cgsize: Optional[float]=1,
        words_used: Optional[List[str]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None,
        ):
        super(lfunc, self).__init__(name,f,resources,pre)
        self.cgsize = cgsize
        self.words_used = words_used
    
    def __repr__(self):
        # return f"func: {self.name}, words_used: {self.words_used}"
        return f"func: {self.name}"

    def __eq__(self, other):
        return isinstance(other, lfunc) and self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.name)

class lfunc_dec(labeling_function):
    def __init__(self,
        cgsize: Optional[float]=1,
        words_used: Optional[List[str]] = None,
        name: Optional[str] = None,
        resources: Optional[Mapping[str, Any]] = None,
        pre: Optional[List[BasePreprocessor]] = None):
        super(lfunc_dec, self).__init__(name,resources,pre)
        self.cgsize = cgsize
        self.words_used = words_used

    def __call__(self, f: Callable[..., int]) -> lfunc:

        if(self.pre):
            self.cgsize += len(self.pre)

        name = self.name or f.__name__

        return lfunc(name=name, f=f, resources=self.resources, pre=self.pre, 
            cgsize=self.cgsize, words_used=self.words_used)

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""
    def __iter__(self):
        cwd = os. getcwd()
        corpus_path = datapath(f'{cwd}/spam_corpus.txt')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)

def keyword_lookup(x, keywords, label):
    keywords = [w.lower() for w in keywords]
    if any(word in x.text.lower() for word in keywords):
        return label
    return ABSTAIN

def make_keyword_lf(keywords, label=SPAM, name=None):
    if(name):
        func_name = name
    else:
        func_name = f"keyword_{keywords[0]}"
    return lfunc(
        name=func_name,
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
        words_used = keywords
    )

def clean_text(text):
    text = text.encode("ascii", "ignore").decode()
    text = re.sub('[^a-zA-Z0-9\s\.\,\!\?\;\:\'\"]', ' ', text)  # Allowing common punctuation
    # punctuationfree="".join([i for i in text if i not in text.lower()])
    res = text.lower()

    # there are some corner cases where the processed text is too short or empty
    # use this part to retur None so we can filter them out
    if(len(res.strip().split(' '))<=2):
        return None
    return text.lower()

@dataclass
class lf_input_internal:
    """
    Input object which has all the arguments
    needed to execute the framework for labelling
    function setting
    """
    funcs:List[lfunc]=None
    func_vectors:Dict[lfunc, List[int]]=None
    filtered_sentences_df: pd.DataFrame=None
    expected_label:int=None
    predicted_label:int=None
    sentence_of_interest:str=None
    model:'typing.Any'=None

