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
    text = re.sub('[^a-zA-Z]', ' ', text)
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
    lattice_dict:Dict[lfunc, lfunc]=None
    topk_funcs:List[lfunc]=None
    model:'typing.Any'=None


        # funcs=LFs, func_vectors=cached_vectors, 
        #             filtered_sentences_df=df_sentences_filtered, expected_label=soi_correct_label, predicted_label=soi_label, sentence_of_interest=soi, lattice_layers=lattice_layers



wrong_check_ids=[8,9,16,29,31,32,39,45,49,52,61,80,82,83,112,115,179,310,318,339,387,417,445,557,568,606,626,643,647,676,706,948,954,1075,1076,1081,1083,1155,1193,
1238,1246,1258,1259,1270,1274,1276,1277,1278,1283,1289,1293,1295,1301,1305,1306,1316,1319,1321,1322,1327,1332,1333,1335,1346,1358,1359,1360,1361,1362,
1385,1392,1403,1419,1422,1444,1452,1479,1486,1487,1488,1514,1534,1542,1559,1570,1577,1580,1737,1743,1747,1769,1773,1780,1782,1788,1814,1840,1873,1884,
1886,1889,1903,1920,1926]

correct_check_ids=[1,2,6,10,17,21,24,27,36,50,53,57,59,72,101,138,142,182,188,221,293,313,315,333,338,359,364,367,378,384,391,399,405,477,493,513,516,527,560,571,575,
638,671,691,980,1007,1097,1098,1103,1116,1117,1127,1149,1168,1243,1272,1323,1330,1368,1370,1380,1407,1410,1411,1424,1429,1445,1447,1449,1451,1457,1460,
1462,1464,1472,1474,1494,1497,1501,1504,1511,1518,1521,1526,1537,1539,1545,1565,1781,1834,1836,1894,1912,1913,1914,1918,1924,1925,1928,1930,1942]