# -*- coding: utf-8 -*-
from rbbm_src.labelling_func_src.src.classes import lfunc_dec, make_keyword_lf,keyword_lookup, SPAM, HAM, ABSTAIN
from textblob import TextBlob
import re
from snorkel.preprocess import preprocessor
from snorkel.labeling.lf.nlp import nlp_labeling_function
from snorkel.preprocess.nlp import SpacyPreprocessor
import nltk

## -----------------Keyowrd Checks------------------------------ # 

@lfunc_dec(words_used = ['check'])
def check(x):
    return SPAM if "check" in x.text.lower() else ABSTAIN

"""Spam comments talk about 'my channel', 'my video', etc."""
keyword_my = make_keyword_lf(keywords=["my"])

"""Spam comments ask users to subscribe to their channels."""
keyword_subscribe = make_keyword_lf(keywords=["subscribe"])

"""Spam comments make requests rather than commenting."""
keyword_please = make_keyword_lf(keywords=["please", "plz"])

"""Ham comments actually talk about the video's content."""
keyword_song = make_keyword_lf(keywords=["song", "songs"], label=HAM)

"""Shakira is HAM"""
keyword_shakira = make_keyword_lf(keywords=["shakira"], label=HAM)

"""google is SPAM :）"""
keyword_google = make_keyword_lf(keywords=["google"])

"""Spam comments related to money"""
keyword_money = make_keyword_lf(keywords=["money", "$"])

# adding some new LFs
keyword_fb = make_keyword_lf(keywords=["facebook"])

keyword_twitter = make_keyword_lf(keywords=["twitter"])

keyword_free = make_keyword_lf(keywords=['free'])

keyword_gift = make_keyword_lf(keywords=['gift'])

keyword_whynot = make_keyword_lf(keywords=["why not"])

keyword_love = make_keyword_lf(keywords=['love'], label=HAM)


@lfunc_dec()
def nothttp(x):
    """HAM if not https"""
    return HAM if not re.search(r"http", x.text, flags=re.I) else ABSTAIN
# -----------------NLP domain related -----------------------------# 

@preprocessor()
def textblob_sentiment(x):
    scores = TextBlob(x.text)
    x.polarity = scores.sentiment.polarity
    x.subjectivity = scores.sentiment.subjectivity
    # print(x.subjectivity)
    return x

@lfunc_dec(pre=[textblob_sentiment], )
def textblob_polarity(x):
    return HAM if x.polarity > 0.9 else ABSTAIN

@lfunc_dec(pre=[textblob_sentiment], )
def textblob_subjectivity(x):
    return HAM if x.subjectivity >= 0.5 else ABSTAIN

@lfunc_dec()
def short_comment(x):
    """Ham comments are often short, such as 'cool video!'"""
    return HAM if len(x.text.split()) < 5 else ABSTAIN

@lfunc_dec()
def regex_link(x):
    return SPAM if re.search(r"http", x.text, flags=re.I) else ABSTAIN

# The SpacyPreprocessor parses the text in text_field and
# stores the new enriched representation in doc_field
spacy = SpacyPreprocessor(text_field="text", doc_field="doc")

@lfunc_dec(pre=[spacy])
def has_person_nlp(x):
    """Ham comments mention specific people and are short."""
    if len(x.doc) < 20 and any([ent.label_ == "PERSON" for ent in x.doc.ents]):
        return HAM
    else:
        return ABSTAIN

@lfunc_dec()
def pos_vbprps_stuff(x):
    """VB with PRP$(subscribe my channel)"""
    text = nltk.word_tokenize(x.text)
    posstr = ''.join([t[1] for t in nltk.pos_tag(text)])
    return SPAM if 'VBPRP$' in posstr else ABSTAIN

############# TWITTER FUNCS ##################################################

@lfunc_dec(pre=[textblob_sentiment])
def clinton_neg(x):
    return SPAM if (x.polarity<=0 and re.search(r".*Clinton.*", x.text, flags=re.I)) else ABSTAIN


keyword_birthday = make_keyword_lf(keywords=['birthday'])


# lattice
@lfunc_dec()
def everything_ham(x):
    """Ham comments mention specific people and are short."""
    return HAM

@lfunc_dec()
def everything_spam(x):
    """Ham comments mention specific people and are short."""
    return SPAM

@lfunc_dec()
def keyword_subscribe_my(x):
    if('subscribe' in x.text and 'my' in x.text):
        return SPAM
    else:
        return ABSTAIN

@lfunc_dec()
def keyword_shakira_short_comment(x):
    if(len(x.text.split()) < 5 and 'shakira' in x.text):
        return HAM
    else: 
        return ABSTAIN

@lfunc_dec()
def keyword_love_song(x):
    if('love' in x.text and (('song' in x.text) or ('songs' in x.text))):
        return HAM
    else: 
        return ABSTAIN


# labelling functions we have

LFs = [
textblob_polarity,
textblob_subjectivity,
regex_link,
has_person_nlp,
pos_vbprps_stuff,
check,
keyword_my,
keyword_subscribe,
keyword_please,
keyword_song,
keyword_shakira,
keyword_google,
keyword_money,
keyword_fb,
keyword_twitter,
keyword_free,
keyword_gift,
keyword_whynot,
nothttp,
short_comment,
]

LFs_running_example = [
textblob_subjectivity,
keyword_my,
keyword_please,
keyword_song,
nothttp,
short_comment
]


LFs_smaller = [check,
 keyword_my,
 keyword_subscribe,
 keyword_please,
 keyword_song,
 keyword_shakira,
 keyword_google,
 keyword_money,
 keyword_fb]
 # keyword_twitter,
 # keyword_free,
 # keyword_gift,
 # keyword_whynot,
 # nothttp,
 # textblob_polarity,
 # textblob_subjectivity,
 # short_comment,
 # regex_link,
 # has_person_nlp,
 # pos_vbprps_stuff]

 # sentence: SEE SOME MORE SONG OPEN GOOGLE AND TYPE Shakira GuruOfMovie﻿ 
 # correct_label : 1 
 # pred_label: 0 
 # vectors: -1,-1,-1,-1,0,0,1,-1,-1,-1,-1,-1,-1,0,-1,0,-1,-1,-1,-1,-1

# lattice_lfs = [ 
#     keyword_subscribe, keyword_my, keyword_shakira, short_comment,
#         keyword_subscribe_my, keyword_shakira_short_comment]

# lattice_dict = {
# # everything_spam:[keyword_subscribe,keyword_my,keyword_subscribe_my],
# # everything_ham:[keyword_shakira,short_comment,keyword_shakira_short_comment],
# keyword_subscribe:[keyword_subscribe_my],
# keyword_my:[keyword_subscribe_my],
# keyword_shakira:[keyword_shakira_short_comment],
# short_comment:[keyword_shakira_short_comment],

# keyword_shakira_short_comment:[],
# keyword_subscribe_my:[]
# }


twitter_lfs = [check,regex_link, keyword_subscribe, keyword_song, has_person_nlp, keyword_gift, keyword_free,
keyword_money,keyword_please, clinton_neg,keyword_birthday]



lattice_lfs = [
# keyword_love, keyword_song, keyword_love_song,
    keyword_subscribe, keyword_my, keyword_shakira, short_comment,
        keyword_subscribe_my, keyword_shakira_short_comment]

        # 0,0,-1,1,1,-1,-1,-1,-1

lattice_dict = {
# everything_spam:[keyword_subscribe,keyword_my,keyword_subscribe_my],
# everything_ham:[keyword_shakira,short_comment,keyword_shakira_short_comment],
keyword_subscribe:[keyword_subscribe_my],
keyword_my:[keyword_subscribe_my],
# keyword_love:[keyword_love_song],
# keyword_song:[keyword_love_song],
keyword_shakira:[keyword_shakira_short_comment],
short_comment:[keyword_shakira_short_comment],
keyword_shakira_short_comment:[],
keyword_subscribe_my:[],
# keyword_love_song:[]
}
