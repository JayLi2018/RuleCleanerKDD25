import pandas as pd
from typing import *
from snorkel.labeling import (
    LabelingFunction, 
    labeling_function, 
    PandasLFApplier, 
    LFAnalysis,
    filter_unlabeled_dataframe
    )
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from snorkel.preprocess import BasePreprocessor,preprocessor
from rbbm_src.labelling_func_src.src.utils import load_spam_dataset
import logging
# from gensim.test.utils import datapath
# from gensim import utils
# import gensim.models
from itertools import chain, combinations
import os
from rbbm_src.labelling_func_src.src.classes import *
from nltk.tokenize import word_tokenize 

logger = logging.getLogger(__name__)

# DISPLAY_ALL_TEXT = False

# pd.set_option("display.max_colwidth", 0 if DISPLAY_ALL_TEXT else 50)
pd.set_option('display.max_colwidth', None)

# For clarity, we define constants to represent the class labels for spam, ham, and abstaining.

# functions
    # 1. reward function
    # 2. complexity definition
    # 3. relevance calculation
    # 4. labeling model: could be any black/white box model
       

# parameters
    # 1. reward stopping point
    # 2. set size?


# inherit existing labeling functions from snorkel
# to add some additional attributes based on our 
# definitions


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))


# def repair(sentence_of_interest, other_sentences, expected_label, 
#     max_num_similar_sentences, labelling_model, repair_candidates, lfs):

#     sentences = MyCorpus()
    
#     model = gensim.models.Word2Vec(sentences=sentences)

#     sentences_to_maximize = similar_sentences(sentence_of_interest, other_sentences, max_num_similar_sentences)
#     # return the subgroup of the df that are topk ranked in terms of the given similarity measure

#     repaired_candidates = [repair_function(r) for r in repair_candidates]
#     # preprocessing the repaired functions

#     if(not repair_candidates):
#         logger.warning('the repair repair_candidates is empty! cant repair')
#         exit()
#     else:
#         for r in repair_candidates:
#             new_func = repair_function(r)
#         for i in range(len(lfs)):
#             if(repr(lfs[i])==repr(new_func)):
#                 lfs[i] = new_func
#                 break

#     applier = PandasLFApplier(lf=lfs)


# def similar_sentences(sentence_of_interest, other_sentences, num_to_return, model):

#     sims = []
#     for (sent1, sent2) in zip(sentences1, sentences2):

#         tokens1 = sent1.tokens_without_stop if use_stoplist else sent1.tokens
#         tokens2 = sent2.tokens_without_stop if use_stoplist else sent2.tokens

#         tokens1 = [token for token in tokens1 if token in model]
#         tokens2 = [token for token in tokens2 if token in model]
        
#         if len(tokens1) == 0 or len(tokens2) == 0:
#             sims.append(0)
#             continue
        
#         tokfreqs1 = Counter(tokens1)
#         tokfreqs2 = Counter(tokens2)
        
#         weights1 = [tokfreqs1[token] * math.log(N/(doc_freqs.get(token, 0)+1)) 
#                     for token in tokfreqs1] if doc_freqs else None
#         weights2 = [tokfreqs2[token] * math.log(N/(doc_freqs.get(token, 0)+1)) 
#                     for token in tokfreqs2] if doc_freqs else None
                
#         embedding1 = np.average([model[token] for token in tokfreqs1], axis=0, weights=weights1).reshape(1, -1)
#         embedding2 = np.average([model[token] for token in tokfreqs2], axis=0, weights=weights2).reshape(1, -1)

#         sim = cosine_similarity(embedding1, embedding2)[0][0]
#         sims.append(sim)


#     for i,s in other_sentences.iterrows():
#         if()


def keyword_search(sentence, label_func):
    # used to find if a labelling function has any 
    # association with any words from the sentence 
    # of interest

    words = [x for x in sentence.split()]
    df_original = pd.DataFrame(data={'text':[sentence]})

    res = []

    for w in words:
        s_wo_w = ' '.join([x for x in words if x!=w])
        df_s_wo_w = pd.DataFrame(data={'text':[s_wo_w]})
        if(label_func(df_original.iloc[0])!=label_func(df_s_wo_w.iloc[0])):
            res.append(w)

    return res  


def update_cand(cur_words=None, cur_funcs=None, w_to_remove=None, f_to_remove=None):
    
    logger.warning(cur_words)
    if(w_to_remove):
        for w in w_to_remove:
            logger.warning(w)
        # do this way because a word that is related to func could have been added to the result
            if w in cur_words:
                cur_words.remove(w)
        return cur_words

    if(f_to_remove is not None):
        return [f for f in cur_funcs if f.name!=f_to_remove.name]


def sentence_filter(words: List[str], sentence: str):
    lsenetnce = sentence.lower()
    return all([x in word_tokenize(lsenetnce) for x in words])

def delete_words(words: List[str], sentence: str):
    lsenetnce = sentence.lower()
    for w in words:
        lsenetnce = lsenetnce.replace(w,'')

    return lsenetnce

def reward_function(
    label_for_s: int,
    sentence_of_interest: str,
    cur_func_set: List[lfunc], 
    sentences: pd.DataFrame,
    w_add: Optional[str]=None,
    model_type: Optional[str]='majority',
    cur_word_set: Optional[List[str]] = None, 
    f_add: Optional[lfunc]=None
    ):

        # relevance 
        rel, num_words_added, num_func_added = relevance(sentence_of_interest, 
                                                          cur_word_set,
                                                          cur_func_set,
                                                          f_add,
                                                          w_add
                                                          )
        # sum of relevant sentences which has 
        # same label with sentence of interest

        # filter sentences to get sentences matching 
        # current word set

        # normalizer
        if(cur_word_set):
            norm = len(cur_word_set) * len(cur_func_set) + 2
        else:
            norm = 1 * len(cur_func_set) + 2

        # logger.warning(f"normal factor: {norm}")

        if(cur_word_set):
            sentences_match_cur_words = sentences[sentences['text'].apply(lambda s: 
                sentence_filter(cur_word_set, s))]
        else:
            sentences_match_cur_words = sentences

        if(sentences_match_cur_words.empty):
            # if no sentences match the word set then sum will be 0
            sentence_value=0
            support=0
        else:
            # apply majority voter to get labels 
            # predicted by the model
            applier = PandasLFApplier(lfs=cur_func_set)
            labels = applier.apply(df=sentences_match_cur_words, progress_bar=False)
            if(model_type=='majority'):
                model = MajorityLabelVoter()
                model_results = model.predict(L=labels)
            else:
                model = LabelModel(cardinality=2, verbose=True)
                # snorkel needs to get an estimator using fit function first
                model.fit(L_train=labels, n_epochs=500, log_freq=100, seed=123)
                # filter out unlabeled data and only predict those that receive signals
                probs_train = model.predict_proba(L=labels)
                df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
                    X=sentences_match_cur_words, y=probs_train, L=labels
                )
                # reset df_train to those receive signals
                df = df_train_filtered.reset_index(drop=True)
                df_train = applier.apply(df=df, progress_bar=False)
                model.fit(L_train=df_train, n_epochs=500, log_freq=100, seed=123)
                model_results = model.predict(L=df_train)


            support = len([x for x in model_results if x==label_for_s])
            sentence_value = support /len(sentences_match_cur_words)

        # logger.warning(f"cur_word_set:{cur_word_set}")
        # logger.warning(f"cur_func_set: {cur_func_set}")
        # logger.warning(f"len(cur_word_set)={len(cur_word_set)} len(cur_func_set)={len(cur_func_set)}")
        # logger.warning(f"support={support}, sentence_value={sentence_value}, norm={norm}")


        while(num_func_added>0):
            cur_func_set.pop()
            num_func_added-=1

        while(num_words_added>0):
            cur_word_set.pop()
            num_words_added-=1

        return support, (rel* sentence_value) / norm


def relevance(sentence_of_interest: str, 
              cur_word_set: List[str],
              cur_func_set: List[lfunc],
              f: Optional[lfunc]=None, 
              w: Optional[str]=None,
              ) -> float : 
    """
    A measure of how closely related the given labelling
    function to the sentence of interest.
    """
    num_words_added = 0
    num_func_addded = 0

    if(f is not None):
        complexity = f.cgsize
        cur_func_set.append(f)
        num_func_addded = 1
    else:
        complexity = 1

    if(f is not None and f.words_used is not None):

        words_related = [x for x in f.words_used if x.lower() in sentence_of_interest]
        
        context = len(words_related)
        if(context>=len(f.words_used)):
            cur_word_set.extend(f.words_used)
            num_words_added+=len(f.words_used)
        else:
            context=0
    elif(w is not None):
        context = 1
        if(w not in cur_word_set):
            num_words_added+=1
            cur_word_set.append(w)
    else: 
        context = 0

    return (1/complexity) * context, num_words_added, num_func_addded


def postprocess(results, min_value_thresh, model_type, 
    label_for_s,sentences):
    
    """
    input the results from the bottom up algorithm
    the postprocessing step will establish connections
    between words and functions
    min_value_thresh: threshold that argmax({w} \cuip {f_i})
    has to be bigger to be included
    model_type: majority / snorkel
    """
    words_to_check = results['words']
    funcs_to_check = results['funcs']
    # sentences = sentences[~(sentences['text']==sentence_of_interest)]
    res = {}

    logger.warning(f"postprocessing:")
    logger.warning(f"results : {results}")

    if(model_type=='majority'):
        model = MajorityLabelVoter() 
        for w in words_to_check:
            logger.warning(f"cheking word : {w}")
            highest_value, func_to_pair = -1, None
            for f in funcs_to_check:
                applier = PandasLFApplier(lfs=[f])
                sentences_match_cur_words = sentences[sentences['text'].apply(lambda s: 
                    sentence_filter([w], s))]
                labels = applier.apply(df=sentences_match_cur_words, progress_bar=False)
                model_results = model.predict(L=labels)
                support = len([x for x in model_results if x==label_for_s])
                sentence_value = support /len(sentences_match_cur_words)
                if(highest_value<sentence_value):
                    highest_value, func_to_pair = sentence_value, f

            logger.warning(f"highest_value for a function combined with word {w}: {func_to_pair} value :{highest_value}")
            if(highest_value>=min_value_thresh):
                res[w] = func_to_pair
    else:
        for w in words_to_check:
            logger.warning(f"cheking word : {w}")
            highest_value, func_to_pair = -1, None
            for f in funcs_to_check:
                logger.warning(f"f.name: {f.name}")
                if(f.name == 'keyword_ '):
                    continue
                logger.warning(f"func being checked with {w}: {f}")
                model = LabelModel(cardinality=2, verbose=True)
                applier = PandasLFApplier(lfs=[keyword_empty, keyword_the, f])
                sentences_match_cur_words = sentences[sentences['text'].apply(lambda s: 
                    sentence_filter([w], s))]
                labels = applier.apply(df=sentences_match_cur_words, progress_bar=False)
                # snorkel has to have at leaste 3 functions to make a prediction
                # snorkel needs to get an estimator using fit function first
                model.fit(L_train=labels, n_epochs=500, log_freq=100, seed=123)
                # filter out unlabeled data and only predict those that receive signals
                probs_train = model.predict_proba(L=labels)
                df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
                    X=sentences_match_cur_words, y=probs_train, L=labels
                )
                if(df_train_filtered.empty):
                    continue
                # reset df_train to those receive signals
                df = df_train_filtered.reset_index(drop=True)
                logger.debug(df)
                df_train = applier.apply(df=df, progress_bar=False)
                model.fit(L_train=df_train, n_epochs=500, log_freq=100, seed=123)
                model_results = model.predict(L=df_train)
                support = len([x for x in model_results if x==label_for_s])
                sentence_value = support /len(sentences_match_cur_words)
                if(highest_value<sentence_value):
                    highest_value, func_to_pair = sentence_value, f
            logger.warning(f"highest_value for a function combined with word {w}: {func_to_pair} value :{highest_value}")
            if(highest_value>=min_value_thresh):
                res[w] = func_to_pair
    return res


def run_algo(
    sentence_of_interest: str,
    sentences: pd.DataFrame,
    predicted_label: int,
    labeling_funcs: List[lfunc],
    model_type: Optional[str]='majority',
    stoping_criterion: Optional[str] = 'exhaust',
    support_thresh: Optional[int] = 0, # only useful when choosing support stopping criterion
    reward_thresh: Optional[float] = 0  # only useful wuen choosing reward stopping criterion
    ) -> dict:
    # %%
    # initialze function set: choose one with the highest reward
    result = {'funcs':[], 'words':[]}
    
    sentences = sentences[~(sentences['text']==sentence_of_interest)]

    # i=1
    cur_func_cand = labeling_funcs

    # initialization step, given the reward func definition we need to 
    # at least have some functions to work with. This step is different
    # between majority model and snorkel, with snorkel has stricter
    # requirements
    
    sdf = pd.DataFrame(data={'text':[sentence_of_interest]})
    # sentence of interest as a dataframe, use this
    # to filter out the functions that would give different
    # label from the predicted label

    if(model_type=='majority'):
        # majority only needs one function to start working
        # we add first one by selecting the function that gives
        # the highest reward
        f_to_add, highest_reward = None, 0
        for f in cur_func_cand:
            applier = PandasLFApplier(lfs=[f])
            flabel = applier.apply(df=sdf, progress_bar=False)[0][0]
            if(flabel!=predicted_label):
                continue
            sup, cur_reward = reward_function(label_for_s = predicted_label, 
                                model_type=model_type,
                                sentence_of_interest=sentence_of_interest,
                                cur_func_set= result['funcs'],
                                sentences = sentences,
                                cur_word_set = result['words'],
                                f_add=f)

            logger.warning(f"initialization : f: {f.name}, reward: {cur_reward}")

            if(stoping_criterion=='support'):
                if(sup<support_thresh):
                    continue
            if(cur_reward>highest_reward):
                f_to_add, highest_reward = f, cur_reward

        if(f_to_add is None):
            logger.warning("cant initiate bottom up algorithm: number of func not fulfilled")
            exit()
        else:
            cur_func_cand = update_cand(cur_funcs=cur_func_cand, f_to_remove=f_to_add)
            result['funcs'].append(f_to_add)
            result['words'].extend(f_to_add.words_used)

    else:
        # for snorkel initial number of functions need to be bigger than 3
        # we select 3 using relevance only since reward function need to use
        # current function set and there's no way for it to work with less than 3
        # functions
        func_relevances = [[f,0] for f in cur_func_cand]
        for t in func_relevances:
            t[1], num_words_added, num_func_added = relevance(sentence_of_interest=sentence_of_interest, 
                                                              cur_word_set=result['words'],
                                                              cur_func_set=result['funcs'],
                                                              f=t[0]
                                                              )

            while(num_func_added>0):
                result['funcs'].pop()
                num_func_added-=1

            while(num_words_added>0):
                result['words'].pop()
                num_words_added-=1

        func_relevances.sort(key=lambda x: x[1], reverse=True)
        logger.warning(func_relevances)
        if(func_relevances[2][1]>0): 
        # make sure at least 3 functions have relevance to sentence of interest
            init_funcs, cur_func_cand = [ti[0] for ti in func_relevances[:3]], [tc[0] for tc in func_relevances[3:]]
            logger.debug(init_funcs)
            logger.debug(cur_func_cand)
            for f in init_funcs:
                result['funcs'].append(f)
                result['words'].extend(f.words_used)
        else:
            logger.warning('no enough relevant function to initialize snorkel label model')
            exit()

    logger.warning(result)

    # update the words available to choose from, dont include the words that 
    # have been added with its "associated" funcs
    cur_word_set_cand = list(set([w.lower() for w in sentence_of_interest.split()]))
    cur_word_set_cand = [w for w in cur_word_set_cand if w not in result['words']]

    while(cur_word_set_cand or cur_func_cand):

        wf_to_add, highest_reward = None, 0
        for w in cur_word_set_cand:
            sup, cur_reward = reward_function(label_for_s = predicted_label,
                                model_type=model_type,
                                sentence_of_interest=sentence_of_interest,
                                cur_func_set= result['funcs'],
                                sentences = sentences,
                                w_add = w, 
                                cur_word_set = result['words']
                               )
            # logger.warning(f"normal iteration: w: {w}, reward: {cur_reward}")

            if(stoping_criterion=='support'):
                if(sup<support_thresh):
                    continue

            if(cur_reward>highest_reward):
                wf_to_add, highest_reward = w, cur_reward

        for f in cur_func_cand:
            applier = PandasLFApplier(lfs=[f])
            flabel = applier.apply(df=sdf, progress_bar=False)[0][0]
            if(flabel!=predicted_label):
                continue
            sup, cur_reward = reward_function(label_for_s = predicted_label, 
                                model_type=model_type,
                                sentence_of_interest=sentence_of_interest,
                                cur_func_set= result['funcs'],
                                sentences = sentences,
                                cur_word_set = result['words'],
                                f_add=f)

            logger.warning(f"normal iteration: f: {f.name}, reward: {cur_reward}")

            if(stoping_criterion=='support'):
                if(sup<support_thresh):
                    continue

            if(cur_reward>highest_reward):
                support, wf_to_add, highest_reward = sup, f, cur_reward

        if(stoping_criterion=='exhaust'):
            if(highest_reward==0):
                break
        elif(stoping_criterion=='support'):
            if(sup<support_thresh):
                break
        elif(stoping_criterion=='reward'):
            if(highest_reward<reward_thresh):
                break
        else:
            logger.info("No valid stopping criterion specified")

        if(isinstance(wf_to_add, lfunc)):
            logger.warning(f'@@@@@@@@@@@@@@@@@  adding {wf_to_add.name}   @@@@@@@@@@@@@@@@@@\n')
            cur_func_cand = update_cand(cur_funcs=cur_func_cand, f_to_remove=wf_to_add)
            cur_word_set_cand = update_cand(cur_words=cur_word_set_cand, w_to_remove=wf_to_add.words_used)
            result['funcs'].append(wf_to_add)
            result['words'].extend(wf_to_add.words_used)
        if(isinstance(wf_to_add, str)):
            logger.warning(f'@@@@@@@@@@@@@@@@@@ adding {wf_to_add} @@@@@@@@@@@@@@@@@@@@@@@\n')
            cur_word_set_cand = update_cand(cur_words=cur_word_set_cand, w_to_remove=[wf_to_add])
            result['words'].append(wf_to_add)
        if(wf_to_add is None):
            break

        # logger.warning('\n\n\n')

    # logger.warning(result)

    return result

    # %%