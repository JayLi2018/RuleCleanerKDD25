from bottom_up import *
import numpy as np
import logging
import logconfig
import argparse
from lfs import snorkel_LFs, majority_LFs, ABSTAIN, HAM, SPAM
from snorkel.labeling import filter_unlabeled_dataframe
from func_responsibility import func_responsibility


logger = logging.getLogger(__name__)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Running experiments')

  parser.add_argument('-s','--stop_criterion', metavar="\b", type=str, default='exhaust', 
    help='the stop criterion of the algorithm: exhaust/support/reward(default: %(default)s)')

  parser.add_argument('-S','--support_thresh', metavar="\b", type=int, default=0, 
    help='the stoping threshold if stopping criterion is "support" (only effectiv if approach is anchor) (default: %(default)s)')

  parser.add_argument('-M','--model', metavar="\b", type=str, default='majority',
    help='the model used to get the label: majority/snorkel (default: %(default)s)')

  parser.add_argument('-R','--reward_thresh', metavar="\b", type=int, default=0, 
    help='the stoping threshold if stopping criterion is "reward" (only effectiv if approach is anchor) (default: %(default)s)')

  parser.add_argument('-t','--words_funcs_together', metavar="\b", type=str, default='true', 
    help='when adding function to result do we add related words too or not (only effectiv if approach is anchor) (default: %(default)s)')

  parser.add_argument('-l', '--log_level', metavar='\b', type=str, default='debug',
    help='loglevel: debug/info/warning/error/critical (default: %(default)s)')

  parser.add_argument('-u', '--user_provide', metavar='\b', type=str, default='yes',
    help='user select from all wrong labels? (default: %(default)s)')

  parser.add_argument('-A', '--approach', metavar='\b', type=str, default='casuality',
    help='approach: casuality based on anchor based (casuality/anchor) (default: %(default)s)')

  log_map = { 'debug': logging.DEBUG,
  'info': logging.INFO,
  'warning': logging.WARNING,
  'error': logging.ERROR,
  'critical': logging.CRITICAL
  }

  args=parser.parse_args()

  logger.info(' '.join(f'{k}={v}' for k, v in vars(args).items()))

  try:
    logconfig.root.setLevel(log_map[args.log_level])
  except KeyError as e:
    print('no such log level')

  df = load_spam_dataset()
  if(args.model=='majority'):
    LFs = all_LFs
  else:
    LFs = all_LFs

  applier = PandasLFApplier(lfs=LFs)

  df_train = applier.apply(df=df, progress_bar=False)

  if(args.model=='majority'):
    model = MajorityLabelVoter()
  else:
    model = LabelModel(cardinality=2, verbose=True)
    # snorkel needs to get an estimator using fit function first
    model.fit(L_train=df_train, n_epochs=500, log_freq=100, seed=123)
    # filter out unlabeled data and only predict those that receive signals
    probs_train = model.predict_proba(L=df_train)
    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
        X=df, y=probs_train, L=df_train
    )
    # reset df_train to those receive signals
    df = df_train_filtered.reset_index(drop=True)
    df_train = applier.apply(df=df, progress_bar=False)
    model.fit(L_train=df_train, n_epochs=500, log_freq=100, seed=123)
    # df.to_csv('snorkel.csv')

  df['pred'] = pd.Series(model.predict(L=df_train))
  # the wrong labels we get
  wrong_preds = df[(df['label']!=df['pred']) & (df['pred']!=ABSTAIN)]
  # wrong_preds.to_csv(f'{args.model}_{args.support_thresh}.csv')

  # wrong_preds = df[(df['label']!=df['pred'])]


  logger.debug(len(wrong_preds))
  # wrong_preds = wrong_preds.reset_index(drop=True)
  if(args.user_provide=='yes'):
    for index, row in wrong_preds.iterrows():
      print("--------------------------------------------------------------------------------------------")  
      print(f"setence#: {index} \n sentence: {row['text']} \n correct_label : {row['label']} \n pred_label: {row['pred']} \n")
    choices = input('please input sentence # of sentence, multiple sentences should be seperated using space')
    logger.debug(f"choices: {choices}")
    choice_indices = [int(x.strip()) for x in choices.split()]
    logger.debug(f"choice_indices: {choice_indices}")
    sentences_of_interest = list(wrong_preds.loc[choice_indices].text.values.astype(str))
    logger.debug(f"sentences_of_interest: {sentences_of_interest}")
  else:
    shakiras = df[(df['text'].str.contains("shakira")) | (df['text'].str.contains("Shakira"))]
    wrong_shakiras = shakiras[(shakiras['label']!=shakiras['pred'])]
    # wrong_shakiras.to_csv(f'{args.model}_wrong_shakiras.csv', index=False)
    # exit()
    logger.warning(f'we have {len(shakiras)} shakiras')
    for index, row in wrong_shakiras.iterrows():
      print("--------------------------------------------------------------------------------------------")  
      print(f"setence#: {index} \n sentence: {row['text']} \n correct_label : {row['label']} \n pred_label: {row['pred']} \n")
    choices = input('please input sentence # of sentence, multiple sentences should be seperated using space')
    logger.debug(f"choices: {choices}")
    choice_indices = [int(x.strip()) for x in choices.split()]
    logger.debug(f"choice_indices: {choice_indices}")
    sentences_of_interest = list(wrong_shakiras.loc[choice_indices].text.values.astype(str))
    logger.debug(f"sentences_of_interest: {sentences_of_interest}")

  structured_result = []

  logger.warning(sentences_of_interest)

  if(args.approach=='anchor'):
    for s in sentences_of_interest:
      soi_df = df[df['text']==s]
      # other_sentences = df[~df['text'].isin(s)]
      soi_label = list(soi_df['pred'].values.astype(int))[0]
      soi_correct_label = list(soi_df['label'].values.astype(int))[0]

      result_d = {'sentence':s,
      'correct_label': soi_correct_label,
      'predicted_label': soi_label,
      # 'repair_info': None
      }

      logger.debug(soi_label)
      soi = list(soi_df['text'].values.astype(str))[0].lower()
      result = run_algo(sentence_of_interest=soi,
          sentences=df,
          model_type=args.model,
          predicted_label=soi_label,
          labeling_funcs=LFs,
          stoping_criterion=args.stop_criterion,
          support_thresh=args.support_thresh, # only useful when choosing support stopping criterion
          reward_thresh=args.reward_thresh  # only useful wuen choosing reward stopping criterion
          )

      # repairs = repair(sentence_of_interest=soi_df, other_sentences=other_sentences,
      #   expected_label=soi_correct_label, max_num_similar_sentences=10, labelling_model=args.model,
      #   repair_candidates=result['funcs'])

      result_d['result']=result
      # result_d['repair_info']=repairs 
      structured_result.append(result_d)

      for sr in structured_result:
        sr['result_processed'] = postprocess(sr['result'], 0.3, args.model, soi_label, df)

    logger.warning("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    logger.warning("Final Results")
    for s in structured_result:
      logger.warning("------------------------------------------------------")
      logger.warning("Sentence of Interest")
      logger.warning(s['sentence'])
      logger.warning("Correct Label")
      logger.warning(s['correct_label'])
      logger.warning("Predicted Label")
      logger.warning(s['predicted_label'])
      logger.warning('before post processing Explanations')
      logger.warning(s['result'])
      logger.warning('after post processing Explanations')
      logger.warning(s['result_processed'])
  
  if(args.approach=='casuality'):
    for s in sentences_of_interest:
      soi_df = df[df['text']==s]
      soi_correct_label = list(soi_df['label'].values.astype(int))[0]
      soi = list(soi_df['text'].values.astype(str))[0].lower()
      casual_results = func_responsibility(funcs=LFs,
                          sentences=df,
                          expected_label=soi_correct_label, 
                          sentence_of_interest=soi,
                          model_type=args.model)
      





