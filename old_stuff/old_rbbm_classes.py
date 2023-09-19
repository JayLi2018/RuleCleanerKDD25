from dataclasses import dataclass
from time import time
import pandas as pd
from rbbm_src.labelling_func_src.src.classes import lfunc_dec, lfunc
from typing import *
import psycopg2
from psycopg2._psycopg import connection
# ********************************************************************************
class ExecStats:
    TIMERS={}

    COUNTERS={}

    PARAMS ={}
    
    def __init__(self):
        self.time = {}
        self.numcalls = {}
        self.timer = {}
        self.counters = {}
        self.params = {}
        
        for t in self.TIMERS:
            self.time[t] = 0
            self.numcalls[t] = 0
            self.timer[t] = 0
        for c in self.COUNTERS:
            self.counters[c] = 0

        for p in self.PARAMS:
            self.params[p] = 0

    def formatStats(self):
        res='TIME MEASUREMENTS:\n{:*<40}\n'.format('')
        for t in sorted(self.time):
            res+="{:40}#calls: {:<20d}total-time: {:.8f}\n".format(t,self.numcalls[t],self.time[t])
        res+='\nCOUNTERS:\n{:*<40}\n'.format('')
        for c in sorted(self.counters):
            res+="number of {:20}{:<20d}\n".format(c,self.counters[c])
        res+='\nPARAMS:\n{:*<40}\n'.format('')
        for p in sorted(self.params):
            res+="{}: {}\n".format(p,self.params[p])
        return res

    def startTimer(self, name):
        self.timer[name] = time()
        self.numcalls[name]+=1
        #log.debug("numcalls: %d", self.numcalls[name])

    def resetTimer(self, name):
        self.time[name] = 0

    def stopTimer(self, name):
        measuredTime = time() - self.timer[name]
        #log.debug("timer %s ran for %f secs", name, measuredTime)
        self.time[name] += measuredTime

    def incr(self,name):
        self.counters[name] += 1

    def getCounter(self,name):
        return self.counters[name]

class StatsTracker(ExecStats):
    """
    Statistics gathered during mining
    """
    TIMERS = {'retrain'}
    COUNTERS = {'count_retrains', 'lookup_count'}
    PARAMS = {'featurize', 'setup_repair_model','fit_repair_model','get_inferred_values','query'}

class HoloCleanTracker(ExecStats):
    PARAMS = {'featurize', 'setup_repair_model','fit_repair_model','get_inferred_values'}

class HoloQueryTracker(ExecStats):
    PARAMS = {'query'}

@dataclass
class lf_input:
    """
    Input object which has all the arguments
    needed to execute the framework for labelling
    function setting
    """
    connection: connection
    contingency_size_threshold:int
    contingency_sample_times:int
    clustering_responsibility:bool
    sample_contingency:bool
    log_level:str
    user_provide:bool
    training_model_type:str
    word_threshold:int
    greedy:bool
    cardinality_thresh:int
    using_lattice:bool
    eval_mode:str
    # lattice: bool
    invoke_type: str # is it from 'terminal' or 'notebook'
    arg_str: str # only used if invoke_type='notebook'
    # lattice_dict:dict
    # lfs:List[lfunc]
    # sentences_df:pd.core.frame.DataFrame
    topk: int # topk value for number of lfs when doing responsibility generation
    random_number_for_complaint:int
    dataset_name: str
    stats: StatsTracker
    prune_only: bool
    return_complaint_and_results: bool


@dataclass
class dc_input:
    """
    Input object which has all the arguments
    needed to execute the framework for denial 
    constraint setting
    """
    connection: connection
    contingency_size_threshold:int
    contingency_sample_times:int
    clustering_responsibility:bool
    input_dc_dir:str 
    input_dc_file:str
    input_csv_dir:str
    input_csv_file:str
    ground_truth_dir:str 
    ground_truth_file:str
    sample_contingency:bool
    user_provide:bool
    random_number_for_complaint:int
    stats:StatsTracker
    prune_only: bool




@dataclass
class FixMonitor:
    """
    object that tracks the stats needed during the fix

    """
    counter:int=0 # count how many rules have been fixed 
    lambda_val: float=0.2
    # threshold predefined to retrain using the current fixes to see if it 
    # has already met the requirement
    rule_set_size: int=0 # total number of rules being used in the model
    overall_fixed_count: int=0 # overall total number of rules fixed so far


@dataclass
class RepairConfig:
    """
    object that contains the information 
    needed to do the repair
    """

    strategy:str # 'naive', 'information gain', 'optimal' 
    # rtype: str # 'dc' or 'lf'
    complaints:List[dict]
    monitor: FixMonitor
    acc_threshold: float 
    runtime:float
    deletion_factor:float
    # early stop threshold, i.e., if after fixing some rules the accuracy of 
    # the complaint set is above this threshold, we stop

    # tid:int=-1 # DC only
