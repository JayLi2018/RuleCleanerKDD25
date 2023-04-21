import sys
sys.path.append('../')
import rbbm_src.holoclean as holoclean
from rbbm_src.holoclean.detect import NullDetector, ViolationDetector
from rbbm_src.holoclean.repair.featurize import *
import shutil
from rbbm_src.classes import HoloCleanTracker, HoloQueryTracker

def main(table_name, csv_dir, csv_file, dc_dir, dc_file, gt_dir, gt_file, initial_training=False, pruned=False):
    # 1. Setup a HoloClean session.

    h=holoclean.HoloClean(
        db_name='holo',
        domain_thresh_1=0,
        domain_thresh_2=0,
        weak_label_thresh=0.99,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=4,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=1,
        batch_size=1,
        verbose=True,
        timeout=3*60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=True
    )

    hc =h.session
    hc.set_timer(HoloCleanTracker())
    hc.ds.engine.set_timer(HoloQueryTracker())

    # 2. Load training data and denial constraints.
    if(pruned):
        hc.load_data(table_name, csv_dir+csv_file, entity_col='_tid_')
    else:
        hc.load_data(table_name, csv_dir+csv_file)
        
    if(initial_training):
        hc.load_dcs(f'{dc_dir+dc_file}')
        shutil.copyfile(f'{dc_dir+dc_file}', 
            f'{dc_dir}subset_{dc_file}')
    else:
        hc.load_dcs(f'{dc_dir+dc_file}')
    hc.ds.set_constraints(hc.get_dcs())

    # 3. Detect erroneous cells using these two detectors.
    detectors = [NullDetector(), ViolationDetector()]
    hc.detect_errors(detectors)

    # 4. Repair errors utilizing the defined features.
    hc.setup_domain()
    featurizers = [
        InitAttrFeaturizer(),
        OccurAttrFeaturizer(),
        FreqFeaturizer(),
        ConstraintFeaturizer(),
    ]

    hc.repair_errors(featurizers)

    # 5. Evaluate the correctness of the results.
    hc.evaluate(fpath=f'{gt_dir}{gt_file}',
                tid_col='tid',
                attr_col='attribute',
                val_col='correct_val')
    hc.ds.engine.close_engine()

    return hc.timer, hc.ds.engine.timer
