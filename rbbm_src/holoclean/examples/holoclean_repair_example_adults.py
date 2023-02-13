import sys
sys.path.append('../')
import holoclean
from rbbm_src.holoclean.detect import NullDetector, ViolationDetector
from rbbm_src.holoclean.repair.featurize import *
import shutil
def main(file,initial_training=False):
    # 1. Setup a HoloClean session.
    hc = holoclean.HoloClean(
        db_name='holo',
        domain_thresh_1=0,
        domain_thresh_2=0,
        weak_label_thresh=0.99,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=10,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=1,
        batch_size=1,
        verbose=True,
        timeout=3*60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=True
    ).session

    # 2. Load training data and denial constraints.
    hc.load_data('adult', '/home/opc/chenjie/holoclean/testdata/Adult500.csv')
    if(initial_training):
        hc.load_dcs(f'{file}.txt')
        shutil.copyfile(f'{file}.txt', 
            f'{file}_test.txt')
    else:
        hc.load_dcs(f'{file}.txt')
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
    hc.evaluate(fpath='/home/opc/chenjie/holoclean/testdata/Adult500_clean.csv',
                tid_col='tid',
                attr_col='attribute',
                val_col='correct_val')

    hc.ds.engine.close_engine()


if __name__ == '__main__':
    main(True)