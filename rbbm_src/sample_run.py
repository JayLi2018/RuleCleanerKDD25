import sys
sys.path.append('../')
import holoclean
from rbbm_src.holoclean.detect import NullDetector, ViolationDetector
from rbbm_src.holoclean.repair.featurize import *
import shutil
def main():
    # 1. Setup a HoloClean session.
    hc = holoclean.HoloClean(
        db_name='holo',
        domain_thresh_1=0.3,
        domain_thresh_2=0.3,
        weak_label_thresh=0.99,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=4,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=1,
        batch_size=32,
        verbose=True,
        timeout=3*60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=True
    ).session

    # 2. Load training data and denial constraints.
    hc.load_data('adult', '/home/opc/chenjie/labelling_explanation/holoclean/testdata/adult500.csv')
    hc.load_dcs(f'/home/opc/chenjie/labelling_explanation/holoclean/testdata/dc_finder_adult_rules.txt')
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
    hc.evaluate(fpath='/home/opc/chenjie/labelling_explanation/holoclean/testdata/adult500_clean.csv',
                tid_col='tid',
                attr_col='attribute',
                val_col='correct_val')

    hc.ds.engine.close_engine()


if __name__ == '__main__':
    main()