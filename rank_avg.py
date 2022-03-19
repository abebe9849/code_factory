

import pandas as pd
import numpy as np
import os,glob

from sklearn.metrics import roc_auc_score

def main():
    PATH = "/home/u094724e/aimed2022/src/ensemble/try1"#oof_{exp_num}.csvとsub_{exp_num}.csvをいれておく。
    FILES = os.listdir(PATH)
    oof_col = "pred_0"
    suc_col="Predict"
    OOF = np.sort([f for f in FILES if "oof" in f])
    OOF_CSV = [pd.read_csv(os.path.join(PATH,k)).rename(columns={oof_col:f"pred_{i}"})[f"pred_{i}"] for i,k in enumerate(OOF)]

    folds_df = pd.read_csv(f"{PATH}/oof.csv")

    SUB = np.sort([f for f in FILES if "sub" in f])
    SUB_CSV = [pd.read_csv(os.path.join(PATH,k)).rename(columns={suc_col:f"pred_{i}"})[f"pred_{i}"] for i,k in enumerate(SUB)]

    oof_concat = pd.concat(OOF_CSV,axis=1)
    sub_concat = pd.concat(SUB_CSV,axis=1)

    folds = pd.read_csv(os.path.join(PATH,OOF[0]))[["file_path","target"]]
    folds = pd.concat([folds,oof_concat],axis=1)
    test = pd.read_csv(os.path.join(PATH,SUB[0]))
    sub = pd.read_csv(os.path.join(PATH,SUB[0]))
    test = pd.concat([test,sub_concat],axis=1)
    print(oof_concat,sub_concat)
    for col in oof_concat.columns:
        folds[col + '_rank'] = folds[col].rank()
        test[col + '_rank'] = test[col].rank()

    folds['rank_sum'] = np.sum(folds[col] for col in folds.columns if '_rank' in col)
    folds['pred_ra'] = folds['rank_sum']/(len(oof_concat.columns) *folds.shape[0])
    test['rank_sum'] = np.sum(test[col] for col in test.columns if '_rank' in col)
    test['pred_ra'] = test['rank_sum']/(len(oof_concat.columns)*test.shape[0])

    score = roc_auc_score(folds["target"], folds['pred_ra'])
    print("rank_avg",score)

    sub[suc_col]=test['pred_ra']

    sub.to_csv("sub_rank_avg.csv",index=False)


if __name__ == "__main__":
    main()



    


