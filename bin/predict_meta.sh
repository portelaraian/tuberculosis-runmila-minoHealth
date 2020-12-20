predict_meta() {
    oof=$1
    test=$2
    name=$3
    python3 -u ./src/meta/trainer.py --inputs-test "${test}" --inputs-oof "${oof}" --output-name ${name} |tee ./meta/${name}.log
}

oofmodel_eff3_i1024_runmila_2fold_50ep="[\
    ['./model/adamW-BCE/model_eff3_i1024_runmila_2fold_50ep/fold0_ep31_valid_tta5.pkl', './model/adamW-BCE/model_eff3_i1024_runmila_2fold_50ep/fold0_ep32_valid_tta5.pkl'],\
    ['./model/adamW-BCE/model_eff3_i1024_runmila_2fold_50ep/fold1_ep36_valid_tta5.pkl', './model/adamW-BCE/model_eff3_i1024_runmila_2fold_50ep/fold1_ep37_valid_tta5.pkl'],\
]"
testmodel_eff3_i1024_runmila_2fold_50ep="[\
    ['./model/adamW-BCE/model_eff3_i1024_runmila_2fold_50ep/fold0_ep31_test_tta5.pkl', './model/adamW-BCE/model_eff3_i1024_runmila_2fold_50ep/fold0_ep32_test_tta5.pkl'],\
    ['./model/adamW-BCE/model_eff3_i1024_runmila_2fold_50ep/fold1_ep36_test_tta5.pkl', './model/adamW-BCE/model_eff3_i1024_runmila_2fold_50ep/fold1_ep37_test_tta5.pkl'],\
]"

oofmodel_seresnext101_32x4d_i768_runmila_2fold_50ep="[\
    ['./model/adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep/fold0_ep43_valid_tta5.pkl', './model/adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep/fold0_ep33_valid_tta5.pkl'],\
    ['./model/adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep/fold1_ep33_valid_tta5.pkl', './model/adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep/fold1_ep25_valid_tta5.pkl'],\
]"
testmodel_seresnext101_32x4d_i768_runmila_2fold_50ep="[\
    ['./model/adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep/fold0_ep43_test_tta5.pkl', './model/adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep/fold0_ep33_test_tta5.pkl'],\
    ['./model/adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep/fold1_ep33_test_tta5.pkl', './model/adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep/fold1_ep25_test_tta5.pkl'],\
]"

oofmodel_seresnext50_32x4d_i1024_runmila_2fold_50ep="[\
    ['./model/adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep/fold0_ep39_valid_tta5.pkl', './model/adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep/fold0_ep47_valid_tta5.pkl'],\
    ['./model/adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep/fold1_ep10_valid_tta5.pkl', './model/adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep/fold1_ep18_valid_tta5.pkl'],\
]"
testmodel_seresnext50_32x4d_i1024_runmila_2fold_50ep="[\
    ['./model/adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep/fold0_ep39_test_tta5.pkl', './model/adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep/fold0_ep47_test_tta5.pkl'],\
    ['./model/adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep/fold1_ep10_test_tta5.pkl', './model/adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep/fold1_ep18_test_tta5.pkl'],\
]"

oofmodel_seresnext50_32x4d_i768_runmila_2fold_50ep="[\
    ['./model/adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep/fold0_ep46_valid_tta5.pkl', './model/adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep/fold0_ep49_valid_tta5.pkl'],\
    ['./model/adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep/fold1_ep22_valid_tta5.pkl', './model/adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep/fold1_ep23_valid_tta5.pkl'],\
]"
testmodel_seresnext50_32x4d_i768_runmila_2fold_50ep="[\
    ['./model/adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep/fold0_ep46_test_tta5.pkl', './model/adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep/fold0_ep49_test_tta5.pkl'],\
    ['./model/adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep/fold1_ep22_test_tta5.pkl', './model/adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep/fold1_ep23_test_tta5.pkl'],\
]"




oof1="[${oofmodel_eff3_i1024_runmila_2fold_50ep}, ${oofmodel_seresnext101_32x4d_i768_runmila_2fold_50ep}, ${oofmodel_seresnext50_32x4d_i1024_runmila_2fold_50ep}, ${oofmodel_seresnext50_32x4d_i768_runmila_2fold_50ep}]"

test1="[${testmodel_eff3_i1024_runmila_2fold_50ep}, ${testmodel_seresnext101_32x4d_i768_runmila_2fold_50ep}, ${testmodel_seresnext50_32x4d_i1024_runmila_2fold_50ep}, ${testmodel_seresnext50_32x4d_i768_runmila_2fold_50ep}]"

predict_meta "${oof1}" "${test1}" meta_eff3_seresnext50_101_i768_1024_tta5