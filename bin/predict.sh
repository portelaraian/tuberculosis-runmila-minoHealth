gpu=0
tta=5

predict_valid() {
    model=$1
    fold=$2
    ep=$3

    conf=./conf/${model}.py
    snapshot=./model/${model}/fold${fold}_ep${ep}.pt
    valid=./model/${model}/fold${fold}_ep${ep}_valid_tta${tta}.pkl

    python3 ./src/cnn/main.py valid ${conf} --snapshot ${snapshot} --output ${valid} --n-tta ${tta} --fold ${fold} --gpu ${gpu}
}

predict_test() {
    model=$1
    fold=$2
    ep=$3

    conf=./conf/${model}.py
    snapshot=./model/${model}/fold${fold}_ep${ep}.pt
    test=./model/${model}/fold${fold}_ep${ep}_test_tta${tta}.pkl

    python3 ./src/cnn/main.py test ${conf} --snapshot ${snapshot} --output ${test} --n-tta ${tta} --fold ${fold} --gpu ${gpu}
}

predict_test adamW-BCE/model_eff3_i1024_runmila_2fold_50ep 0 32
predict_test adamW-BCE/model_eff3_i1024_runmila_2fold_50ep 0 31
predict_test adamW-BCE/model_eff3_i1024_runmila_2fold_50ep 1 36
predict_test adamW-BCE/model_eff3_i1024_runmila_2fold_50ep 1 37

predict_test adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep 0 43
predict_test adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep 0 33
predict_test adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep 1 33
predict_test adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep 1 25

predict_test adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep 0 39
predict_test adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep 0 47
predict_test adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep 1 10
predict_test adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep 1 18

predict_test adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep 0 46
predict_test adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep 0 49
predict_test adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep 1 23
predict_test adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep 1 22

predict_valid adamW-BCE/model_eff3_i1024_runmila_2fold_50ep 0 32
predict_valid adamW-BCE/model_eff3_i1024_runmila_2fold_50ep 0 31
predict_valid adamW-BCE/model_eff3_i1024_runmila_2fold_50ep 1 36
predict_valid adamW-BCE/model_eff3_i1024_runmila_2fold_50ep 1 37

predict_valid adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep 0 43
predict_valid adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep 0 33
predict_valid adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep 1 33
predict_valid adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep 1 25

predict_valid adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep 0 39
predict_valid adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep 0 47
predict_valid adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep 1 10
predict_valid adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep 1 18

predict_valid adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep 0 46
predict_valid adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep 0 49
predict_valid adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep 1 23
predict_valid adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep 1 22
