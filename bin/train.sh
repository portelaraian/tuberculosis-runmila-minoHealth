gpu=0

train() {
    model=$1
    fold=$2

    conf=./conf/${model}.py
    python3 ./src/cnn/main.py train ${conf} --fold ${fold} --gpu ${gpu}
}

# Efficientnets 
## B3
train adamW-BCE/model_eff3_i1024_runmila_2fold_50ep 0
train adamW-BCE/model_eff3_i1024_runmila_2fold_50ep 1

# Squeeze-nets
## 101
train adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep 0
train adamW-BCE/model_seresnext101_32x4d_i768_runmila_2fold_50ep 1
## 50
train adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep 0
train adamW-BCE/model_seresnext50_32x4d_i1024_runmila_2fold_50ep 1

train adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep 0
train adamW-BCE/model_seresnext50_32x4d_i768_runmila_2fold_50ep 1

