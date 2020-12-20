mkdir -p meta submission model notbooks

# Train
python3 ./src/preprocess/make_folds.py --input ./input/train-runmila.csv --output-train ./cache/train-runmila_2folds_seed123.pkl --n-fold 2 --seed 123



