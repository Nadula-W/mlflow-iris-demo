# MLflow Iris Demo

## Ubuntu setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run experiments

```bash
python src/train.py --n-estimators 20 --max-depth 2
python src/train.py --n-estimators 50 --max-depth 3
python src/train.py --n-estimators 100 --max-depth 5
```

## Open MLflow UI

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Then open http://127.0.0.1:5000
# mlflow-iris-demo
