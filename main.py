from src.preprocess import run as preprocess
from src.train_classical import run as train_classical
from src.evaluate import run as evaluate

if __name__ == '__main__':
    print("=" * 60)
    print("ARABIC DIALECT IDENTIFICATION — CLASSICAL PIPELINE")
    print("=" * 60)

    print("\n[1/3] Preprocessing...")
    train, test = preprocess()

    print("\n[2/3] Training classical models...")
    results = train_classical(train, test)

    print("\n[3/3] Generating evaluation figures...")
    evaluate(results)

    print("\nDone. Results saved to results/")