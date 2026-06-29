import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

IADD_PATH  = os.path.join(DATA_DIR, 'IADD.json')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH  = os.path.join(DATA_DIR, 'test.csv')

REGIONS_TO_KEEP   = ['EGY', 'GLF', 'LEV', 'MGH']  # drop Iraqi dialect
SAMPLES_PER_CLASS = 4837                            # capped at Egyptian count
RANDOM_SEED       = 42
TEST_SIZE         = 0.2                             # 80/20 split

TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 2)                         
TFIDF_MIN_DF       = 2                             

# MarBERT settings (used in Colab notebook) 
MARBERT_MODEL      = 'UBC-NLP/MARBERTv2'
MARBERT_MAX_LENGTH = 128
MARBERT_EPOCHS     = 4
MARBERT_LR         = 2e-5
MARBERT_BATCH_SIZE = 16
MARBERT_WARMUP     = 0.1                            