# Arabic Dialect Identification: Classical Models vs. MarBERT

A comparative study of Multinomial Naïve Bayes, Logistic Regression,
Linear SVM, and MarBERT for Arabic dialect identification across four
regional Arabic dialect classes (Egyptian, Gulf, Levantine, Maghrebi)
using the IADD dataset.

## Dataset
IADD: An integrated Arabic dialect identification dataset  
Zahir, J. (2022). Data in Brief, 40, 107777.  
https://doi.org/10.1016/j.dib.2021.107777  
https://github.com/JihadZa/IADD

Download IADD.json and place it in the data/ folder before running.

## Project Structure

Arabic-Dialect-Identification/
├── data/                          ← place IADD.json here (not tracked)
├── src/
│   ├── preprocess.py              ← data loading, cleaning, splitting
│   ├── train_classical.py         ← NB, LR, SVM training + evaluation
│   └── evaluate.py                ← metrics, confusion matrices, figures
├── notebooks/
│   └── marbert_finetuning.ipynb   ← MarBERT fine-tuning (Google Colab)
├── results/
│   ├── figures/                   ← confusion matrices, charts
│   └── metrics/                   ← results CSVs
├── config.py                      ← all hyperparameters in one place
├── main.py                        ← runs full classical pipeline
└── requirements.txt

## Setup

pip install -r requirements.txt
python main.py

## Authors
- Hana Wahban (F2300076) — American University of Bahrain
- Hamza Alkhaldi (F2300066) — American University of Bahrain

## Reference
Zahir, J. (2022). IADD: An integrated Arabic dialect identification
dataset. Data in Brief, 40, 107777.
https://doi.org/10.1016/j.dib.2021.107777