import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    IADD_PATH, TRAIN_PATH, TEST_PATH,
    REGIONS_TO_KEEP, SAMPLES_PER_CLASS,
    RANDOM_SEED, TEST_SIZE
)



def load_iadd(path):
    print("Loading IADD.json...")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)[['Sentence', 'Region']]
    print(f"  Loaded {len(df):,} records")
    print(f"  Region counts:\n{df['Region'].value_counts()}\n")
    return df


#Filter 
def filter_regions(df, regions):
    print(f"Keeping regions: {regions}")
    df = df[df['Region'].isin(regions)].copy()
    df = df.dropna(subset=['Sentence'])
    df = df[df['Sentence'].str.strip() != '']
    print(f"  After filtering: {len(df):,} records")
    print(f"  Region counts:\n{df['Region'].value_counts()}\n")
    return df


# Normalize
def normalize_arabic(text):
    # Normalize Arabic character variants
    text = re.sub('[إأآا]', 'ا', text)
    text = re.sub('ى', 'ي', text)
    text = re.sub('ؤ', 'و', text)
    text = re.sub('ئ', 'ي', text)
    text = re.sub('ة', 'ه', text)

    # Remove non-Arabic characters
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def apply_normalization(df):
    print("Applying Arabic normalization...")
    df = df.copy()
    df['Sentence'] = df['Sentence'].apply(normalize_arabic)

    # Drop rows that became empty after normalization
    df = df[df['Sentence'].str.strip() != '']
    print(f"  After normalization: {len(df):,} records\n")
    return df

#downsample 
def downsample(df, samples_per_class, seed):
    print(f"Downsampling to {samples_per_class:,} samples per class...")
    
    sampled_groups = []
    for region, group in df.groupby('Region'):
        sampled = group.sample(
            min(len(group), samples_per_class),
            random_state=seed
        )
        sampled_groups.append(sampled)
    
    df_balanced = pd.concat(sampled_groups).reset_index(drop=True)
    print(f"  After downsampling: {len(df_balanced):,} records")
    print(f"  Region counts:\n{df_balanced['Region'].value_counts()}\n")
    return df_balanced

# Split
def split_and_save(df, test_size, seed, train_path, test_path):
    print(f"Splitting into {int((1-test_size)*100)}/{int(test_size*100)} train/test...")
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df['Region']
    )
    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    print(f"  Train size: {len(train):,}")
    print(f"  Train region counts:\n{train['Region'].value_counts()}")
    print(f"  Test size: {len(test):,}")
    print(f"  Test region counts:\n{test['Region'].value_counts()}\n")

    train.to_csv(train_path, index=False, encoding='utf-8-sig')
    test.to_csv(test_path,  index=False, encoding='utf-8-sig')
    print(f"  Saved train.csv → {train_path}")
    print(f"  Saved test.csv  → {test_path}\n")
    return train, test

def run():
    df = load_iadd(IADD_PATH)
    df = filter_regions(df, REGIONS_TO_KEEP)
    df = apply_normalization(df)
    df = downsample(df, SAMPLES_PER_CLASS, RANDOM_SEED)
    train, test = split_and_save(df, TEST_SIZE, RANDOM_SEED, TRAIN_PATH, TEST_PATH)
    print("Preprocessing complete.")
    return train, test


if __name__ == '__main__':
    run()