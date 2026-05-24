import pandas as pd
import numpy as np
import os

# Назви колонок датасету NASA C-MAPSS
COLUMN_NAMES = [
    'engine_id', 'cycle',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    's1',  's2',  's3',  's4',  's5',
    's6',  's7',  's8',  's9',  's10',
    's11', 's12', 's13', 's14', 's15',
    's16', 's17', 's18', 's19', 's20', 's21'
]

def load_train(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не знайдено: {filepath}")
    df = pd.read_csv(filepath, sep=r'\s+', engine='python', header=None, names=COLUMN_NAMES)
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycles, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df.drop(columns=['max_cycle'], inplace=True)
    print(f"Навчальна вибірка завантажена: {df.shape[0]} записів, {df['engine_id'].nunique()} двигунів")
    return df

def load_test(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не знайдено: {filepath}")
    df = pd.read_csv(filepath, sep=r'\s+', engine='python', header=None, names=COLUMN_NAMES)
    print(f"Тестова вибірка завантажена: {df.shape[0]} записів, {df['engine_id'].nunique()} двигунів")
    return df

def load_rul(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не знайдено: {filepath}")
    rul = pd.read_csv(filepath, header=None, names=['true_RUL'])
    rul['engine_id'] = range(1, len(rul) + 1)
    rul = rul[['engine_id', 'true_RUL']]
    print(f"Еталонні RUL завантажені: {len(rul)} двигунів")
    return rul

def load_all(data_dir='data/raw'):
    return {
        'train': load_train(os.path.join(data_dir, 'train_FD001.txt')),
        'test':  load_test(os.path.join(data_dir, 'test_FD001.txt')),
        'rul':   load_rul(os.path.join(data_dir, 'RUL_FD001.txt')),
    }

if __name__ == '__main__':
    data = load_all('data/raw')
    print("\n── Навчальна вибірка ──")
    print(data['train'].head())
    print(f"Колонки: {list(data['train'].columns)}")
    print(f"RUL min: {data['train']['RUL'].min()}, max: {data['train']['RUL'].max()}")
    print("\n── Тестова вибірка ──")
    print(data['test'].head())
    print("\n── Еталонні RUL ──")
    print(data['rul'].head())
