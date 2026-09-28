import pandas as pd
import random
import re
from nltk.stem.snowball import SnowballStemmer

def get_test_queries(
        df_train: pd.DataFrame,
        n: int = 5000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    search_cols = [
        'search_query', 
        'search_location_id', 
        'search_is_delivery_search', 
        'search_infm_params_text', 
        'search_category'
    ]

    df_train['search_infm_params_text'] = df_train['search_infm_params_text'].fillna('')
    df_train['search_category'] = df_train['search_category'].fillna('')
    df_train['search_location_id'] = df_train['search_location_id'].fillna(0)

    grouped_train = df_train.groupby(search_cols)['item_id'].apply(list).reset_index()
    grouped_train['val_query_id'] = [f'val_query_{i}' for i in range(len(grouped_train))]

    val_set = grouped_train.sample(n=n, random_state=42).copy()
    val_ground_truth = val_set.set_index('val_query_id')['item_id'].to_dict()
    val_queries_df = val_set[['val_query_id'] + search_cols].copy()

    return val_ground_truth, val_queries_df

def parse_search_params(params_text: str) -> dict:
    if not isinstance(params_text, str) or not params_text.strip():
        return {'min_rating': 0.0, 'clean_text': ''}

    parsed = {'min_rating': 0.0, 'clean_text': ''}

    rating_match = re.search(
        r'Рейтинг пользователя (\d+(?:[.,]\d+)?) звезды и выше',
        params_text, flags=re.IGNORECASE
    )
    if rating_match:
        rating_str = rating_match.group(1).replace(',', '.')
        parsed['min_rating'] = float(rating_str)

    clean_text = params_text
    clean_text = re.sub(
        r'Рейтинг пользователя (\d+(?:[.,]\d+)?) звезды и выше',
        '', clean_text, flags=re.IGNORECASE
    )

    prefixes = [
        'Вид услуги',
        'Тип услуги',
        'Где вы оказываете услуги',
        'Ваши клиенты',
        'Кто оказывает услуги'
    ]

    for prefix in prefixes:
        clean_text = re.sub(prefix, '', clean_text, flags=re.IGNORECASE)

    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
    clean_text = ' '.join(clean_text.split())

    parsed['clean_text'] = clean_text
    return parsed

def tokenize(text: str, stemmer=SnowballStemmer('russian')) -> list[str]:
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [stemmer.stem(w) for w in text.split()]