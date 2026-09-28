import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from rank_bm25 import BM25Okapi
from nltk.stem.snowball import SnowballStemmer
from src.data import parse_search_params, tokenize
from sentence_transformers import SentenceTransformer
import os
import pickle

class CandidateGenerationPipeline:
    """
    Пайплайн кандидатогенерации.
    Объединяет лексический поиск (BM25), семантический поиск (Bi-Encoder) 
    и бизнес-эвристики (рейтинг, количество отзывов, локация и пр.) для 
    отбора топ-N релевантных объявлений.
    """
    def __init__(
            self,
            df_items: pd.DataFrame,
            sentence_transformer_model_name: str = 'cointegrated/rubert-tiny2',
            params_description_max_length: int = 300,
            stemmer = SnowballStemmer('russian'),
            hf_token: str | None = None,
            device: str = 'cuda',
            cache_dir: str = 'cache',
            batch_size=128
    ):
        self.stemmer = stemmer
        self.sentence_transformer_model_name = sentence_transformer_model_name
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self._init_features(df_items)

        self._init_bm25(df_items, params_description_max_length)

        self.encoder = SentenceTransformer(sentence_transformer_model_name, token=hf_token, device=device)
        self._init_embeddings(df_items, params_description_max_length)

    def _init_features(self, df_items):
        """
        Извлекает табличные признаки объявлений (категории, локации, рейтинги) 
        в Numpy-массивы для быстрой векторизованной фильтрации.
        """
        self.item_ids = df_items['item_id'].values
        self.item_cat_ids = df_items['item_category_id'].values
        self.item_loc_ids = df_items['item_location_id'].values
        self.item_ratings = df_items['item_rating'].fillna(0.0).values
        self.item_reviews = df_items['item_rating_reviews_count'].fillna(0.0).values
        self.item_prices = df_items['item_price'].fillna(0.0).apply(lambda x: x if x > 0 else 0.0).values
        self.item_is_phone_hidden = df_items['item_is_phone_hidden'].fillna(0.0).values
        self.item_is_message_forbidden = df_items['item_is_message_forbidden'].fillna(0.0).values

    def _init_bm25(self, df_items: pd.DataFrame, max_len: int):
        """
        Строит или загружает из кэша лексические индексы BM25Okapi 
        отдельно для заголовков и описаний объявлений.
        """
        title_path = os.path.join(self.cache_dir, 'bm25_title.pkl')
        desc_path = os.path.join(self.cache_dir, 'bm25_desc.pkl')

        if os.path.exists(title_path) and os.path.exists(desc_path):
            with open(title_path, 'rb') as f:
                self.bm25_title = pickle.load(f)
            with open(desc_path, 'rb') as f:
                self.bm25_desc = pickle.load(f)
        else:
            titles = df_items['item_title_raw'].fillna("").tolist()
            descs = [t[:max_len] for t in (df_items['item_infm_params_text'].fillna("") 
                                           + " " + df_items['item_description_raw'].fillna("")).tolist()]
            
            self.bm25_title = BM25Okapi([tokenize(t, stemmer=self.stemmer) for t in tqdm(titles, desc="Titles")])
            self.bm25_desc = BM25Okapi([tokenize(d, stemmer=self.stemmer) for d in tqdm(descs, desc="Descriptions")])
            
            with open(title_path, 'wb') as f:
                pickle.dump(self.bm25_title, f)
            with open(desc_path, 'wb') as f:
                pickle.dump(self.bm25_desc, f)

    def _init_embeddings(self, df_items: pd.DataFrame, max_len: int):
        """
        Векторизует тексты объявлений с помощью нейросети (SentenceTransformer) 
        или мгновенно загружает готовую матрицу эмбеддингов из кэша (.npy).
        """
        emb_path = os.path.join(self.cache_dir, 'item_embeddings.npy')
        if os.path.exists(emb_path):
            self.item_embeddings = np.load(emb_path, mmap_mode='r') 
        else:
            titles = df_items['item_title_raw'].fillna("").tolist()
            descs = [t[:max_len] for t in (df_items['item_infm_params_text'].fillna("") 
                                           + " " + df_items['item_description_raw'].fillna("")).tolist()]
            
            start = "passage: " if "intfloat/multilingual-e5" in self.sentence_transformer_model_name else ""
            nn_texts = [start + str(t) + " " + str(d) for t, d in zip(titles, descs)]
            
            self.item_embeddings = self.encoder.encode(nn_texts, show_progress_bar=True, normalize_embeddings=True, batch_size=self.batch_size)
            np.save(emb_path, self.item_embeddings)

    def _prepare_queries_batch(self, df_queries: pd.DataFrame):
        """Парсит и векторизует все запросы одним батчем"""
        processed_queries, min_ratings = [], []
        start_prefix = "query: " if "intfloat/multilingual-e5" in self.sentence_transformer_model_name else ""
        
        for _, row in df_queries.iterrows():
            parsed = parse_search_params(row.get('search_infm_params_text', ''))
            min_ratings.append(parsed['min_rating'])
            processed_queries.append(start_prefix + str(row.get('search_query', '')) 
                                     + ' ' + parsed['clean_text'])

        embeddings = self.encoder.encode(
            processed_queries,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=self.batch_size
        )
        return processed_queries, min_ratings, embeddings

    def _score_candidates(
            self,
            row: pd.Series,
            query_tokens: list[str],
            query_emb,
            min_rating,
            title_weight,
            bm25_weight,
            norm: str = 'min_max'
        ):
        """
        Вычисляет базовый текстовые баллы кандидатов. 
        Применяет жесткие фильтры и объединяет баллы 
        BM25 и нейросети методами Min-Max нормализации или RRF.
        """        
        target_cat_id = row.get('search_category', -1)
        target_cat_id = -1 if pd.isna(target_cat_id) else target_cat_id
        
        valid_mask = (self.item_cat_ids == target_cat_id)
        if min_rating > 0: valid_mask &= (self.item_ratings >= min_rating)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0 and min_rating > 0:
            valid_mask = (self.item_cat_ids == target_cat_id)
            valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(len(self.item_ids))

        scores_title = np.array(self.bm25_title.get_scores(query_tokens))[valid_indices]
        scores_desc = np.array(self.bm25_desc.get_scores(query_tokens))[valid_indices]
        bm25_scores = title_weight * scores_title + (1 - title_weight) * scores_desc
        
        nn_scores = np.dot(self.item_embeddings[valid_indices], query_emb).flatten()

        if norm.lower() == 'min_max':
            bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)
            nn_norm = (nn_scores - np.min(nn_scores)) / (np.max(nn_scores) - np.min(nn_scores) + 1e-9)
            final_scores = bm25_weight * bm25_norm + (1 - bm25_weight) * nn_norm
        elif norm.lower() == 'rrf':
            rank_bm25 = len(bm25_scores) - np.argsort(np.argsort(bm25_scores))
            rank_nn = len(nn_scores) - np.argsort(np.argsort(nn_scores))

            k_rrf = 60.0
            final_scores = bm25_weight * (1.0 / (k_rrf + rank_bm25)) + (1 - bm25_weight) * (1.0 / (k_rrf + rank_nn))
        else:
            final_scores = bm25_weight * bm25_scores + (1 - bm25_weight) * nn_scores

        return valid_indices, bm25_scores, nn_scores, final_scores

    def predict_batch(
            self,
            df_queries: pd.DataFrame,
            top_n: int = 50,
            query_id_col='val_query_id', # query_id для теста
            title_weight: float = 0.5, 
            bm25_weight: float = 0.6,
            location_weight: float = 2.0,
            hidden_phone_weight: float = 0.8,
            forbidden_msg_weight: float = 0.8,
            review_count_weight: float = 0.15,
            norm: str = 'min_max'
        ):
        """
        Прогоняет запросы через пайплайн, применяет бизнес-эвристики 
        и возвращает словарь с топ-N item_id для каждого запроса.
        """
        processed_queries, min_ratings, queries_embeddings = self._prepare_queries_batch(df_queries)
        predictions = {}

        for i, (idx, row) in enumerate(tqdm(df_queries.iterrows(), total=len(df_queries), desc='Predicting')):
            query_id = row[query_id_col]
            query_tokens = tokenize(processed_queries[i], stemmer=self.stemmer)
            
            if not query_tokens:
                predictions[query_id] = self.item_ids[:top_n].tolist()
                continue

            valid_indices, bm25_scores, nn_scores, final_scores = self._score_candidates(
                row,
                query_tokens,
                queries_embeddings[i],
                min_ratings[i],
                title_weight,
                bm25_weight,
                norm=norm
            )

            c_ratings = self.item_ratings[valid_indices]
            rating_factor = np.ones(len(valid_indices), dtype=float)
            has_rating = (c_ratings > 0)
            rating_factor[has_rating] = c_ratings[has_rating] / 5.0
            
            final_scores *= rating_factor

            c_reviews = self.item_reviews[valid_indices]

            review_multiplier = 1.0 + review_count_weight * (c_reviews / (c_reviews + 25.0))
            
            final_scores *= review_multiplier

            target_loc_id = row.get('search_location_id', -1)
            if not pd.isna(target_loc_id) and target_loc_id != -1:
                loc_mask = (self.item_loc_ids[valid_indices] == target_loc_id)
                final_scores[loc_mask & (final_scores > 0)] *= location_weight

            hidden_phone = self.item_is_phone_hidden[valid_indices]
            forbidden_msg = self.item_is_message_forbidden[valid_indices]
            
            contact_multiplier = np.ones(len(valid_indices), dtype=float)
            
            contact_multiplier[hidden_phone == 1.0] *= hidden_phone_weight
            contact_multiplier[forbidden_msg == 1.0] *= forbidden_msg_weight

            final_scores *= contact_multiplier

            top_k = min(top_n, len(final_scores))
            top_local_indices = np.argpartition(final_scores, -top_k)[-top_k:]
            top_local_indices = top_local_indices[np.argsort(final_scores[top_local_indices])[::-1]]

            final_top_indices = valid_indices[top_local_indices]
            predictions[query_id] = self.item_ids[final_top_indices].tolist()

        return predictions