from __future__ import annotations

import random

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer


def build_tfidf_matrix(texts: list[str], config):
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        max_df=config.tfidf_max_df,
        min_df=config.tfidf_min_df,
        max_features=config.tfidf_max_features,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())
    return X, terms


def reduce_and_cluster(X, config):
    svd = TruncatedSVD(n_components=config.svd_components, random_state=config.random_state)
    X_svd = svd.fit_transform(X)
    X_dense = Normalizer(copy=False).fit_transform(X_svd)

    km = MiniBatchKMeans(
        n_clusters=config.n_clusters,
        random_state=config.random_state,
        batch_size=1024,
        n_init=10,
        max_iter=120,
    )
    labels = km.fit_predict(X_dense)
    centers = km.cluster_centers_
    return X_dense, labels, centers


def top_terms_for_cluster(X, terms, labels, cluster_id: int, topn: int = 10):
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return [], []
    mean_weights = np.asarray(X[idx].mean(axis=0)).ravel()
    order = mean_weights.argsort()[::-1]
    top_tokens = [terms[i] for i in order if "_" not in terms[i]][:topn]
    top_bigrams = [terms[i] for i in order if "_" in terms[i]][:topn]
    return top_tokens, top_bigrams


def cluster_display_name(top_tokens: list[str], top_bigrams: list[str], stopwords: set[str]) -> str:
    filtered_tokens = [t for t in top_tokens if t not in stopwords and len(t) > 2]
    filtered_bigrams = []
    for bg in top_bigrams:
        parts = bg.split("_")
        if all(p not in stopwords and len(p) > 2 for p in parts):
            filtered_bigrams.append(bg)

    parts = filtered_tokens[:3]
    if len(parts) < 2:
        for bg in filtered_bigrams:
            human_bg = bg.replace("_", " ")
            if human_bg not in parts:
                parts.append(human_bg)
            if len(parts) >= 3:
                break
    if not parts:
        parts = ["без названия"]
    return " / ".join(parts[:3])


def lda_topics_for_cluster(cluster_texts: list[str], config):
    if len(cluster_texts) < config.lda_min_cluster_size:
        return []
    vect = CountVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        min_df=3,
        max_df=0.6,
        max_features=3000,
    )
    try:
        X_counts = vect.fit_transform(cluster_texts)
    except ValueError:
        return []
    if X_counts.shape[1] < 20:
        return []
    terms = np.array(vect.get_feature_names_out())
    lda = LatentDirichletAllocation(
        n_components=config.lda_topics_per_cluster,
        random_state=config.random_state,
        learning_method="batch",
    )
    lda.fit(X_counts)
    topics = []
    for comp in lda.components_:
        top_idx = comp.argsort()[::-1][: config.lda_words_per_topic]
        topics.append([terms[i] for i in top_idx])
    return topics


def sample_for_plot(X_dense, labels, centers, config):
    rng = random.Random(config.random_state)
    sampled_indices = []
    per_cluster = max(220, config.plot_max_points // config.n_clusters)

    for c in range(config.n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        sims = cosine_similarity(X_dense[idx], centers[c].reshape(1, -1)).ravel()
        sim_threshold = np.quantile(sims, config.plot_outlier_quantile)
        keep_mask = sims >= sim_threshold
        idx = idx[keep_mask]
        sims = sims[keep_mask]
        order = idx[np.argsort(sims)[::-1]]
        sampled_indices.extend(order[: min(len(order), per_cluster)].tolist())

    sampled_indices = list(dict.fromkeys(sampled_indices))
    if len(sampled_indices) < config.plot_max_points:
        remaining = list(set(range(len(labels))) - set(sampled_indices))
        rng.shuffle(remaining)
        sampled_indices.extend(remaining[: config.plot_max_points - len(sampled_indices)])
    return np.array(sampled_indices[: config.plot_max_points])


def reduce_for_plot(X_dense_subset, config):
    if config.use_pacmap:
        try:
            import pacmap
            reducer = pacmap.PaCMAP(
                n_components=2,
                n_neighbors=12,
                MN_ratio=0.5,
                FP_ratio=2.0,
                random_state=config.random_state,
            )
            return reducer.fit_transform(X_dense_subset), "PaCMAP"
        except Exception:
            pass

    import umap
    reducer = umap.UMAP(
        n_neighbors=24,
        min_dist=0.45,
        spread=1.25,
        metric="cosine",
        random_state=config.random_state,
    )
    return reducer.fit_transform(X_dense_subset), "UMAP"
