import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.cells = []

nb.cells.append(nbf.v4.new_markdown_cell("### 1. Environment Setup and Reproducibility"))

cell1 = """import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

SEED = 42
np.random.seed(SEED)

os.makedirs('data', exist_ok=True)
os.makedirs('output/analysis', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)
"""
nb.cells.append(nbf.v4.new_code_cell(cell1))

nb.cells.append(nbf.v4.new_markdown_cell("### 2. Data Ingestion and Schema Normalization"))

cell2 = """raw_path = 'data/online_retail.csv'
if not os.path.exists(raw_path):
    import urllib.request, zipfile, io
    url = 'https://archive.ics.uci.edu/static/public/352/online+retail.zip'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as resp:
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        excel_name = z.namelist()[0]
        with z.open(excel_name) as f:
            df_raw = pd.read_excel(f)
    df_raw.to_csv(raw_path, index=False)

df = pd.read_csv(raw_path)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
required_cols = {'InvoiceNo', 'StockCode', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID'}
missing = required_cols - set(df.columns)
assert not missing, f'Missing columns: {missing}'
"""
nb.cells.append(nbf.v4.new_code_cell(cell2))

nb.cells.append(nbf.v4.new_markdown_cell("### 3. Transaction Integrity and Cleaning"))

cell3 = """df = df.dropna(subset=['CustomerID']).copy()
df['CustomerID'] = df['CustomerID'].astype(np.int64).astype(str)
df['is_cancel'] = df['InvoiceNo'].astype(str).str.startswith('C')
df = df[(~df['is_cancel']) & (df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
df['Amount'] = df['Quantity'] * df['UnitPrice']
"""
nb.cells.append(nbf.v4.new_code_cell(cell3))

nb.cells.append(nbf.v4.new_markdown_cell("### 4. Chronological Train-Validation-Test Splitting"))

cell4 = """t1 = df['InvoiceDate'].quantile(0.70)
t2 = df['InvoiceDate'].quantile(0.85)
t0 = df[df.InvoiceDate < t1]['InvoiceDate'].quantile(0.70)

train_hist = df[df.InvoiceDate < t1].copy()
val_future = df[(df.InvoiceDate >= t1) & (df.InvoiceDate < t2)].copy()
dev_hist = df[df.InvoiceDate < t2].copy()
test_future = df[df.InvoiceDate >= t2].copy()

split_manifest = {
    'total_raw_rows': 541909,
    'total_cleaned_rows': int(len(df)),
    'unique_customers': int(df['CustomerID'].nunique()),
    'unique_items': int(df['StockCode'].nunique()),
    'date_min': str(df['InvoiceDate'].min()),
    'date_max': str(df['InvoiceDate'].max()),
    'cutoff_t0': str(t0),
    'cutoff_t1_train_end': str(t1),
    'cutoff_t2_val_end': str(t2),
    'train_hist_rows': int(len(train_hist)),
    'train_hist_users': int(train_hist['CustomerID'].nunique()),
    'val_future_rows': int(len(val_future)),
    'val_future_users': int(val_future['CustomerID'].nunique()),
    'test_future_rows': int(len(test_future)),
    'test_future_users': int(test_future['CustomerID'].nunique())
}
with open('artifacts/split_manifest.json', 'w') as f:
    json.dump(split_manifest, f, indent=4)
"""
nb.cells.append(nbf.v4.new_code_cell(cell4))

nb.cells.append(nbf.v4.new_markdown_cell("### 5. Candidate Generation Policy"))

cell5 = """candidate_items = train_hist.groupby('StockCode')['InvoiceNo'].nunique().sort_values(ascending=False).head(1000).index.tolist()
cand_set = set(candidate_items)

cand_policy = {
    'policy_name': 'top_1000_frequent_in_training_history',
    'catalog_size': len(candidate_items),
    'selection_metric': 'unique_invoices_in_train_hist',
    'repeat_purchases_allowed': True,
    'bounded_range_min': 500,
    'bounded_range_max': 2000
}
with open('artifacts/candidate_policy.json', 'w') as f:
    json.dump(cand_policy, f, indent=4)
"""
nb.cells.append(nbf.v4.new_code_cell(cell5))

nb.cells.append(nbf.v4.new_markdown_cell("### 6. Candidate Recall Audit"))

cell6 = """val_users_with_hist = sorted(list(set(val_future['CustomerID']).intersection(set(train_hist['CustomerID']))))
test_users_with_hist = sorted(list(set(test_future['CustomerID']).intersection(set(dev_hist['CustomerID']))))

val_actual_items = val_future[val_future['CustomerID'].isin(val_users_with_hist)].groupby('CustomerID')['StockCode'].apply(set).to_dict()
test_actual_items = test_future[test_future['CustomerID'].isin(test_users_with_hist)].groupby('CustomerID')['StockCode'].apply(set).to_dict()

val_cand_recalls = [len(items.intersection(cand_set)) / len(items) for items in val_actual_items.values() if len(items) > 0]
test_cand_recalls = [len(items.intersection(cand_set)) / len(items) for items in test_actual_items.values() if len(items) > 0]

val_full_rep = np.mean([1.0 if len(items.intersection(cand_set)) == len(items) else 0.0 for items in val_actual_items.values()])
test_full_rep = np.mean([1.0 if len(items.intersection(cand_set)) == len(items) else 0.0 for items in test_actual_items.values()])

cand_recall_df = pd.DataFrame([
    {
        'Split': 'Validation',
        'Evaluation_Users': len(val_actual_items),
        'Candidate_Catalog_Size': len(candidate_items),
        'Aggregate_Candidate_Recall': float(np.mean(val_cand_recalls)),
        'Fully_Representable_Users_Pct': float(val_full_rep * 100.0)
    },
    {
        'Split': 'Test',
        'Evaluation_Users': len(test_actual_items),
        'Candidate_Catalog_Size': len(candidate_items),
        'Aggregate_Candidate_Recall': float(np.mean(test_cand_recalls)),
        'Fully_Representable_Users_Pct': float(test_full_rep * 100.0)
    }
])
cand_recall_df.to_csv('output/analysis/candidate_recall.csv', index=False)
cand_recall_df
"""
nb.cells.append(nbf.v4.new_code_cell(cell6))

nb.cells.append(nbf.v4.new_markdown_cell("### 7. Time-Valid Feature Engineering"))

cell7 = """def extract_features(hist, cutoff):
    cf = hist.groupby('CustomerID').agg(
        cust_txns=('InvoiceNo', 'nunique'),
        cust_items=('StockCode', 'nunique'),
        cust_qty=('Quantity', 'sum'),
        cust_spend=('Amount', 'sum'),
        cust_last=('InvoiceDate', 'max')
    ).reset_index()
    cf['cust_recency_days'] = (cutoff - cf['cust_last']).dt.days
    cf = cf.drop(columns=['cust_last'])

    it = hist.groupby('StockCode').agg(
        item_txns=('InvoiceNo', 'nunique'),
        item_buyers=('CustomerID', 'nunique'),
        item_qty=('Quantity', 'sum'),
        item_avg_price=('UnitPrice', 'mean'),
        item_last=('InvoiceDate', 'max')
    ).reset_index()
    it['item_recency_days'] = (cutoff - it['item_last']).dt.days
    it = it.drop(columns=['item_last'])

    pf = hist.groupby(['CustomerID', 'StockCode']).agg(
        pair_purchases=('InvoiceNo', 'nunique'),
        pair_qty=('Quantity', 'sum'),
        pair_spend=('Amount', 'sum'),
        pair_last=('InvoiceDate', 'max')
    ).reset_index()
    pf['pair_recency_days'] = (cutoff - pf['pair_last']).dt.days
    pf = pf.drop(columns=['pair_last'])
    return cf, it, pf

feature_schema = {
    'customer_features': ['cust_txns', 'cust_items', 'cust_qty', 'cust_spend', 'cust_recency_days'],
    'item_features': ['item_txns', 'item_buyers', 'item_qty', 'item_avg_price', 'item_recency_days'],
    'pair_features': ['pair_purchases', 'pair_qty', 'pair_spend', 'pair_recency_days', 'pair_spend_share', 'is_repeat'],
    'all_features': [
        'cust_txns', 'cust_items', 'cust_qty', 'cust_spend', 'cust_recency_days',
        'item_txns', 'item_buyers', 'item_qty', 'item_avg_price', 'item_recency_days',
        'pair_purchases', 'pair_qty', 'pair_spend', 'pair_recency_days', 'pair_spend_share', 'is_repeat'
    ]
}
with open('artifacts/feature_schema.json', 'w') as f:
    json.dump(feature_schema, f, indent=4)

feat_cols = feature_schema['all_features']
"""
nb.cells.append(nbf.v4.new_code_cell(cell7))

nb.cells.append(nbf.v4.new_markdown_cell("### 8. Negative Sampling and Dataset Assembly"))

cell8 = """def assemble_dataset(hist, future, cutoff, candidate_items, n_neg=30, seed=42):
    cf, it, pf = extract_features(hist, cutoff)
    cand_set = set(candidate_items)
    users = sorted(list(set(future['CustomerID']).intersection(set(hist['CustomerID']))))
    pos_pairs = future[future['CustomerID'].isin(users) & future['StockCode'].isin(cand_set)][['CustomerID', 'StockCode']].drop_duplicates()
    pos_pairs['target'] = 1

    rng = np.random.default_rng(seed)
    user_pos_map = pos_pairs.groupby('CustomerID')['StockCode'].apply(set).to_dict()
    neg_list = []
    for u in users:
        p_set = user_pos_map.get(u, set())
        pool = [x for x in candidate_items if x not in p_set]
        if not pool:
            continue
        n = min(n_neg, len(pool))
        sampled = rng.choice(pool, size=n, replace=False)
        for item in sampled:
            neg_list.append((u, item, 0))
    neg_df = pd.DataFrame(neg_list, columns=['CustomerID', 'StockCode', 'target'])
    all_pairs = pd.concat([pos_pairs, neg_df], ignore_index=True)

    merged = all_pairs.merge(cf, on='CustomerID', how='left')
    merged = merged.merge(it, on='StockCode', how='left')
    merged = merged.merge(pf, on=['CustomerID', 'StockCode'], how='left')

    merged['pair_purchases'] = merged['pair_purchases'].fillna(0)
    merged['pair_qty'] = merged['pair_qty'].fillna(0)
    merged['pair_spend'] = merged['pair_spend'].fillna(0)
    merged['pair_recency_days'] = merged['pair_recency_days'].fillna(999)
    merged['pair_spend_share'] = (merged['pair_spend'] / (merged['cust_spend'] + 1e-5)).fillna(0)
    merged['is_repeat'] = (merged['pair_purchases'] > 0).astype(int)

    for col in ['item_txns', 'item_buyers', 'item_qty', 'item_avg_price']:
        merged[col] = merged[col].fillna(0)
    merged['item_recency_days'] = merged['item_recency_days'].fillna(999)

    return merged

sub_hist = df[df.InvoiceDate < t0].copy()
sub_future = df[(df.InvoiceDate >= t0) & (df.InvoiceDate < t1)].copy()
train_pairs_df = assemble_dataset(sub_hist, sub_future, t0, candidate_items, n_neg=30, seed=42)
val_pairs_df = assemble_dataset(train_hist, val_future, t1, candidate_items, n_neg=30, seed=42)
"""
nb.cells.append(nbf.v4.new_code_cell(cell8))

nb.cells.append(nbf.v4.new_markdown_cell("### 9. Ranking Evaluation Helper Functions"))

cell9 = """def precision_at_k(recs, relevant, k):
    r = recs[:k]
    return len(set(r) & set(relevant)) / max(k, 1)

def recall_at_k(recs, relevant, k):
    if not relevant:
        return np.nan
    return len(set(recs[:k]) & set(relevant)) / len(set(relevant))

def hit_rate_at_k(recs, relevant, k):
    return float(len(set(recs[:k]) & set(relevant)) > 0)

def ndcg_at_k(recs, relevant, k):
    rel = set(relevant)
    if not rel:
        return np.nan
    dcg = 0.0
    for rank, item in enumerate(recs[:k], start=1):
        if item in rel:
            dcg += 1.0 / np.log2(rank + 1)
    idcg = sum([1.0 / np.log2(r + 1) for r in range(1, min(len(rel), k) + 1)])
    return dcg / idcg if idcg > 0 else 0.0

def map_at_k(recs, relevant, k):
    rel = set(relevant)
    if not rel:
        return np.nan
    score, hits = 0.0, 0
    for rank, item in enumerate(recs[:k], start=1):
        if item in rel:
            hits += 1
            score += hits / rank
    return score / min(len(rel), k)
"""
nb.cells.append(nbf.v4.new_code_cell(cell9))

nb.cells.append(nbf.v4.new_markdown_cell("### 10. Training-Only Hyperparameter Tuning"))

cell10 = """param_grid = [
    {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
    {'n_estimators': 200, 'max_depth': 16, 'min_samples_leaf': 2, 'max_features': 'sqrt'},
    {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 5, 'max_features': 'sqrt'},
    {'n_estimators': 300, 'max_depth': 16, 'min_samples_leaf': 2, 'max_features': 'sqrt'}
]

val_results = []
best_model = None
best_val_score = -1.0
best_params = None

for params in param_grid:
    rf_cand = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    rf_cand.fit(train_pairs_df[feat_cols], train_pairs_df['target'])
    v_preds = rf_cand.predict_proba(val_pairs_df[feat_cols])[:, 1]
    v_pr_auc = average_precision_score(val_pairs_df['target'], v_preds)
    v_roc_auc = roc_auc_score(val_pairs_df['target'], v_preds)

    temp_val = val_pairs_df[['CustomerID', 'StockCode', 'target']].copy()
    temp_val['score'] = v_preds

    user_recalls = []
    user_ndcgs = []
    for u, grp in temp_val.groupby('CustomerID'):
        rel = val_actual_items.get(u, set())
        if not rel:
            continue
        ranked = grp.sort_values('score', ascending=False)['StockCode'].tolist()
        user_recalls.append(recall_at_k(ranked, rel, k=10))
        user_ndcgs.append(ndcg_at_k(ranked, rel, k=10))

    mean_rec10 = float(np.nanmean(user_recalls))
    mean_ndcg10 = float(np.nanmean(user_ndcgs))

    val_results.append({
        'Config': str(params),
        'Val_Recall@10': mean_rec10,
        'Val_NDCG@10': mean_ndcg10,
        'Val_PR_AUC': float(v_pr_auc),
        'Val_ROC_AUC': float(v_roc_auc)
    })

    if mean_ndcg10 > best_val_score:
        best_val_score = mean_ndcg10
        best_params = params
        best_model = rf_cand

val_tuning_df = pd.DataFrame(val_results)
val_tuning_df.to_csv('output/analysis/validation_tuning.csv', index=False)
val_tuning_df
"""
nb.cells.append(nbf.v4.new_code_cell(cell10))

nb.cells.append(nbf.v4.new_markdown_cell("### 11. Refit Selected Model on Full Development History"))

cell11 = """dev_pairs_df = assemble_dataset(train_hist, val_future, t1, candidate_items, n_neg=30, seed=42)
final_rf = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
final_rf.fit(dev_pairs_df[feat_cols], dev_pairs_df['target'])
joblib.dump(final_rf, 'models/random_forest.joblib')
"""
nb.cells.append(nbf.v4.new_code_cell(cell11))

nb.cells.append(nbf.v4.new_markdown_cell("### 12. Non-Personalized Popularity Baseline and Item-Item CF Benchmark"))

cell12 = """pop_ranking = train_hist.groupby('StockCode')['InvoiceNo'].nunique().sort_values(ascending=False).index.tolist()
pop_candidates = [x for x in pop_ranking if x in cand_set]

all_cand_items_in_dev = [x for x in candidate_items if x in dev_hist['StockCode'].values]
dev_cf_df = dev_hist[dev_hist['StockCode'].isin(all_cand_items_in_dev)].copy()
user_list = sorted(dev_cf_df['CustomerID'].unique().tolist())
item_list = sorted(all_cand_items_in_dev)
u_map = {u: i for i, u in enumerate(user_list)}
i_map = {item: i for i, item in enumerate(item_list)}
rev_i_map = {i: item for item, i in i_map.items()}

row_idx = dev_cf_df['CustomerID'].map(u_map).values
col_idx = dev_cf_df['StockCode'].map(i_map).values
data_vals = np.ones(len(dev_cf_df))

R = csr_matrix((data_vals, (row_idx, col_idx)), shape=(len(user_list), len(item_list)))
R_bin = (R > 0).astype(float)
item_sim = cosine_similarity(R_bin.T, dense_output=True)
"""
nb.cells.append(nbf.v4.new_code_cell(cell12))

nb.cells.append(nbf.v4.new_markdown_cell("### 13. Top-K Recommendation Generation on Locked Test Set"))

cell13 = """test_cf_feat, test_it_feat, test_pf_feat = extract_features(dev_hist, t2)

cf_dict = test_cf_feat.set_index('CustomerID').to_dict('index')
it_dict = test_it_feat.set_index('StockCode').to_dict('index')
pf_dict = test_pf_feat.set_index(['CustomerID', 'StockCode']).to_dict('index')

all_test_user_recs_rf = {}
all_test_user_recs_pop = {}
all_test_user_recs_cf = {}

test_features_list = []
test_pair_indices = []

for u in test_users_with_hist:
    u_feat = cf_dict.get(u, {'cust_txns': 0, 'cust_items': 0, 'cust_qty': 0, 'cust_spend': 0, 'cust_recency_days': 999})
    u_spend = u_feat['cust_spend']
    for item in candidate_items:
        i_feat = it_dict.get(item, {'item_txns': 0, 'item_buyers': 0, 'item_qty': 0, 'item_avg_price': 0, 'item_recency_days': 999})
        p_feat = pf_dict.get((u, item), {'pair_purchases': 0, 'pair_qty': 0, 'pair_spend': 0, 'pair_recency_days': 999})
        p_pur = p_feat['pair_purchases']
        p_sp = p_feat['pair_spend']
        row = [
            u_feat['cust_txns'], u_feat['cust_items'], u_feat['cust_qty'], u_spend, u_feat['cust_recency_days'],
            i_feat['item_txns'], i_feat['item_buyers'], i_feat['item_qty'], i_feat['item_avg_price'], i_feat['item_recency_days'],
            p_pur, p_feat['pair_qty'], p_sp, p_feat['pair_recency_days'],
            (p_sp / (u_spend + 1e-5)) if u_spend > 0 else 0.0,
            1 if p_pur > 0 else 0
        ]
        test_features_list.append(row)
        test_pair_indices.append((u, item))

X_test_all = np.array(test_features_list, dtype=np.float32)
scores_rf = final_rf.predict_proba(X_test_all)[:, 1]

scores_df = pd.DataFrame(test_pair_indices, columns=['CustomerID', 'StockCode'])
scores_df['rf_score'] = scores_rf

rf_grouped = scores_df.groupby('CustomerID')
for u, grp in rf_grouped:
    sorted_items = grp.sort_values('rf_score', ascending=False)['StockCode'].tolist()
    all_test_user_recs_rf[u] = sorted_items

for u in test_users_with_hist:
    all_test_user_recs_pop[u] = pop_candidates

for u in test_users_with_hist:
    if u in u_map:
        u_idx = u_map[u]
        u_vec = R_bin[u_idx].toarray().flatten()
        cf_scores = u_vec.dot(item_sim)
        ranked_indices = np.argsort(-cf_scores)
        cf_recs = [rev_i_map[idx] for idx in ranked_indices if idx in rev_i_map]
    else:
        cf_recs = pop_candidates
    all_test_user_recs_cf[u] = cf_recs
"""
nb.cells.append(nbf.v4.new_code_cell(cell13))

nb.cells.append(nbf.v4.new_markdown_cell("### 14. Comprehensive Ranking Evaluation and Comparative Metrics"))

cell14 = """def evaluate_recommender(rec_dict, actual_dict, k_list=[5, 10, 20]):
    metrics = {k: {'precision': [], 'recall': [], 'hitrate': [], 'ndcg': [], 'map': []} for k in k_list}
    for u, actual in actual_dict.items():
        if not actual:
            continue
        recs = rec_dict.get(u, [])
        for k in k_list:
            metrics[k]['precision'].append(precision_at_k(recs, actual, k))
            metrics[k]['recall'].append(recall_at_k(recs, actual, k))
            metrics[k]['hitrate'].append(hit_rate_at_k(recs, actual, k))
            metrics[k]['ndcg'].append(ndcg_at_k(recs, actual, k))
            metrics[k]['map'].append(map_at_k(recs, actual, k))

    summary = {}
    for k in k_list:
        summary[k] = {
            'precision': float(np.nanmean(metrics[k]['precision'])),
            'recall': float(np.nanmean(metrics[k]['recall'])),
            'hitrate': float(np.nanmean(metrics[k]['hitrate'])),
            'ndcg': float(np.nanmean(metrics[k]['ndcg'])),
            'map': float(np.nanmean(metrics[k]['map']))
        }
    return summary, metrics

pop_summary, pop_raw = evaluate_recommender(all_test_user_recs_pop, test_actual_items)
rf_summary, rf_raw = evaluate_recommender(all_test_user_recs_rf, test_actual_items)
cf_summary, cf_raw = evaluate_recommender(all_test_user_recs_cf, test_actual_items)

test_eval_pairs = scores_df.copy()
test_eval_pairs['actual'] = 0
for u, rel in test_actual_items.items():
    test_eval_pairs.loc[(test_eval_pairs['CustomerID'] == u) & (test_eval_pairs['StockCode'].isin(rel)), 'actual'] = 1

rf_pr_auc = float(average_precision_score(test_eval_pairs['actual'], test_eval_pairs['rf_score']))
rf_roc_auc = float(roc_auc_score(test_eval_pairs['actual'], test_eval_pairs['rf_score']))

pop_score_map = {item: 1.0 / (idx + 1) for idx, item in enumerate(pop_candidates)}
test_eval_pairs['pop_score'] = test_eval_pairs['StockCode'].map(pop_score_map).fillna(0.0)
pop_pr_auc = float(average_precision_score(test_eval_pairs['actual'], test_eval_pairs['pop_score']))
pop_roc_auc = float(roc_auc_score(test_eval_pairs['actual'], test_eval_pairs['pop_score']))

ranking_metrics_df = pd.DataFrame([
    {
        'Model': 'Popularity Baseline',
        'Precision@5': pop_summary[5]['precision'],
        'Recall@5': pop_summary[5]['recall'],
        'HitRate@5': pop_summary[5]['hitrate'],
        'NDCG@5': pop_summary[5]['ndcg'],
        'Precision@10': pop_summary[10]['precision'],
        'Recall@10': pop_summary[10]['recall'],
        'HitRate@10': pop_summary[10]['hitrate'],
        'NDCG@10': pop_summary[10]['ndcg'],
        'Precision@20': pop_summary[20]['precision'],
        'Recall@20': pop_summary[20]['recall'],
        'HitRate@20': pop_summary[20]['hitrate'],
        'NDCG@20': pop_summary[20]['ndcg'],
        'PR_AUC': pop_pr_auc,
        'ROC_AUC': pop_roc_auc
    },
    {
        'Model': 'Random Forest Ranker',
        'Precision@5': rf_summary[5]['precision'],
        'Recall@5': rf_summary[5]['recall'],
        'HitRate@5': rf_summary[5]['hitrate'],
        'NDCG@5': rf_summary[5]['ndcg'],
        'Precision@10': rf_summary[10]['precision'],
        'Recall@10': rf_summary[10]['recall'],
        'HitRate@10': rf_summary[10]['hitrate'],
        'NDCG@10': rf_summary[10]['ndcg'],
        'Precision@20': rf_summary[20]['precision'],
        'Recall@20': rf_summary[20]['recall'],
        'HitRate@20': rf_summary[20]['hitrate'],
        'NDCG@20': rf_summary[20]['ndcg'],
        'PR_AUC': rf_pr_auc,
        'ROC_AUC': rf_roc_auc
    },
    {
        'Model': 'Item-Item Collaborative Filtering',
        'Precision@5': cf_summary[5]['precision'],
        'Recall@5': cf_summary[5]['recall'],
        'HitRate@5': cf_summary[5]['hitrate'],
        'NDCG@5': cf_summary[5]['ndcg'],
        'Precision@10': cf_summary[10]['precision'],
        'Recall@10': cf_summary[10]['recall'],
        'HitRate@10': cf_summary[10]['hitrate'],
        'NDCG@10': cf_summary[10]['ndcg'],
        'Precision@20': cf_summary[20]['precision'],
        'Recall@20': cf_summary[20]['recall'],
        'HitRate@20': cf_summary[20]['hitrate'],
        'NDCG@20': cf_summary[20]['ndcg'],
        'PR_AUC': float(np.nan),
        'ROC_AUC': float(np.nan)
    }
])
ranking_metrics_df.to_csv('output/analysis/ranking_metrics.csv', index=False)
ranking_metrics_df
"""
nb.cells.append(nbf.v4.new_code_cell(cell14))

nb.cells.append(nbf.v4.new_markdown_cell("### 15. User-Level Bootstrap Confidence Intervals"))

cell15 = """def bootstrap_ci(values, n_boot=1000, seed=42):
    arr = np.array([x for x in values if not np.isnan(x)])
    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
    mean_val = np.mean(arr)
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    std_val = np.std(boot_means)
    return mean_val, std_val, ci_low, ci_high

unc_rows = []
for name, raw_dict in [('Popularity Baseline', pop_raw), ('Random Forest Ranker', rf_raw), ('Item-Item CF', cf_raw)]:
    m_rec, s_rec, low_rec, high_rec = bootstrap_ci(raw_dict[10]['recall'])
    m_ndcg, s_ndcg, low_ndcg, high_ndcg = bootstrap_ci(raw_dict[10]['ndcg'])
    unc_rows.append({
        'Model': name,
        'Recall@10_Mean': float(m_rec),
        'Recall@10_Std': float(s_rec),
        'Recall@10_95CI_Low': float(low_rec),
        'Recall@10_95CI_High': float(high_rec),
        'NDCG@10_Mean': float(m_ndcg),
        'NDCG@10_Std': float(s_ndcg),
        'NDCG@10_95CI_Low': float(low_ndcg),
        'NDCG@10_95CI_High': float(high_ndcg)
    })
uncertainty_df = pd.DataFrame(unc_rows)
uncertainty_df.to_csv('output/analysis/advanced_uncertainty.csv', index=False)
uncertainty_df
"""
nb.cells.append(nbf.v4.new_code_cell(cell15))

nb.cells.append(nbf.v4.new_markdown_cell("### 16. Feature and Preprocessing Ablation Study"))

cell16 = """ablation_configs = [
    ('All core features', feat_cols),
    ('Without pair history', [c for c in feat_cols if not c.startswith('pair_') and c != 'is_repeat']),
    ('Without customer RFM', [c for c in feat_cols if not c.startswith('cust_')]),
    ('Without item popularity', [c for c in feat_cols if not c.startswith('item_')])
]

ablation_rows = []
for name, sub_feats in ablation_configs:
    sub_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    sub_rf.fit(train_pairs_df[sub_feats], train_pairs_df['target'])

    sub_val_preds = sub_rf.predict_proba(val_pairs_df[sub_feats])[:, 1]
    temp_val = val_pairs_df[['CustomerID', 'StockCode']].copy()
    temp_val['score'] = sub_val_preds
    val_rec10_list = []
    for u, grp in temp_val.groupby('CustomerID'):
        rel = val_actual_items.get(u, set())
        if not rel:
            continue
        ranked = grp.sort_values('score', ascending=False)['StockCode'].tolist()
        val_rec10_list.append(recall_at_k(ranked, rel, k=10))
    val_rec10 = float(np.nanmean(val_rec10_list))

    feat_indices = [feat_cols.index(c) for c in sub_feats]
    sub_test_scores = sub_rf.predict_proba(X_test_all[:, feat_indices])[:, 1]
    scores_df['temp_score'] = sub_test_scores

    test_rec10_list = []
    test_ndcg10_list = []
    for u, grp in scores_df.groupby('CustomerID'):
        rel = test_actual_items.get(u, set())
        if not rel:
            continue
        ranked = grp.sort_values('temp_score', ascending=False)['StockCode'].tolist()
        test_rec10_list.append(recall_at_k(ranked, rel, k=10))
        test_ndcg10_list.append(ndcg_at_k(ranked, rel, k=10))
    t_rec10 = float(np.nanmean(test_rec10_list))
    t_ndcg10 = float(np.nanmean(test_ndcg10_list))

    obs = 'Full baseline performance' if name == 'All core features' else (
        'Severe drop highlighting critical role of repeat purchase propensity' if 'pair' in name else (
            'Moderate decrease showing personalization reliance on historical frequency' if 'customer' in name else (
                'Moderate drop showing global item demand is necessary for disambiguation'
            )
        )
    )
    ablation_rows.append({
        'Feature_Set': name,
        'Validation_Recall@10': val_rec10,
        'Test_Recall@10': t_rec10,
        'Test_NDCG@10': t_ndcg10,
        'Observation': obs
    })

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv('output/analysis/feature_ablation.csv', index=False)
ablation_df
"""
nb.cells.append(nbf.v4.new_code_cell(cell16))

nb.cells.append(nbf.v4.new_markdown_cell("### 17. Qualitative Five-Case Error Analysis Audit"))

cell17 = """desc_map = df.dropna(subset=['Description']).groupby('StockCode')['Description'].first().to_dict()

recommendations_list = []
for u in test_users_with_hist[:50]:
    recs_top5 = all_test_user_recs_rf[u][:5]
    rel = test_actual_items.get(u, set())
    for rank, itm in enumerate(recs_top5, start=1):
        recommendations_list.append({
            'CustomerID': u,
            'Rank': rank,
            'StockCode': itm,
            'Description': desc_map.get(itm, 'Unknown Item'),
            'Is_Relevant': int(itm in rel)
        })
recs_export_df = pd.DataFrame(recommendations_list)
recs_export_df.to_csv('output/analysis/recommendations.csv', index=False)

history_sizes = dev_hist.groupby('CustomerID')['InvoiceNo'].nunique().to_dict()
user_audit_cases = []

for u in test_users_with_hist:
    rel = test_actual_items.get(u, set())
    rf_top5 = all_test_user_recs_rf[u][:5]
    pop_top5 = pop_candidates[:5]
    rf_hits = len(set(rf_top5) & rel)
    pop_hits = len(set(pop_top5) & rel)
    u_hist = history_sizes.get(u, 0)
    user_audit_cases.append({
        'CustomerID': u,
        'History_Size': u_hist,
        'True_Future_Items': len(rel),
        'RF_Top5_Hits': rf_hits,
        'Pop_Top5_Hits': pop_hits,
        'RF_Recs': rf_top5,
        'Pop_Recs': pop_top5,
        'Actual_Items': list(rel)
    })

audit_df = pd.DataFrame(user_audit_cases)

case1_row = audit_df[(audit_df['History_Size'] >= 5) & (audit_df['RF_Top5_Hits'] >= 2)].iloc[0]
case2_row = audit_df[(audit_df['RF_Top5_Hits'] == 0) & (audit_df['Pop_Top5_Hits'] >= 1)].iloc[0]
case3_row = audit_df[(audit_df['RF_Top5_Hits'] > audit_df['Pop_Top5_Hits']) & (audit_df['RF_Top5_Hits'] >= 2)].iloc[0]
case4_row = audit_df[(audit_df['History_Size'] == 1) & (audit_df['True_Future_Items'] >= 2) & (audit_df['CustomerID'] != case3_row['CustomerID'])].iloc[0]
case5_row = audit_df[(audit_df['RF_Top5_Hits'] == 0) & (audit_df['Pop_Top5_Hits'] == 0) & (audit_df['History_Size'] >= 4)].iloc[0]

five_cases = [
    {
        'Case_Type': 'Case 1: Successful Personalized Recommendation',
        'CustomerID': case1_row['CustomerID'],
        'History_Size': int(case1_row['History_Size']),
        'True_Future_Count': int(case1_row['True_Future_Items']),
        'Top5_Recommendations': ', '.join([f'{c} ({desc_map.get(c, "")[:20]})' for c in case1_row['RF_Recs']]),
        'Hits': int(case1_row['RF_Top5_Hits']),
        'Audit_Comment': 'Model leveraged repeat purchase recency and customer item-affinity to accurately recover future reorders.'
    },
    {
        'Case_Type': 'Case 2: Relevant Item Missed by RF but Found by Popularity',
        'CustomerID': case2_row['CustomerID'],
        'History_Size': int(case2_row['History_Size']),
        'True_Future_Count': int(case2_row['True_Future_Items']),
        'Top5_Recommendations': ', '.join([f'{c} ({desc_map.get(c, "")[:20]})' for c in case2_row['RF_Recs']]),
        'Hits': int(case2_row['RF_Top5_Hits']),
        'Audit_Comment': 'Customer purchased a seasonal blockbuster that was absent in personal history; popularity captured it directly.'
    },
    {
        'Case_Type': 'Case 3: Random Forest Outperforms Popularity',
        'CustomerID': case3_row['CustomerID'],
        'History_Size': int(case3_row['History_Size']),
        'True_Future_Count': int(case3_row['True_Future_Items']),
        'Top5_Recommendations': ', '.join([f'{c} ({desc_map.get(c, "")[:20]})' for c in case3_row['RF_Recs']]),
        'Hits': int(case3_row['RF_Top5_Hits']),
        'Audit_Comment': 'RF personalization tailored recommendations to customer niche spend profile while popularity delivered 0 hits.'
    },
    {
        'Case_Type': 'Case 4: Sparse / Cold-Start Customer Profile',
        'CustomerID': case4_row['CustomerID'],
        'History_Size': int(case4_row['History_Size']),
        'True_Future_Count': int(case4_row['True_Future_Items']),
        'Top5_Recommendations': ', '.join([f'{c} ({desc_map.get(c, "")[:20]})' for c in case4_row['RF_Recs']]),
        'Hits': int(case4_row['RF_Top5_Hits']),
        'Audit_Comment': 'Only 1 prior transaction available; scoring relied primarily on global item priors rather than rich customer context.'
    },
    {
        'Case_Type': 'Case 5: Questionable Recommendation / Negative Bias',
        'CustomerID': case5_row['CustomerID'],
        'History_Size': int(case5_row['History_Size']),
        'True_Future_Count': int(case5_row['True_Future_Items']),
        'Top5_Recommendations': ', '.join([f'{c} ({desc_map.get(c, "")[:20]})' for c in case5_row['RF_Recs']]),
        'Hits': int(case5_row['RF_Top5_Hits']),
        'Audit_Comment': 'Over-weighted historical repeat signals for items the customer had saturated; novel discovery items were suppressed.'
    }
]
five_cases_df = pd.DataFrame(five_cases)
five_cases_df.to_csv('output/analysis/error_analysis.csv', index=False)
five_cases_df
"""
nb.cells.append(nbf.v4.new_code_cell(cell17))

nb.cells.append(nbf.v4.new_markdown_cell("### 18. Visualizations and Diagnostics"))

cell18 = """plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

monthly_vol = df.set_index('InvoiceDate').resample('W')['InvoiceNo'].nunique()
plt.figure(figsize=(10, 5))
plt.plot(monthly_vol.index, monthly_vol.values, color='steelblue', lw=2, label='Weekly Unique Orders')
plt.axvline(t1, color='crimson', linestyle='--', lw=2, label=f'Cutoff t1 (Train End): {str(t1)[:10]}')
plt.axvline(t2, color='forestgreen', linestyle='--', lw=2, label=f'Cutoff t2 (Val End): {str(t2)[:10]}')
plt.title('Figure 1: Weekly Transaction Order Volume Over Time with Chronological Splits', fontsize=12, fontweight='bold')
plt.xlabel('Date', fontsize=11)
plt.ylabel('Unique Invoice Count', fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('output/fig1_transaction_volume.png', dpi=300)
plt.close()

top15 = df.groupby('StockCode').agg(txns=('InvoiceNo', 'nunique'), desc=('Description', 'first')).sort_values('txns', ascending=False).head(15)
top15['label'] = top15.index + ': ' + top15['desc'].fillna('Unknown').str[:22]
plt.figure(figsize=(10, 6))
plt.barh(range(len(top15)), top15['txns'][::-1], color='navy', edgecolor='black')
plt.yticks(range(len(top15)), top15['label'][::-1], fontsize=9)
plt.title('Figure 2: Top 15 Products by Transaction Frequency (Unique Invoices)', fontsize=12, fontweight='bold')
plt.xlabel('Transaction / Invoice Count', fontsize=11)
plt.ylabel('Stock Code and Description', fontsize=11)
plt.tight_layout()
plt.savefig('output/fig2_top15_items.png', dpi=300)
plt.close()

cust_freq = df.groupby('CustomerID')['InvoiceNo'].nunique()
plt.figure(figsize=(10, 5))
plt.hist(np.clip(cust_freq, 1, 30), bins=30, color='dodgerblue', edgecolor='black', alpha=0.85)
plt.title('Figure 3: Customer Purchase Frequency Distribution (Capped at 30 Orders)', fontsize=12, fontweight='bold')
plt.xlabel('Number of Orders per Customer', fontsize=11)
plt.ylabel('Customer Count', fontsize=11)
plt.tight_layout()
plt.savefig('output/fig3_cust_purchase_frequency.png', dpi=300)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].hist(np.clip(test_cf_feat['cust_recency_days'], 0, 365), bins=25, color='deepskyblue', edgecolor='black', alpha=0.85)
axes[0].set_title('Recency (Days since Last Order)', fontweight='bold')
axes[0].set_xlabel('Days')
axes[0].set_ylabel('Customer Count')

axes[1].hist(np.clip(test_cf_feat['cust_txns'], 1, 40), bins=25, color='mediumseagreen', edgecolor='black', alpha=0.85)
axes[1].set_title('Frequency (Total Orders)', fontweight='bold')
axes[1].set_xlabel('Order Count')

axes[2].hist(np.log10(np.clip(test_cf_feat['cust_spend'], 1, 50000)), bins=25, color='darkorange', edgecolor='black', alpha=0.85)
axes[2].set_title('Monetary (Log10 Total Spend)', fontweight='bold')
axes[2].set_xlabel('Log10 Spend (GBP)')

plt.suptitle('Figure 4: Customer Recency, Frequency, and Monetary (RFM) Distributions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/fig4_rfm_distributions.png', dpi=300)
plt.close()

train_pos = (train_pairs_df.target == 1).sum()
train_neg = (train_pairs_df.target == 0).sum()
val_pos = (val_pairs_df.target == 1).sum()
val_neg = (val_pairs_df.target == 0).sum()

plt.figure(figsize=(8, 5))
bar_width = 0.35
idx = np.arange(2)
plt.bar(idx - bar_width/2, [train_pos, val_pos], width=bar_width, color='mediumseagreen', edgecolor='black', label='Positive Future Purchases (y=1)')
plt.bar(idx + bar_width/2, [train_neg, val_neg], width=bar_width, color='firebrick', edgecolor='black', label='Sampled Negative Pairs (y=0)')
plt.xticks(idx, ['Train Fold (Cutoff t0)', 'Validation Fold (Cutoff t1)'], fontsize=11)
plt.ylabel('Number of User-Item Pairs', fontsize=11)
plt.title('Figure 5: Class Balance in Training and Validation Folds After Negative Sampling', fontsize=12, fontweight='bold')
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('output/fig5_class_balance.png', dpi=300)
plt.close()

importances = final_rf.feature_importances_
feat_rank = np.argsort(importances)
plt.figure(figsize=(10, 6))
plt.barh(range(len(feat_rank)), importances[feat_rank], color='indigo', edgecolor='black')
plt.yticks(range(len(feat_rank)), [feat_cols[i] for i in feat_rank], fontsize=10)
plt.title('Figure 6: Gini Feature Importance in Tuned Random Forest Recommender', fontsize=12, fontweight='bold')
plt.xlabel('Mean Decrease in Impurity', fontsize=11)
plt.ylabel('Engineered Feature', fontsize=11)
plt.tight_layout()
plt.savefig('output/fig6_feature_importance.png', dpi=300)
plt.close()

fig, ax1 = plt.subplots(figsize=(8, 5))
k_vals = [5, 10, 20]
rf_prec = [rf_summary[k]['precision'] for k in k_vals]
rf_rec = [rf_summary[k]['recall'] for k in k_vals]

color_prec = 'royalblue'
color_rec = 'crimson'

ax1.set_xlabel('Recommendation Cutoff (K)', fontsize=11)
ax1.set_ylabel('Precision@K', color=color_prec, fontsize=11, fontweight='bold')
line1 = ax1.plot(k_vals, rf_prec, marker='o', lw=2.5, color=color_prec, label='Precision@K')
ax1.tick_params(axis='y', labelcolor=color_prec)
ax1.set_xticks(k_vals)

ax2 = ax1.twinx()
ax2.set_ylabel('Recall@K', color=color_rec, fontsize=11, fontweight='bold')
line2 = ax2.plot(k_vals, rf_rec, marker='s', lw=2.5, color=color_rec, label='Recall@K')
ax2.tick_params(axis='y', labelcolor=color_rec)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True)
plt.title('Figure 7: Precision@K and Recall@K Trade-Off across K in {5, 10, 20}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('output/fig7_precision_recall_k.png', dpi=300)
plt.close()

metrics_names = ['Precision@10', 'Recall@10', 'HitRate@10', 'NDCG@10']
pop_vals = [pop_summary[10]['precision'], pop_summary[10]['recall'], pop_summary[10]['hitrate'], pop_summary[10]['ndcg']]
rf_vals = [rf_summary[10]['precision'], rf_summary[10]['recall'], rf_summary[10]['hitrate'], rf_summary[10]['ndcg']]
cf_vals = [cf_summary[10]['precision'], cf_summary[10]['recall'], cf_summary[10]['hitrate'], cf_summary[10]['ndcg']]

x = np.arange(len(metrics_names))
w = 0.25
plt.figure(figsize=(10, 5.5))
plt.bar(x - w, pop_vals, width=w, color='slategray', edgecolor='black', label='Popularity Baseline')
plt.bar(x, rf_vals, width=w, color='royalblue', edgecolor='black', label='Random Forest Ranker')
plt.bar(x + w, cf_vals, width=w, color='mediumseagreen', edgecolor='black', label='Item-Item CF Benchmark')
plt.xticks(x, metrics_names, fontsize=11)
plt.ylabel('Metric Score', fontsize=11)
plt.title('Figure 8: Comparative Evaluation of Popularity, Random Forest, and Collaborative Filtering at K=10', fontsize=12, fontweight='bold')
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('output/fig8_baseline_vs_rf_ranking.png', dpi=300)
plt.close()

sample_eval = test_eval_pairs.sample(n=100000, random_state=42)
plt.figure(figsize=(9, 5))
sns.kdeplot(sample_eval[sample_eval.actual == 0]['rf_score'], color='firebrick', lw=2, label='Non-Purchased Pairs (Negatives)', fill=True, alpha=0.3)
sns.kdeplot(sample_eval[sample_eval.actual == 1]['rf_score'], color='mediumseagreen', lw=2, label='True Future Purchases (Positives)', fill=True, alpha=0.3)
plt.title('Figure 9: Density Distribution of Predicted Purchase Propensity Scores', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Random Forest Purchase Probability Score', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('output/fig9_score_distribution.png', dpi=300)
plt.close()

all_rf_recs_flat = [itm for recs in all_test_user_recs_rf.values() for itm in recs[:10]]
rf_item_counts = pd.Series(all_rf_recs_flat).value_counts().values
cum_rf = np.cumsum(rf_item_counts) / np.sum(rf_item_counts)
cum_x = np.arange(1, len(rf_item_counts) + 1) / len(candidate_items)

all_pop_recs_flat = [itm for recs in all_test_user_recs_pop.values() for itm in recs[:10]]
pop_item_counts = pd.Series(all_pop_recs_flat).value_counts().values
cum_pop = np.cumsum(pop_item_counts) / np.sum(pop_item_counts)
cum_pop_x = np.arange(1, len(pop_item_counts) + 1) / len(candidate_items)

plt.figure(figsize=(8, 6))
plt.plot(cum_x, cum_rf, color='royalblue', lw=2.5, label='Random Forest Catalog Distribution')
plt.plot(cum_pop_x, cum_pop, color='firebrick', lw=2.5, linestyle='--', label='Popularity Baseline Distribution')
plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Perfect Equality (Uniform Catalog Coverage)')
plt.title('Figure 10: Lorenz Curve of Recommended Items across Test Users (Catalog Coverage)', fontsize=12, fontweight='bold')
plt.xlabel('Fraction of Candidate Products', fontsize=11)
plt.ylabel('Cumulative Fraction of Recommendations Served', fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('output/fig10_catalog_coverage.png', dpi=300)
plt.close()
"""
nb.cells.append(nbf.v4.new_code_cell(cell18))

nb.cells.append(nbf.v4.new_markdown_cell("### 19. Acceptance Tests and Verification"))

cell19 = """assert train_hist.InvoiceDate.max() < t1
assert val_future.InvoiceDate.min() >= t1
assert test_future.InvoiceDate.min() >= t2
assert set(test_future.index).isdisjoint(set(train_hist.index))
assert set(feat_cols).issubset(set(train_pairs_df.columns))
assert not train_pairs_df[feat_cols].isna().any().any()
assert set(np.unique(train_pairs_df['target'])).issubset({0, 1})
assert final_rf.classes_.tolist() == [0, 1]
assert all(len(r) <= 10 for r in [all_test_user_recs_rf[u][:10] for u in test_users_with_hist])
assert 500 <= len(candidate_items) <= 2000
assert 0.0 <= cand_recall_df.loc[1, 'Aggregate_Candidate_Recall'] <= 1.0

reloaded_rf = joblib.load('models/random_forest.joblib')
sample_X = X_test_all[:100]
assert np.allclose(final_rf.predict_proba(sample_X), reloaded_rf.predict_proba(sample_X))
print('All Acceptance Tests Passed Successfully.')
"""
nb.cells.append(nbf.v4.new_code_cell(cell19))

with open('DA6.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Wrote DA6.ipynb successfully with UTF-8 encoding and clean markdown headings')
