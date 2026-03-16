"""
prepare_dashboard_data.py
=========================
Run this script ONCE locally to generate:
  - dashboard_data.json  (updated with embeddings, clusters, anomaly scores)
  - leaseup_model.pkl    (trained Random Forest for lease-up time prediction)

Usage:
  python prepare_dashboard_data.py

Requirements:
  pip install sentence-transformers scikit-learn pandas numpy joblib openpyxl
"""

import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sentence_transformers import SentenceTransformer
from datetime import datetime

print("=" * 60)
print("Dashboard Data Preparation Script")
print("=" * 60)

# ── 1. Load & process raw data ────────────────────────────────────────────────
print("\n[1/6] Loading MSA data...")

def load_msa(path, market_label):
    sheets = {}
    for sheet in ['Property Status', 'Occ & Concession', 'Rent']:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        month_cols = raw.iloc[2, 30:180].tolist()
        meta_cols  = raw.iloc[2, 0:30].tolist()
        data = raw.iloc[3:].reset_index(drop=True)
        meta = data.iloc[:, 0:30].copy()
        meta.columns = meta_cols
        meta['market_label'] = market_label
        if sheet == 'Property Status':
            ts = data.iloc[:, 30:180].copy(); ts.columns = month_cols
            sheets['meta'] = meta; sheets['status'] = ts
        elif sheet == 'Occ & Concession':
            occ = data.iloc[:, 30:180].copy(); occ.columns = month_cols
            con = data.iloc[:, 330:480].copy(); con.columns = month_cols
            sheets['occupancy'] = occ; sheets['concession_pct'] = con
        elif sheet == 'Rent':
            eff = data.iloc[:, 180:330].copy(); eff.columns = month_cols
            sheets['eff_rent'] = eff
    return sheets

msa1 = load_msa('MSA1.xlsx', 'Austin-Round Rock, TX')
msa2 = load_msa('MSA2.xlsx', 'Akron, OH')

MONTHS      = msa1['status'].columns.tolist()
MONTH_DATES = [datetime.strptime(m, '%b-%y') for m in MONTHS]
CUTOFF      = datetime(2008, 4, 1)

def safe_float(val):
    try: return float(val)
    except: return None

def find_delivery_idx(status_row):
    for i, val in enumerate(status_row):
        if pd.isna(val): continue
        val = str(val).strip()
        if val in ('LU', 'UC/LU'): return i if MONTH_DATES[i] >= CUTOFF else None
        else: return None
    return None

def find_leaseup_idx(occ_row, delivery_idx):
    if delivery_idx is None: return None
    for i in range(delivery_idx, len(occ_row)):
        v = occ_row.iloc[i]
        try:
            if pd.notna(v) and float(v) >= 0.90: return i
        except: continue
    return None

def get_season(dt):
    m = dt.month
    if m in (3,4,5): return 'Spring'
    if m in (6,7,8): return 'Summer'
    if m in (9,10,11): return 'Fall'
    return 'Winter'

records = []
for msa_label, msa in [('Austin-Round Rock, TX', msa1), ('Akron, OH', msa2)]:
    for idx in range(len(msa['meta'])):
        meta_row   = msa['meta'].iloc[idx]
        status_row = msa['status'].iloc[idx]
        occ_row    = msa['occupancy'].iloc[idx]
        rent_row   = msa['eff_rent'].iloc[idx]
        con_row    = msa['concession_pct'].iloc[idx]
        del_idx    = find_delivery_idx(status_row)
        if del_idx is None: continue
        lu_idx     = find_leaseup_idx(occ_row, del_idx)
        lease_up_months = int(lu_idx - del_idx) if lu_idx is not None else None
        rent_del   = safe_float(rent_row.iloc[del_idx])
        rent_lu    = safe_float(rent_row.iloc[lu_idx]) if lu_idx is not None else None
        neg_rent   = bool(rent_lu < rent_del) if (rent_del and rent_lu) else None
        rent_change_pct = round((rent_lu - rent_del)/rent_del*100, 1) if (rent_del and rent_lu) else None
        occ_vals = [safe_float(occ_row.iloc[i]) for i in range(del_idx, lu_idx+1)] if lu_idx else []
        con_vals = [safe_float(con_row.iloc[i]) for i in range(del_idx, lu_idx+1)] if lu_idx else []
        occ_vals = [v for v in occ_vals if v is not None]
        con_vals = [v for v in con_vals if v is not None]
        avg_occ  = round(float(np.mean(occ_vals)), 3) if occ_vals else None
        avg_con  = round(float(np.mean(con_vals)), 3) if con_vals else None
        area = safe_float(meta_row.get('AreaPerUnit'))
        yr   = safe_float(meta_row.get('YearBuilt'))
        qty  = safe_float(meta_row.get('Quantity'))
        del_date = MONTH_DATES[del_idx]
        rent_psf = round(rent_del/area, 3) if (rent_del and area) else None
        age_at_del = int(del_date.year - yr) if yr else None
        records.append({
            'msa': msa_label,
            'name': str(meta_row.get('Name') or '').replace('`', "'"),
            'submarket': str(meta_row.get('Submarket') or '').replace('`', "'"),
            'city': str(meta_row.get('City') or ''),
            'address': str(meta_row.get('Address') or ''),
            'proj_id': str(meta_row.get('ProjID') or ''),
            'year_built': yr, 'quantity': qty, 'area_per_unit': area,
            'latitude': safe_float(meta_row.get('Latitude')),
            'longitude': safe_float(meta_row.get('Longitude')),
            'mgmt_company': str(meta_row.get('ManagementCompany') or '').replace('`', "'"),
            'true_owner': str(meta_row.get('True Owner') or ''),
            'delivery_month': MONTHS[del_idx],
            'delivery_year': del_date.year,
            'delivery_season': get_season(del_date),
            'leaseup_month': MONTHS[lu_idx] if lu_idx is not None else None,
            'lease_up_months': lease_up_months,
            'rent_at_delivery': rent_del, 'rent_at_leaseup': rent_lu,
            'neg_rent_growth': neg_rent, 'rent_change_pct': rent_change_pct,
            'avg_occ_during_leaseup': avg_occ, 'concession_intensity': avg_con,
            'rent_per_sqft': rent_psf, 'property_age_at_delivery': age_at_del,
        })

df = pd.DataFrame(records)
print(f"  Total delivered properties: {len(df)}")

# ── 2. Sentence-Transformer Embedding ────────────────────────────────────────
print("\n[2/6] Generating sentence-transformer embeddings...")

NUM_FEATURES = ['quantity', 'area_per_unit', 'rent_per_sqft',
                'avg_occ_during_leaseup', 'concession_intensity',
                'property_age_at_delivery']

embed_df = df.dropna(subset=NUM_FEATURES).copy().reset_index(drop=True)
print(f"  Properties with complete features: {len(embed_df)}")

scaler = StandardScaler()
X_num = scaler.fit_transform(embed_df[NUM_FEATURES])

text_corpus = (
    "Apartment property located in " +
    embed_df['submarket'].fillna('unknown submarket') +
    ", managed by " +
    embed_df['mgmt_company'].fillna('unknown operator') +
    ", delivered " +
    embed_df['delivery_month'].fillna('unknown date')
)

model = SentenceTransformer('all-MiniLM-L6-v2')
X_text_raw = model.encode(text_corpus.tolist(), show_progress_bar=True)

pca_text = PCA(n_components=20, random_state=42)
X_text_comp = pca_text.fit_transform(X_text_raw)
X_combined = np.hstack([X_num, X_text_comp])

# PCA for visualization (2D)
pca_viz = PCA(n_components=2, random_state=42)
X_viz = pca_viz.fit_transform(X_combined)
print(f"  PCA variance explained: PC1={pca_viz.explained_variance_ratio_[0]:.1%}, PC2={pca_viz.explained_variance_ratio_[1]:.1%}")

# ── 3. Clustering ─────────────────────────────────────────────────────────────
print("\n[3/6] Running K-Means clustering...")

from sklearn.metrics import silhouette_score
scores = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_combined)
    scores[k] = silhouette_score(X_combined, lbl)

best_k = max(scores, key=scores.get)
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
embed_df['cluster'] = km_final.fit_predict(X_combined)
embed_df['pca1'] = X_viz[:, 0]
embed_df['pca2'] = X_viz[:, 1]

# Store full embedding vectors for similarity search (use PCA-compressed version)
pca_store = PCA(n_components=10, random_state=42)
X_store = pca_store.fit_transform(X_combined)
embed_df['embedding'] = [v.tolist() for v in X_store]

print(f"  Best k={best_k}, silhouette={scores[best_k]:.3f}")
print(f"  Cluster sizes: {embed_df['cluster'].value_counts().to_dict()}")

# ── 4. Isolation Forest Anomaly Detection ────────────────────────────────────
print("\n[4/6] Running Isolation Forest anomaly detection...")

iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
embed_df['anomaly_score'] = -iso.fit(X_combined).score_samples(X_combined)
embed_df['is_anomaly'] = iso.fit_predict(X_combined) == -1

n_anomalies = embed_df['is_anomaly'].sum()
print(f"  Anomalies detected: {n_anomalies} ({n_anomalies/len(embed_df):.1%})")

# ── 5. Predictive Model ───────────────────────────────────────────────────────
print("\n[5/6] Training lease-up time prediction model...")

pred_features = ['quantity', 'area_per_unit', 'rent_per_sqft',
                 'avg_occ_during_leaseup', 'concession_intensity',
                 'property_age_at_delivery']

model_df = embed_df.dropna(subset=pred_features + ['lease_up_months']).copy()
X_model = model_df[pred_features].values
y_model = model_df['lease_up_months'].values

# Train Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
cv_scores = cross_val_score(rf, X_model, y_model, cv=5,
                            scoring='neg_mean_absolute_error')
cv_mae = -cv_scores.mean()
print(f"  Cross-val MAE: {cv_mae:.2f} months")

rf.fit(X_model, y_model)
train_mae = mean_absolute_error(y_model, rf.predict(X_model))
print(f"  Train MAE: {train_mae:.2f} months")

# Feature importances
importances = dict(zip(pred_features, rf.feature_importances_.tolist()))
print(f"  Feature importances: { {k: round(v,3) for k,v in importances.items()} }")

# Save model + scaler + metadata
model_bundle = {
    'model': rf,
    'feature_names': pred_features,
    'feature_stats': {
        f: {'mean': float(model_df[f].mean()), 'std': float(model_df[f].std()),
            'min': float(model_df[f].min()), 'max': float(model_df[f].max())}
        for f in pred_features
    },
    'cv_mae': round(float(cv_mae), 2),
    'train_mae': round(float(train_mae), 2),
    'importances': importances,
    'n_train': len(model_df),
}
joblib.dump(model_bundle, 'leaseup_model.pkl')
print(f"  Saved: leaseup_model.pkl")

# ── 6. Build final JSON ───────────────────────────────────────────────────────
print("\n[6/6] Building dashboard_data.json...")

# Merge embedding results back to main df
merge_cols = ['proj_id', 'cluster', 'pca1', 'pca2',
              'anomaly_score', 'is_anomaly', 'embedding']
df = df.merge(
    embed_df[['proj_id'] + merge_cols[1:]],
    on='proj_id', how='left'
)

# Z-score for legacy anomaly tab
valid = df['lease_up_months'].notna()
mu = df.loc[valid, 'lease_up_months'].mean()
sd = df.loc[valid, 'lease_up_months'].std()
df['leaseup_zscore'] = None
df.loc[valid, 'leaseup_zscore'] = ((df.loc[valid, 'lease_up_months'] - mu) / sd).round(2)

# Summary tables
yr_summary = df.groupby('delivery_year').agg(
    count=('name','count'),
    avg_leaseup=('lease_up_months','mean'),
    pct_neg_rent=('neg_rent_growth', lambda x: x.dropna().mean()),
    avg_rent_psf=('rent_per_sqft','mean')
).reset_index()

season_summary = df.groupby('delivery_season').agg(
    count=('name','count'),
    avg_leaseup=('lease_up_months','mean')
).reset_index()

cluster_summary = embed_df.groupby('cluster').agg(
    count=('name','count'),
    avg_leaseup=('lease_up_months','mean'),
    avg_rent_psf=('rent_per_sqft','mean'),
    pct_neg_rent=('neg_rent_growth', lambda x: x.dropna().mean())
).reset_index()
cluster_summary['cluster'] = cluster_summary['cluster'].apply(lambda x: f'Cluster {x}')

# PCA scatter data for embedding viz
pca_scatter = embed_df[['name','submarket','msa','cluster','pca1','pca2',
                         'lease_up_months','rent_per_sqft',
                         'avg_occ_during_leaseup','concession_intensity',
                         'anomaly_score','is_anomaly']].copy()
pca_scatter['cluster_label'] = pca_scatter['cluster'].apply(lambda x: f'Cluster {x}')

# Anomaly table (top anomalies from Isolation Forest)
anomaly_table = embed_df[embed_df['is_anomaly']].copy()
anomaly_table = anomaly_table.sort_values('anomaly_score', ascending=False)
anomaly_table['anomaly_reason'] = anomaly_table.apply(lambda r: _anomaly_reason(r), axis=1) \
    if False else ''

def anomaly_reason(row):
    reasons = []
    if pd.notna(row['lease_up_months']):
        if row['lease_up_months'] > mu + 1.5*sd:
            reasons.append(f"very slow lease-up ({row['lease_up_months']:.0f} mo)")
        elif row['lease_up_months'] < mu - 1.5*sd:
            reasons.append(f"unusually fast lease-up ({row['lease_up_months']:.0f} mo)")
    if pd.notna(row['concession_intensity']) and row['concession_intensity'] > 0.15:
        reasons.append(f"high concessions ({row['concession_intensity']*100:.0f}%)")
    if pd.notna(row['rent_per_sqft']) and row['rent_per_sqft'] > 2.5:
        reasons.append(f"premium rent/sqft (${row['rent_per_sqft']:.2f})")
    return '; '.join(reasons) if reasons else 'unusual feature combination'

anomaly_table['anomaly_reason'] = anomaly_table.apply(anomaly_reason, axis=1)

def clean_records(d):
    rows = d.where(pd.notna(d), None).to_dict(orient='records')
    return [{k: (None if isinstance(v, float) and np.isnan(v) else v)
             for k, v in r.items()} for r in rows]

# Build payload
payload = {
    'properties': clean_records(df.drop(columns=['embedding'], errors='ignore')),
    'yr_summary': clean_records(yr_summary),
    'season_summary': clean_records(season_summary),
    'cluster_summary': clean_records(cluster_summary),
    'pca_scatter': clean_records(pca_scatter.drop(columns=['embedding'], errors='ignore')),
    'anomaly_table': clean_records(anomaly_table[[
        'name', 'submarket', 'delivery_month', 'lease_up_months',
        'rent_per_sqft', 'concession_intensity', 'anomaly_score', 'anomaly_reason'
    ]]),
    'embeddings': {
        row['proj_id']: row['embedding']
        for _, row in embed_df[['proj_id','name','embedding']].iterrows()
        if isinstance(row['embedding'], list)
    },
    'embed_meta': {
        row['proj_id']: {
            'name': row['name'],
            'submarket': row['submarket'],
            'cluster': int(row['cluster']) if pd.notna(row['cluster']) else None,
            'lease_up_months': row['lease_up_months'],
            'rent_per_sqft': row['rent_per_sqft'],
        }
        for _, row in embed_df.iterrows()
    },
    'model_info': {
        'cv_mae': model_bundle['cv_mae'],
        'train_mae': model_bundle['train_mae'],
        'n_train': model_bundle['n_train'],
        'importances': importances,
        'feature_stats': model_bundle['feature_stats'],
    },
    'stats': {
        'total': int(len(df)),
        'austin_count': int((df['msa']=='Austin-Round Rock, TX').sum()),
        'akron_count': int((df['msa']=='Akron, OH').sum()),
        'avg_leaseup': round(float(df['lease_up_months'].mean()), 1),
        'pct_neg_rent': round(float(df['neg_rent_growth'].dropna().mean()*100), 1),
        'mu': round(float(mu), 1),
        'sd': round(float(sd), 1),
        'n_anomalies': int(n_anomalies),
        'best_k': int(best_k),
        'sil_score': round(float(scores[best_k]), 3),
    }
}

json_str = json.dumps(payload, default=str, ensure_ascii=True)
with open('dashboard_data.json', 'w') as f:
    f.write(json_str)

print(f"  Saved: dashboard_data.json ({len(json_str)//1024}KB)")
print("\n" + "="*60)
print("Done! Files generated:")
print("  dashboard_data.json  — upload to GitHub repo")
print("  leaseup_model.pkl    — upload to GitHub repo")
print("="*60)
