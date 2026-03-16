import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import joblib
import os

st.set_page_config(
    page_title="Lease-Up Intelligence Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }
div[data-testid="stMetric"] {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 16px 20px;
    border-top: 3px solid #f0883e;
}
div[data-testid="stMetric"] label { color: #b0bac4 !important; font-size: .75rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e6edf3 !important; }
.stSelectbox > div > div { background: #161b22 !important; border-color: #21262d !important; }
.stTextInput > div > div > input { background: #161b22 !important; border-color: #21262d !important; color: #e6edf3 !important; }
.stNumberInput > div > div > input { background: #161b22 !important; border-color: #21262d !important; color: #e6edf3 !important; }
.stButton > button { background: #f0883e !important; color: #000 !important; border: none !important; font-weight: 600 !important; border-radius: 9px !important; }
.stSlider > div > div { color: #f0883e !important; }
.section-title { font-family: 'DM Serif Display'; font-size: 1.3rem; color: #e6edf3; }
.genai-badge {
    display: inline-block; background: rgba(240,136,62,.1);
    border: 1px solid rgba(240,136,62,.3); border-radius: 6px;
    padding: 2px 10px; font-size: .7rem; color: #f0883e;
    font-family: 'DM Mono'; margin-left: 8px;
}
.info-box {
    background: #161b22; border: 1px solid #21262d;
    border-left: 3px solid #f0883e;
    border-radius: 8px; padding: 14px 16px;
    font-size: .82rem; line-height: 1.7;
}
hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

PLOT_BG = '#161b22'
PAPER_BG = '#161b22'
FONT_COLOR = '#e6edf3'
GRID_COLOR = 'rgba(255,255,255,0.05)'
TICK_COLOR = '#a8b3bc'

def dark_layout(title='', height=350):
    return dict(
        title=dict(text=title, font=dict(size=13, color=FONT_COLOR)),
        plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
        font_color=FONT_COLOR, height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR, size=10)),
        yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR, size=10)),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TICK_COLOR, size=10))
    )

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, "dashboard_data.json")) as f:
        d = json.load(f)
    df  = pd.DataFrame(d['properties'])
    yr  = pd.DataFrame(d['yr_summary'])
    sea = pd.DataFrame(d['season_summary'])
    cl  = pd.DataFrame(d['cluster_summary'])
    pca = pd.DataFrame(d['pca_scatter'])
    ano = pd.DataFrame(d['anomaly_table'])
    return df, yr, sea, cl, pca, ano, d

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, 'leaseup_model.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None

df, yr_df, sea_df, cl_df, pca_df, ano_df, raw = load_data()
model_bundle = load_model()
stats = raw['stats']
embeddings = raw.get('embeddings', {})
embed_meta = raw.get('embed_meta', {})
model_info = raw.get('model_info', {})

CLUSTER_COLORS = ['#f0883e','#58a6ff','#3fb950','#f85149','#bc8cff','#ffa657']

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:1px solid #21262d;padding-bottom:16px;margin-bottom:24px">
  <div style="display:flex;align-items:center;gap:12px">
    <div style="width:40px;height:40px;background:#f0883e;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px">🏢</div>
    <div>
      <div style="font-family:'DM Serif Display';font-size:1.5rem;color:#e6edf3">Lease-Up Intelligence Dashboard</div>
      <div style="font-size:.73rem;color:#a8b3bc">Property Analytics · Affinius Capital DS Assessment 2026 · Yining (Evelyn) Huang, UT Austin PhD</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Properties Delivered", stats['total'])
c2.metric("Avg Lease-Up Time", f"{stats['avg_leaseup']} mo")
c3.metric("Negative Rent Growth", f"{stats['pct_neg_rent']}%")
c4.metric("Austin / Akron", f"{stats['austin_count']} / {stats['akron_count']}")
c5.metric("AI Anomalies Found", stats.get('n_anomalies', '—'))
st.markdown("<br>", unsafe_allow_html=True)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_market, tab_embed, tab_anomaly, tab_predict, tab_similar = st.tabs([
    "📊 Market Overview",
    "🧠 Embedding Clusters",
    "⚠ AI Anomaly Detection",
    "🔮 Lease-Up Predictor",
    "🔍 Similar Property Finder",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: Market Overview
# ════════════════════════════════════════════════════════════════════════════════
with tab_market:
    st.markdown('<div class="section-title">Market Trends</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=yr_df['delivery_year'], y=yr_df['count'],
            name='Count', marker_color='rgba(88,166,255,0.2)',
            marker_line_color='rgba(88,166,255,0.5)', marker_line_width=1
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=yr_df['delivery_year'], y=yr_df['avg_leaseup'].round(1),
            name='Avg LU (mo)', line=dict(color='#f0883e', width=2.5),
            mode='lines+markers', marker=dict(size=6, color='#f0883e'),
            fill='tozeroy', fillcolor='rgba(240,136,62,0.08)'
        ), secondary_y=False)
        fig.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font_color=FONT_COLOR,
            height=300, margin=dict(l=10,r=10,t=40,b=10), showlegend=True,
            title=dict(text='Avg Lease-Up Time by Delivery Year', font=dict(size=13, color=FONT_COLOR)),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TICK_COLOR,size=10))
        )
        fig.update_yaxes(title_text="Avg Months", secondary_y=False,
                         gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR, size=10))
        fig.update_yaxes(title_text="# Properties", secondary_y=True,
                         gridcolor='rgba(0,0,0,0)', tickfont=dict(color=TICK_COLOR, size=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        neg_pct = (yr_df['pct_neg_rent'] * 100).round(1)
        fig2 = go.Figure(go.Scatter(
            x=yr_df['delivery_year'], y=neg_pct,
            line=dict(color='#f85149', width=2.5),
            mode='lines+markers', marker=dict(size=6, color='#f85149'),
            fill='tozeroy', fillcolor='rgba(248,81,73,0.1)'
        ))
        fig2.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font_color=FONT_COLOR,
            height=300, margin=dict(l=10,r=10,t=40,b=10), showlegend=False,
            title=dict(text='Negative Rent Growth %', font=dict(size=13, color=FONT_COLOR)),
            xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR, size=10)),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR, size=10),
                       ticksuffix='%', range=[0, 100])
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Season + Histogram
    c1, c2 = st.columns(2)
    with c1:
        s_order = ['Spring','Summer','Fall','Winter']
        s_colors = {'Spring':'#3fb950','Summer':'#f0883e','Fall':'#f85149','Winter':'#58a6ff'}
        s_data = sea_df.set_index('delivery_season').reindex(s_order).reset_index()
        fig3 = go.Figure(go.Bar(
            x=s_data['delivery_season'], y=s_data['avg_leaseup'].round(1),
            marker_color=[s_colors[s] for s in s_data['delivery_season']],
            text=s_data['avg_leaseup'].round(1), textposition='outside',
            textfont=dict(color=TICK_COLOR, size=11)
        ))
        fig3.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font_color=FONT_COLOR,
            height=280, margin=dict(l=10,r=10,t=40,b=10),
            title=dict(text='Delivery Season Effect', font=dict(size=13, color=FONT_COLOR)),
            xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=10)),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=10))
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        lu_vals = df['lease_up_months'].dropna()
        fig4 = go.Figure(go.Histogram(
            x=lu_vals, nbinsx=20,
            marker_color='rgba(88,166,255,0.5)',
            marker_line_color='#58a6ff', marker_line_width=1
        ))
        fig4.add_vline(x=lu_vals.mean(), line_dash='dash', line_color='#f0883e',
                       annotation_text=f'Mean: {lu_vals.mean():.1f} mo',
                       annotation_font_color='#f0883e')
        fig4.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font_color=FONT_COLOR,
            height=280, margin=dict(l=10,r=10,t=40,b=10),
            title=dict(text='Lease-Up Distribution', font=dict(size=13, color=FONT_COLOR)),
            xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=10)),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=10))
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Property table
    st.markdown('<div class="section-title" style="margin-top:8px">Property Explorer</div>', unsafe_allow_html=True)
    f1,f2,f3,f4 = st.columns(4)
    fm  = f1.selectbox("Market", ["All"] + sorted(df['msa'].dropna().unique().tolist()))
    fcl = f2.selectbox("Cluster", ["All"] + sorted(df['cluster'].dropna().unique().tolist()) if 'cluster' in df else ["All"])
    fr  = f3.selectbox("Rent Growth", ["All","Negative Only","Positive Only"])
    fs  = f4.selectbox("Season", ["All","Spring","Summer","Fall","Winter"])

    fdf = df.copy()
    if fm  != "All": fdf = fdf[fdf['msa']==fm]
    if fcl != "All" and 'cluster' in fdf: fdf = fdf[fdf['cluster']==fcl]
    if fr  == "Negative Only": fdf = fdf[fdf['neg_rent_growth']==True]
    if fr  == "Positive Only": fdf = fdf[fdf['neg_rent_growth']==False]
    if fs  != "All": fdf = fdf[fdf['delivery_season']==fs]

    st.caption(f"{len(fdf)} properties")
    disp = fdf[['name','submarket','delivery_month','quantity',
                'lease_up_months','rent_at_delivery','rent_change_pct','cluster']].copy() \
        if 'cluster' in fdf.columns else \
        fdf[['name','submarket','delivery_month','quantity',
             'lease_up_months','rent_at_delivery','rent_change_pct']].copy()
    disp.columns = [c.replace('_',' ').title() for c in disp.columns]
    st.dataframe(disp.head(200), use_container_width=True, height=300, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: Embedding Clusters
# ════════════════════════════════════════════════════════════════════════════════
with tab_embed:
    st.markdown(
        '<div class="section-title">Transformer Embedding Clusters'
        '<span class="genai-badge">✦ GenAI · sentence-transformers</span></div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="info-box" style="margin:12px 0 20px">
    <strong>How it works:</strong> Each property is encoded into a 384-dimensional semantic vector
    using <code>all-MiniLM-L6-v2</code> (a transformer-based embedding model), capturing the
    semantic identity of its submarket and operator. This text embedding is combined with 6 numerical
    features, then compressed via PCA and clustered with K-Means. The 2D scatter below shows the
    PCA projection of this combined embedding space.
    </div>
    """, unsafe_allow_html=True)

    if not pca_df.empty:
        col1, col2 = st.columns([3, 1])
        with col1:
            n_clusters = pca_df['cluster'].nunique() if 'cluster' in pca_df.columns else 2
            color_col = 'cluster_label' if 'cluster_label' in pca_df.columns else 'cluster'
            color_map = {f'Cluster {i}': CLUSTER_COLORS[i] for i in range(n_clusters)}

            fig_emb = px.scatter(
                pca_df, x='pca1', y='pca2', color=color_col,
                color_discrete_map=color_map,
                hover_data=['name','submarket','lease_up_months','rent_per_sqft'],
                labels={'pca1':'PC1','pca2':'PC2','cluster_label':'Cluster'},
                title=f'Property Embedding Space (PCA 2D) · k={stats.get("best_k",2)}, '
                      f'silhouette={stats.get("sil_score","—")}',
            )
            fig_emb.update_traces(marker=dict(size=8, opacity=0.75, line=dict(width=0)))
            fig_emb.update_layout(
                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                font_color=FONT_COLOR, height=420,
                margin=dict(l=10,r=10,t=50,b=10),
                xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=10)),
                yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=10)),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TICK_COLOR,size=11))
            )
            st.plotly_chart(fig_emb, use_container_width=True)

        with col2:
            st.markdown("**Cluster Profiles**")
            for _, row in cl_df.iterrows():
                lu = f"{row['avg_leaseup']:.1f} mo" if pd.notna(row.get('avg_leaseup')) else '—'
                psf = f"${row['avg_rent_psf']:.2f}/sqft" if pd.notna(row.get('avg_rent_psf')) else '—'
                neg = f"{row['pct_neg_rent']*100:.0f}% neg rent" if pd.notna(row.get('pct_neg_rent')) else '—'
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #21262d;border-radius:8px;
                            padding:10px 14px;margin-bottom:8px;font-size:.8rem">
                  <strong style="color:#f0883e">{row['cluster']}</strong>
                  <div style="color:#a8b3bc;margin-top:4px">n={int(row['count'])} · {lu} · {psf}</div>
                  <div style="color:#a8b3bc">{neg}</div>
                </div>""", unsafe_allow_html=True)

        # Cluster feature profiles
        st.markdown("**Feature Profiles by Cluster**")
        profile_cols = ['avg_leaseup','avg_rent_psf','pct_neg_rent']
        profile_labels = ['Avg Lease-Up (mo)', 'Avg Rent/sqft ($)', 'Neg Rent Rate']
        fig_prof = go.Figure()
        for i, row in cl_df.iterrows():
            vals = [row.get('avg_leaseup',0), row.get('avg_rent_psf',0),
                    (row.get('pct_neg_rent',0) or 0)*100]
            fig_prof.add_trace(go.Bar(
                name=str(row['cluster']),
                x=profile_labels, y=vals,
                marker_color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            ))
        fig_prof.update_layout(
            plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font_color=FONT_COLOR,
            height=280, margin=dict(l=10,r=10,t=20,b=10),
            barmode='group', showlegend=True,
            xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=10)),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=10)),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TICK_COLOR,size=10))
        )
        st.plotly_chart(fig_prof, use_container_width=True)
    else:
        st.info("Run prepare_dashboard_data.py to generate embedding data.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: AI Anomaly Detection
# ════════════════════════════════════════════════════════════════════════════════
with tab_anomaly:
    st.markdown(
        '<div class="section-title">AI Anomaly Detection'
        '<span class="genai-badge">✦ GenAI · Isolation Forest</span></div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="info-box" style="margin:12px 0 20px">
    <strong>How it works:</strong> An <strong>Isolation Forest</strong> model was trained on the
    combined embedding space (numerical features + transformer embeddings). It identifies properties
    that are difficult to "isolate" — i.e., whose feature combination is unusual relative to the
    rest of the market. Anomaly score: higher = more anomalous. Contamination rate set to 10%.
    </div>
    """, unsafe_allow_html=True)

    if not pca_df.empty and 'anomaly_score' in pca_df.columns:
        col1, col2 = st.columns([3,1])
        with col1:
            pca_plot = pca_df.copy()
            pca_plot['Type'] = pca_plot['is_anomaly'].map(
                {True: '⚠ Anomaly', False: 'Normal'})
            pca_plot['color'] = pca_plot['is_anomaly'].map(
                {True: '#f85149', False: '#58a6ff'})
            pca_plot['size'] = pca_plot['anomaly_score'].fillna(0) * 15 + 6

            fig_ano = px.scatter(
                pca_plot, x='pca1', y='pca2',
                color='Type',
                color_discrete_map={'⚠ Anomaly':'#f85149','Normal':'rgba(88,166,255,0.4)'},
                size='size', size_max=18,
                hover_data=['name','submarket','lease_up_months','anomaly_score'],
                title=f'Isolation Forest Anomaly Detection · {stats.get("n_anomalies","?")} anomalies (10% contamination)'
            )
            fig_ano.update_layout(
                plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG,
                font_color=FONT_COLOR, height=400,
                margin=dict(l=10,r=10,t=50,b=10),
                xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=10)),
                yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=10)),
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color=TICK_COLOR,size=11))
            )
            st.plotly_chart(fig_ano, use_container_width=True)

        with col2:
            st.markdown(f"**{stats.get('n_anomalies','?')} Anomalies Detected**")
            st.caption("Properties with unusual feature combinations relative to market peers")

        if not ano_df.empty:
            st.markdown("**Top Anomalous Properties**")
            ano_disp = ano_df[['name','submarket','delivery_month',
                               'lease_up_months','rent_per_sqft',
                               'concession_intensity','anomaly_reason']].copy()
            ano_disp['concession_intensity'] = ano_disp['concession_intensity'].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else '—')
            ano_disp['rent_per_sqft'] = ano_disp['rent_per_sqft'].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) else '—')
            ano_disp.columns = ['Property','Submarket','Delivery','LU (mo)',
                                 'Rent/sqft','Concession','Why Anomalous']
            st.dataframe(ano_disp.head(20), use_container_width=True,
                         height=320, hide_index=True)
    else:
        st.info("Run prepare_dashboard_data.py to generate anomaly detection data.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: Lease-Up Predictor
# ════════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown(
        '<div class="section-title">Lease-Up Time Predictor'
        '<span class="genai-badge">✦ AI · Random Forest</span></div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="info-box" style="margin:12px 0 20px">
    <strong>How it works:</strong> A <strong>Random Forest Regressor</strong> was trained on
    historical lease-up data to predict how long a new property will take to reach 90% occupancy.
    Input the property characteristics below to get an AI-generated prediction with confidence context.
    </div>
    """, unsafe_allow_html=True)

    if model_bundle:
        fstats = model_info.get('feature_stats', {})

        def fval(feat, label, fmt='%.0f', step=1.0):
            fs = fstats.get(feat, {})
            mn, mx, mean = fs.get('min',0), fs.get('max',100), fs.get('mean',50)
            return st.number_input(label, min_value=float(mn), max_value=float(mx),
                                   value=float(mean), step=step, format=fmt)

        col1, col2, col3 = st.columns(3)
        with col1:
            qty = fval('quantity', '# of Units', step=10.0)
            area = fval('area_per_unit', 'Avg Unit Size (sqft)', step=50.0)
        with col2:
            rent_psf = fval('rent_per_sqft', 'Rent per sqft ($)', fmt='%.2f', step=0.05)
            avg_occ = fval('avg_occ_during_leaseup', 'Expected Avg Occ (0–1)',
                           fmt='%.2f', step=0.01)
        with col3:
            concession = fval('concession_intensity', 'Expected Concession Rate (0–1)',
                              fmt='%.3f', step=0.005)
            age = fval('property_age_at_delivery', 'Property Age at Delivery (yrs)',
                       step=1.0)

        if st.button("Predict Lease-Up Time", key='predict_btn'):
            X_input = np.array([[qty, area, rent_psf, avg_occ, concession, age]])
            pred = model_bundle['model'].predict(X_input)[0]
            pred_rounded = round(pred, 1)

            # Confidence context from tree variance
            preds_all = np.array([
                tree.predict(X_input)[0]
                for tree in model_bundle['model'].estimators_
            ])
            pred_std = preds_all.std()
            pred_lo = max(1, pred - 1.5*pred_std)
            pred_hi = pred + 1.5*pred_std

            mu_hist = stats['mu']
            delta = pred_rounded - mu_hist
            delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
            color = '#f85149' if delta > 3 else '#3fb950' if delta < -3 else '#f0883e'

            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #21262d;border-radius:12px;
                        padding:24px;margin-top:16px;text-align:center">
              <div style="color:#a8b3bc;font-size:.8rem;margin-bottom:6px">Predicted Lease-Up Time</div>
              <div style="font-family:'DM Serif Display';font-size:3.5rem;color:{color};line-height:1">
                {pred_rounded} <span style="font-size:1.5rem">months</span>
              </div>
              <div style="color:#a8b3bc;font-size:.82rem;margin-top:8px">
                95% range: {pred_lo:.1f} – {pred_hi:.1f} months &nbsp;·&nbsp;
                Market avg: {mu_hist} mo ({delta_str} vs avg)
              </div>
              <div style="color:#a8b3bc;font-size:.75rem;margin-top:6px">
                Model CV MAE: ±{model_info.get('cv_mae','?')} months &nbsp;·&nbsp;
                Trained on {model_info.get('n_train','?')} properties
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Feature importance chart
            st.markdown("<br>**Feature Importance (what drives predictions)**", unsafe_allow_html=True)
            imp = model_info.get('importances', {})
            if imp:
                imp_df = pd.DataFrame(list(imp.items()), columns=['Feature','Importance'])
                imp_df = imp_df.sort_values('Importance', ascending=True)
                fig_imp = go.Figure(go.Bar(
                    x=imp_df['Importance'], y=imp_df['Feature'],
                    orientation='h', marker_color='#f0883e', opacity=0.85
                ))
                fig_imp.update_layout(
                    plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font_color=FONT_COLOR,
                    height=220, margin=dict(l=10,r=10,t=20,b=10),
                    xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR, size=10)),
                    yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR, size=10))
                )
                st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("leaseup_model.pkl not found. Run prepare_dashboard_data.py first.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5: Similar Property Finder
# ════════════════════════════════════════════════════════════════════════════════
with tab_similar:
    st.markdown(
        '<div class="section-title">Similar Property Finder'
        '<span class="genai-badge">✦ GenAI · Embedding Similarity</span></div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="info-box" style="margin:12px 0 20px">
    <strong>How it works:</strong> Each property is represented as a vector in the transformer
    embedding space. Given a query property, we compute <strong>cosine similarity</strong> between
    its embedding and all others to find the most similar properties — capturing similarity in
    location, operator, size, rent level, and lease-up dynamics simultaneously.
    </div>
    """, unsafe_allow_html=True)

    if embeddings:
        all_names = sorted([
            m['name'] for m in embed_meta.values()
            if m.get('name')
        ])

        query_name = st.selectbox("Select a property to find similar ones:", all_names)
        n_results = st.slider("Number of similar properties to return:", 3, 15, 5)

        if st.button("Find Similar Properties", key='similar_btn'):
            # Find query property ID
            query_pid = next(
                (pid for pid, m in embed_meta.items() if m.get('name') == query_name),
                None
            )

            if query_pid and query_pid in embeddings:
                q_vec = np.array(embeddings[query_pid])

                # Compute cosine similarities
                sims = []
                for pid, vec in embeddings.items():
                    if pid == query_pid: continue
                    v = np.array(vec)
                    cos_sim = np.dot(q_vec, v) / (np.linalg.norm(q_vec) * np.linalg.norm(v) + 1e-10)
                    meta = embed_meta.get(pid, {})
                    sims.append({
                        'name': meta.get('name','—'),
                        'submarket': meta.get('submarket','—'),
                        'cluster': f"Cluster {meta.get('cluster','?')}",
                        'lease_up_months': meta.get('lease_up_months'),
                        'rent_per_sqft': meta.get('rent_per_sqft'),
                        'similarity': round(float(cos_sim), 4)
                    })

                sims_df = pd.DataFrame(sims).sort_values('similarity', ascending=False).head(n_results)

                # Show query property info
                q_meta = embed_meta.get(query_pid, {})
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #f0883e;border-radius:10px;
                            padding:16px;margin:12px 0">
                  <div style="color:#f0883e;font-size:.72rem;font-family:'DM Mono';margin-bottom:6px">QUERY PROPERTY</div>
                  <div style="font-size:1rem;font-weight:600">{query_name}</div>
                  <div style="color:#a8b3bc;font-size:.8rem;margin-top:4px">
                    {q_meta.get('submarket','—')} · {q_meta.get('cluster','?')} ·
                    LU: {q_meta.get('lease_up_months','—')} mo ·
                    Rent/sqft: ${q_meta.get('rent_per_sqft','—')}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"**Top {n_results} most similar properties:**")

                # Similarity bar chart
                fig_sim = go.Figure(go.Bar(
                    x=sims_df['similarity'],
                    y=sims_df['name'],
                    orientation='h',
                    marker_color=[
                        f'rgba(240,136,62,{0.4 + 0.6*s})'
                        for s in (sims_df['similarity'] - sims_df['similarity'].min()) /
                                  (sims_df['similarity'].max() - sims_df['similarity'].min() + 1e-10)
                    ],
                    text=sims_df['similarity'].apply(lambda x: f'{x:.3f}'),
                    textposition='outside',
                    textfont=dict(color=TICK_COLOR, size=10),
                    customdata=sims_df[['submarket','lease_up_months','rent_per_sqft']].values,
                    hovertemplate='<b>%{y}</b><br>Similarity: %{x:.4f}<br>'
                                  'Submarket: %{customdata[0]}<br>'
                                  'Lease-Up: %{customdata[1]} mo<br>'
                                  'Rent/sqft: $%{customdata[2]}<extra></extra>'
                ))
                fig_sim.update_layout(
                    plot_bgcolor=PLOT_BG, paper_bgcolor=PAPER_BG, font_color=FONT_COLOR,
                    height=max(280, n_results * 45), margin=dict(l=10,r=10,t=50,b=10),
                    title=dict(text=f'Cosine Similarity to "{query_name}"', font=dict(size=13, color=FONT_COLOR)),
                    xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=10), range=[0,1]),
                    yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR,size=10))
                )
                st.plotly_chart(fig_sim, use_container_width=True)

                # Table
                disp = sims_df.copy()
                disp['rent_per_sqft'] = disp['rent_per_sqft'].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else '—')
                disp['lease_up_months'] = disp['lease_up_months'].apply(
                    lambda x: f"{x:.0f} mo" if pd.notna(x) else '—')
                disp['similarity'] = disp['similarity'].apply(lambda x: f"{x:.4f}")
                disp.columns = ['Property','Submarket','Cluster','LU Time','Rent/sqft','Similarity']
                st.dataframe(disp, use_container_width=True,
                             height=min(400, (n_results+1)*45), hide_index=True)
            else:
                st.warning("Embedding not found for selected property.")
    else:
        st.info("Run prepare_dashboard_data.py to generate embedding data.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div style="display:flex;justify-content:space-between;font-size:.7rem;'
    'color:#a8b3bc;padding-bottom:20px;flex-wrap:wrap;gap:8px">'
    '<span>Affinius Capital DS Assessment 2026 &nbsp;·&nbsp; Yining (Evelyn) Huang, UT Austin PhD</span>'
    '<span>GenAI: sentence-transformers (all-MiniLM-L6-v2) · Isolation Forest · Random Forest · Cosine Similarity</span>'
    '</div>',
    unsafe_allow_html=True
)
