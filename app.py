import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import anthropic
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lease-Up Intelligence Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&family=DM+Mono&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #0d1117; }
.stApp { background: #0d1117; color: #e6edf3; }
.metric-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 20px 24px;
    border-top: 3px solid #f0883e;
}
.metric-label { font-size: .7rem; color: #7d8590; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 6px; }
.metric-value { font-family: 'DM Serif Display'; font-size: 2.2rem; line-height: 1; }
.metric-sub { font-size: .7rem; color: #7d8590; margin-top: 5px; }
.section-title { font-family: 'DM Serif Display'; font-size: 1.3rem; color: #e6edf3; margin-bottom: 4px; }
.ai-box {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 20px;
}
.ai-label {
    display: inline-block; background: rgba(240,136,62,.1);
    border: 1px solid rgba(240,136,62,.2); border-radius: 5px;
    padding: 2px 8px; font-size: .7rem; color: #f0883e;
    font-family: 'DM Mono'; margin-bottom: 10px;
}
div[data-testid="stMetric"] {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 16px 20px;
    border-top: 3px solid #f0883e;
}
div[data-testid="stMetric"] label { color: #7d8590 !important; font-size: .75rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #e6edf3 !important; }
.stSelectbox > div > div { background: #161b22 !important; border-color: #21262d !important; }
.stTextInput > div > div > input { background: #161b22 !important; border-color: #21262d !important; color: #e6edf3 !important; }
.stButton > button { background: #f0883e !important; color: #000 !important; border: none !important; font-weight: 600 !important; border-radius: 9px !important; }
.stButton > button:hover { opacity: .85 !important; }
hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open("dashboard_data.json") as f:
        d = json.load(f)
    df = pd.DataFrame(d['properties'])
    yr = pd.DataFrame(d['yr_summary'])
    seas = pd.DataFrame(d['season_summary'])
    cl = pd.DataFrame(d['cluster_summary'])
    anom = pd.DataFrame(d['anomalies'])
    stats = d['stats']
    return df, yr, seas, cl, anom, stats

df, yr_df, seas_df, cl_df, anom_df, stats = load_data()

CLUSTER_COLORS = {
    'Premium Fast': '#f0883e',
    'Mid-Market': '#58a6ff',
    'Luxury Challenged': '#f85149',
    'Affordable Slow': '#3fb950'
}
PLOT_THEME = dict(
    plot_bgcolor='#161b22',
    paper_bgcolor='#161b22',
    font_color='#e6edf3',
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#7d8590', family='DM Mono', size=11)),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#7d8590', family='DM Mono', size=11)),
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-bottom:1px solid #21262d;padding-bottom:18px;margin-bottom:28px">
  <div style="display:flex;align-items:center;gap:12px">
    <div style="width:40px;height:40px;background:#f0883e;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px">🏢</div>
    <div>
      <div style="font-family:'DM Serif Display';font-size:1.5rem;color:#e6edf3">Lease-Up Intelligence Dashboard</div>
      <div style="font-size:.75rem;color:#7d8590">Property Analytics · Affinius Capital DS Assessment 2026 · Yining (Evelyn) Huang</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Properties Delivered", stats['total'], help="Since April 2008 · 2 markets")
with k2:
    st.metric("Avg Lease-Up Time", f"{stats['avg_leaseup']} mo", help="Months to reach 90% occupancy")
with k3:
    st.metric("Negative Rent Growth", f"{stats['pct_neg_rent']}%", help="Properties where effective rent declined during lease-up")
with k4:
    st.metric("Austin / Akron", f"{stats['austin_count']} / {stats['akron_count']}", help="Deliveries by market")

st.markdown("<br>", unsafe_allow_html=True)

# ── Trend Charts ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Market Trends</div>', unsafe_allow_html=True)
st.markdown('<span style="font-size:.75rem;color:#7d8590">Temporal Analysis</span>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=yr_df['delivery_year'], y=yr_df['count'],
        name='Property Count', marker_color='rgba(88,166,255,0.2)',
        marker_line_color='rgba(88,166,255,0.5)', marker_line_width=1
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=yr_df['delivery_year'], y=yr_df['avg_leaseup'].round(1),
        name='Avg Lease-Up (mo)', line=dict(color='#f0883e', width=2.5),
        mode='lines+markers', marker=dict(size=6, color='#f0883e'),
        fill='tozeroy', fillcolor='rgba(240,136,62,0.08)'
    ), secondary_y=False)
    fig.update_layout(
        title=dict(text='Average Lease-Up Time by Delivery Year', font=dict(size=13, color='#e6edf3')),
        **PLOT_THEME, height=300, showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#7d8590', size=11)),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.update_yaxes(title_text="Avg Months", secondary_y=False, title_font=dict(color='#7d8590', size=11))
    fig.update_yaxes(title_text="# Properties", secondary_y=True, title_font=dict(color='#7d8590', size=11), gridcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    neg_pct = (yr_df['pct_neg_rent'] * 100).round(1)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=yr_df['delivery_year'], y=neg_pct,
        line=dict(color='#f85149', width=2.5),
        mode='lines+markers', marker=dict(size=6, color='#f85149'),
        fill='tozeroy', fillcolor='rgba(248,81,73,0.1)', name='Neg Rent %'
    ))
    fig2.update_layout(
        title=dict(text='Negative Rent Growth Rate', font=dict(size=13, color='#e6edf3')),
        **PLOT_THEME, height=300, showlegend=False,
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#7d8590', family='DM Mono', size=11), ticksuffix='%', range=[0, 100]),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Segmentation Charts ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">Property Segmentation</div>', unsafe_allow_html=True)
st.markdown('<span style="font-size:.75rem;color:#7d8590">Cluster Analysis</span>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    fig3 = go.Figure(go.Pie(
        labels=cl_df['cluster'],
        values=cl_df['count'],
        hole=0.62,
        marker=dict(colors=[CLUSTER_COLORS.get(c, '#7d8590') for c in cl_df['cluster']],
                    line=dict(color='#0d1117', width=3)),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Avg LU: %{customdata:.1f} mo<extra></extra>',
        customdata=cl_df['avg_leaseup']
    ))
    fig3.update_layout(
        title=dict(text='Property Clusters', font=dict(size=13, color='#e6edf3')),
        paper_bgcolor='#161b22', plot_bgcolor='#161b22',
        font_color='#e6edf3', height=300,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#7d8590', size=10)),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

with c2:
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']
    season_colors = {'Spring': '#3fb950', 'Summer': '#f0883e', 'Fall': '#f85149', 'Winter': '#58a6ff'}
    seas_sorted = seas_df.set_index('delivery_season').reindex(season_order).reset_index()
    fig4 = go.Figure(go.Bar(
        x=seas_sorted['delivery_season'],
        y=seas_sorted['avg_leaseup'].round(1),
        marker_color=[season_colors[s] for s in seas_sorted['delivery_season']],
        text=seas_sorted['avg_leaseup'].round(1),
        textposition='outside',
        textfont=dict(color='#7d8590', size=11)
    ))
    fig4.update_layout(
        title=dict(text='Delivery Season Effect', font=dict(size=13, color='#e6edf3')),
        **PLOT_THEME, height=300, showlegend=False,
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#7d8590', family='DM Mono', size=11), title=dict(text='Avg Months', font=dict(color='#7d8590', size=11))),
        margin=dict(l=0, r=0, t=40, b=20)
    )
    st.plotly_chart(fig4, use_container_width=True)

with c3:
    lu_vals = df['lease_up_months'].dropna()
    fig5 = go.Figure(go.Histogram(
        x=lu_vals, nbinsx=20,
        marker_color='rgba(88,166,255,0.5)',
        marker_line_color='#58a6ff', marker_line_width=1
    ))
    fig5.add_vline(x=lu_vals.mean(), line_dash='dash', line_color='#f0883e',
                   annotation_text=f'Mean: {lu_vals.mean():.1f}mo', annotation_font_color='#f0883e')
    fig5.update_layout(
        title=dict(text='Lease-Up Distribution', font=dict(size=13, color='#e6edf3')),
        **PLOT_THEME, height=300, showlegend=False,
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#7d8590', family='DM Mono', size=11), title=dict(text='Months', font=dict(color='#7d8590', size=11))),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig5, use_container_width=True)

# ── Property Explorer ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Property Explorer</div>', unsafe_allow_html=True)
st.markdown('<span style="font-size:.75rem;color:#7d8590">All Delivered Properties</span>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns(4)
with f1:
    filt_msa = st.selectbox("Market", ["All"] + sorted(df['msa'].unique().tolist()), key='fmsa')
with f2:
    filt_cl = st.selectbox("Cluster", ["All"] + sorted(df['cluster'].unique().tolist()), key='fcl')
with f3:
    filt_rent = st.selectbox("Rent Growth", ["All", "Negative Only", "Positive Only"], key='frent')
with f4:
    filt_sea = st.selectbox("Season", ["All", "Spring", "Summer", "Fall", "Winter"], key='fsea')

fdf = df.copy()
if filt_msa != "All": fdf = fdf[fdf['msa'] == filt_msa]
if filt_cl != "All": fdf = fdf[fdf['cluster'] == filt_cl]
if filt_rent == "Negative Only": fdf = fdf[fdf['neg_rent_growth'] == True]
if filt_rent == "Positive Only": fdf = fdf[fdf['neg_rent_growth'] == False]
if filt_sea != "All": fdf = fdf[fdf['delivery_season'] == filt_sea]

st.caption(f"{len(fdf)} properties")

disp = fdf[['name','submarket','delivery_month','quantity','lease_up_months','rent_at_delivery','rent_change_pct','cluster']].copy()
disp.columns = ['Property Name','Submarket','Delivery','Units','Lease-Up (mo)','Rent at Del. ($)','Rent Change (%)','Cluster']
disp['Rent at Del. ($)'] = disp['Rent at Del. ($)'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
disp['Rent Change (%)'] = disp['Rent Change (%)'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
disp['Lease-Up (mo)'] = disp['Lease-Up (mo)'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
disp['Units'] = disp['Units'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
st.dataframe(disp.head(200), use_container_width=True, height=320, hide_index=True)

# ── AI Section ────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">✦ AI-Powered Analysis <span style="font-size:.7rem;color:#f0883e;font-family:DM Mono;background:rgba(240,136,62,.1);border:1px solid rgba(240,136,62,.3);border-radius:5px;padding:2px 8px;margin-left:8px">GenAI · Claude API</span></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

def get_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", ""))
    if not api_key:
        st.error("ANTHROPIC_API_KEY not found. Add it to Streamlit Secrets.")
        return None
    return anthropic.Anthropic(api_key=api_key)

def build_context():
    cl_txt = "; ".join([
        f"{r['cluster']}: {r['count']} props, avg {r['avg_leaseup']:.1f} mo, {r['pct_neg_rent']*100:.0f}% neg rent"
        for _, r in cl_df.iterrows()
    ])
    yr_txt = "; ".join([f"{int(r['delivery_year'])}: {r['count']} props, avg {r['avg_leaseup']:.1f} mo" for _, r in yr_df.iterrows()])
    seas_txt = "; ".join([f"{r['delivery_season']}: avg {r['avg_leaseup']:.1f} mo" for _, r in seas_df.iterrows()])
    return (
        f"You are a real estate data analyst. Data from two US apartment markets.\n"
        f"MARKETS: Austin-Round Rock TX ({stats['austin_count']} props) and Akron OH ({stats['akron_count']} props).\n"
        f"OVERALL: {stats['total']} delivered since Apr-2008. Avg lease-up: {stats['avg_leaseup']} months (sd={stats['sd']}). {stats['pct_neg_rent']}% had negative rent growth.\n"
        f"CLUSTERS: {cl_txt}\n"
        f"YEAR TREND: {yr_txt}\n"
        f"SEASONALITY: {seas_txt}\n"
        f"DEFINITION: Lease-up time = months from first LU/UC-LU status to first month with occupancy >= 90%."
    )

def call_claude(system, user):
    client = get_client()
    if not client:
        return None
    with st.spinner("Thinking..."):
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
    return msg.content[0].text

tab1, tab2, tab3 = st.tabs(["💬 Natural Language Query", "✦ Auto Insight Generation", "⚠ Anomaly Explanation"])

with tab1:
    st.markdown('<p style="font-size:.8rem;color:#7d8590">Ask any question about the lease-up data. The AI will return data-backed insights.</p>', unsafe_allow_html=True)
    chips = [
        "Which submarket has the fastest lease-up?",
        "What % of properties had negative rent growth in 2015?",
        "Compare Summer vs Winter delivery performance",
        "Top 5 slowest lease-up properties",
        "Which cluster has highest concession intensity?",
    ]
    cols = st.columns(len(chips))
    for i, chip in enumerate(chips):
        if cols[i].button(chip, key=f"chip_{i}", use_container_width=True):
            st.session_state['nl_query'] = chip

    query = st.text_input("Ask a question", value=st.session_state.get('nl_query', ''), placeholder="e.g. What drove the spike in lease-up time in 2016?", key="nl_input")
    if st.button("Ask AI", key="nl_btn"):
        if query:
            answer = call_claude(build_context(), f'Answer concisely (3-5 sentences, cite numbers): "{query}"')
            if answer:
                st.markdown(f'<div class="ai-box"><div class="ai-label">✦ AI Response</div><div style="font-size:.85rem;line-height:1.75">{answer}</div></div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<p style="font-size:.8rem;color:#7d8590">Select a view. The AI will generate a structured summary of key patterns, risks, and opportunities.</p>', unsafe_allow_html=True)
    view = st.selectbox("Analysis view", [
        "Overall Market Summary",
        "Cluster Performance Analysis",
        "Year-over-Year Trend Analysis",
        "Rent Growth Risk Analysis"
    ], key="ins_view")
    prompts = {
        "Overall Market Summary": "Generate a concise executive summary (5-7 sentences) of overall lease-up performance. Cover: market size, average performance, key risk signals, and one strategic recommendation.",
        "Cluster Performance Analysis": "Analyze the 4 property clusters (Premium Fast, Mid-Market, Luxury Challenged, Affordable Slow). For each: performance, risk, and what drives properties there. End with which cluster offers best risk-adjusted return.",
        "Year-over-Year Trend Analysis": "Analyze year-over-year trends in lease-up time and negative rent growth 2008-2020. Identify peak stress periods, likely macro drivers (GFC, supply cycles), and trajectory into 2020.",
        "Rent Growth Risk Analysis": "Perform a rent growth risk analysis. Which years, seasons, and clusters show highest negative rent growth? What does this signal for underwriting assumptions?"
    }
    if st.button("Generate Insight", key="ins_btn"):
        answer = call_claude(build_context(), prompts[view])
        if answer:
            st.markdown(f'<div class="ai-box"><div class="ai-label">✦ AI Response</div><div style="font-size:.85rem;line-height:1.75">{answer}</div></div>', unsafe_allow_html=True)

with tab3:
    st.markdown(f'<p style="font-size:.8rem;color:#7d8590"><strong style="color:#f85149">{len(anom_df)} properties</strong> with anomalous lease-up times (|z-score| > 1.5) detected.</p>', unsafe_allow_html=True)
    anom_labels = [
        f"[z={row['leaseup_zscore']:+.1f}] {row['name']} — {row['lease_up_months']} mo ({row['delivery_month']})"
        for _, row in anom_df.iterrows()
    ]
    selected = st.selectbox("Select property", anom_labels, key="anom_sel")
    if st.button("Explain Anomaly", key="anom_btn"):
        idx = anom_labels.index(selected)
        p = anom_df.iloc[idx]
        desc = (
            f"Property: \"{p['name']}\", Submarket: {p['submarket']}, Delivery: {p['delivery_month']}, "
            f"Units: {p['quantity']}, Lease-up: {p['lease_up_months']} months (z={p['leaseup_zscore']}), "
            f"Rent at delivery: ${p['rent_at_delivery']:.0f if pd.notna(p['rent_at_delivery']) else 'N/A'}, "
            f"Rent change: {str(p['rent_change_pct'])+'%' if pd.notna(p['rent_change_pct']) else 'N/A'}, "
            f"Cluster: {p['cluster']}, Season: {p['delivery_season']}."
        )
        direction = "significantly above" if p['leaseup_zscore'] > 0 else "significantly below"
        answer = call_claude(
            build_context() + " Be specific, reference delivery year context.",
            f"This property has anomalous lease-up (z-score {direction} average).\n{desc}\n"
            "Explain: (1) what makes it anomalous, (2) likely drivers, (3) what investor should have monitored, (4) one underwriting lesson."
        )
        if answer:
            st.markdown(f'<div class="ai-box"><div class="ai-label">✦ AI Response</div><div style="font-size:.85rem;line-height:1.75">{answer}</div></div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div style="display:flex;justify-content:space-between;font-size:.7rem;color:#7d8590;padding-bottom:20px;flex-wrap:wrap;gap:8px">'
    '<span>Affinius Capital DS Assessment 2026 &nbsp;·&nbsp; Yining (Evelyn) Huang, UT Austin PhD Candidate</span>'
    '<span>GenAI: Claude API (claude-sonnet-4-20250514) &nbsp;·&nbsp; Charts: Plotly &nbsp;·&nbsp; Data: CoStar MSA panels</span>'
    '</div>',
    unsafe_allow_html=True
)
