import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

st.set_page_config(
    page_title="Segmentasi Tamu Hotel",
    layout="wide"
)

# ── Load dan proses data ──────────────────────────────────────────
@st.cache_data
def load_and_process():
    df_raw = pd.read_csv("hotel_bookings.csv")
    df_raw["total_stays"] = (
        df_raw["stays_in_weekend_nights"] + df_raw["stays_in_week_nights"]
    )

    cols_focus = [
        "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "total_stays", "adults", "children", "meal", "market_segment",
        "distribution_channel", "is_repeated_guest", "previous_cancellations",
        "reserved_room_type", "booking_changes", "deposit_type",
        "days_in_waiting_list", "customer_type", "adr",
        "total_of_special_requests"
    ]

    df = df_raw[cols_focus].copy()

    cat_cols = df.select_dtypes(include="object").columns.tolist()

    df = df.drop_duplicates()
    df["children"] = df["children"].fillna(df["children"].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    cols_iqr = ["lead_time", "adr", "days_in_waiting_list",
                "previous_cancellations", "booking_changes"]
    for col in cols_iqr:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    df_clean = df.copy()

    le         = LabelEncoder()
    df_encoded = df_clean.copy()
    for col in cat_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    scaler       = MinMaxScaler()
    num_cols_all = df_encoded.select_dtypes(include="number").columns.tolist()
    df_scaled    = df_encoded.copy()
    df_scaled[num_cols_all] = scaler.fit_transform(df_encoded[num_cols_all])

    K      = 3
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)

    df_clean["cluster"]  = labels
    df_scaled["cluster"] = labels

    return df_clean, cat_cols, K

df_clean, cat_cols, K_OPTIMAL = load_and_process()

COLORS = ["steelblue", "coral", "mediumseagreen", "mediumpurple", "goldenrod"]

LABEL_CLUSTER = {
    0: "Tamu Reguler",
    1: "Tamu Premium",
    2: "Tamu Budget"
}

# ── Header ────────────────────────────────────────────────────────
st.title("Segmentasi Tamu Hotel")
st.write(
    "Dashboard ini menampilkan karakteristik tamu hotel berdasarkan "
    "hasil pengelompokan menggunakan algoritma K-Means Clustering."
)
st.markdown("---")

# ── Ringkasan jumlah tamu ─────────────────────────────────────────
st.subheader("Ringkasan Segmentasi")

cols_metric = st.columns(K_OPTIMAL)
for i in range(K_OPTIMAL):
    jumlah = (df_clean["cluster"] == i).sum()
    persen = jumlah / len(df_clean) * 100
    cols_metric[i].metric(
        label=f"Cluster {i} - {LABEL_CLUSTER.get(i, '')}",
        value=f"{jumlah:,} tamu",
        delta=f"{persen:.1f}% dari total"
    )

st.markdown("---")

# ── Pilih cluster ─────────────────────────────────────────────────
st.subheader("Detail Karakteristik per Cluster")

pilihan = st.selectbox(
    "Pilih cluster yang ingin dilihat:",
    options=[f"Cluster {i} - {LABEL_CLUSTER.get(i, '')}" for i in range(K_OPTIMAL)]
)

idx_cluster = int(pilihan.split(" ")[1])
df_cluster  = df_clean[df_clean["cluster"] == idx_cluster]

# ── Info umum cluster terpilih ────────────────────────────────────
st.markdown(f"### Cluster {idx_cluster} - {LABEL_CLUSTER.get(idx_cluster, '')}")

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Jumlah Tamu",         f"{len(df_cluster):,}")
col_b.metric("Rata-rata Harga/Malam (adr)", f"{df_cluster['adr'].mean():.2f}")
col_c.metric("Rata-rata Lama Menginap",     f"{df_cluster['total_stays'].mean():.1f} malam")
col_d.metric("Rata-rata Lead Time",         f"{df_cluster['lead_time'].mean():.0f} hari")

st.markdown("---")

# ── Statistik deskriptif cluster ─────────────────────────────────
st.subheader("Statistik Deskriptif")
num_cols_show = ["lead_time", "total_stays", "adr", "adults", "children",
                 "booking_changes", "total_of_special_requests",
                 "previous_cancellations", "days_in_waiting_list"]
st.dataframe(
    df_cluster[num_cols_show].describe().round(2),
    use_container_width=True
)

st.markdown("---")

# ── Perbandingan rata-rata antar cluster ──────────────────────────
st.subheader("Perbandingan Rata-rata Antar Cluster")

profil_cols = ["lead_time", "total_stays", "adr",
               "total_of_special_requests", "booking_changes", "adults"]
profil      = df_clean.groupby("cluster")[profil_cols].mean().round(2)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes      = axes.flatten()

for i, col in enumerate(profil_cols):
    bar_colors = [
        COLORS[c] if c != idx_cluster else "black"
        for c in profil.index
    ]
    bars = axes[i].bar(
        [f"Cluster {c}" for c in profil.index],
        profil[col],
        color=bar_colors,
        edgecolor="white"
    )
    axes[i].set_title(col, fontsize=10, fontweight="bold")
    axes[i].set_ylabel("Rata-rata", fontsize=8)
    axes[i].tick_params(labelsize=8)

plt.suptitle(
    f"Perbandingan Rata-rata per Cluster (Cluster {idx_cluster} = hitam)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── Distribusi variabel numerik cluster terpilih ──────────────────
st.subheader(f"Distribusi Variabel Numerik - Cluster {idx_cluster}")

n_cols_plot = 3
n_rows_plot = -(-len(num_cols_show) // n_cols_plot)

fig2, axes2 = plt.subplots(n_rows_plot, n_cols_plot,
                            figsize=(16, n_rows_plot * 4))
axes2 = axes2.flatten()

for i, col in enumerate(num_cols_show):
    axes2[i].hist(df_cluster[col].dropna(), bins=25,
                  color=COLORS[idx_cluster], edgecolor="white", alpha=0.85)
    axes2[i].set_title(col, fontsize=10, fontweight="bold")
    axes2[i].set_xlabel("Nilai", fontsize=8)
    axes2[i].set_ylabel("Frekuensi", fontsize=8)
    axes2[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes2)):
    fig2.delaxes(axes2[j])

plt.suptitle(
    f"Distribusi Variabel Numerik - Cluster {idx_cluster}",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("---")

# ── Distribusi variabel kategorikal cluster terpilih ─────────────
st.subheader(f"Distribusi Variabel Kategorikal - Cluster {idx_cluster}")

cat_show    = ["meal", "market_segment", "customer_type",
               "deposit_type", "reserved_room_type"]
n_cat_cols  = 3
n_cat_rows  = -(-len(cat_show) // n_cat_cols)

fig3, axes3 = plt.subplots(n_cat_rows, n_cat_cols,
                            figsize=(16, n_cat_rows * 4))
axes3 = axes3.flatten()

for i, col in enumerate(cat_show):
    counts = df_cluster[col].value_counts()
    axes3[i].bar(counts.index, counts.values,
                 color=COLORS[idx_cluster], edgecolor="white", alpha=0.85)
    axes3[i].set_title(col, fontsize=10, fontweight="bold")
    axes3[i].set_ylabel("Jumlah", fontsize=8)
    axes3[i].tick_params(axis="x", rotation=30, labelsize=7)
    axes3[i].tick_params(axis="y", labelsize=7)

for j in range(i + 1, len(axes3)):
    fig3.delaxes(axes3[j])

plt.suptitle(
    f"Distribusi Variabel Kategorikal - Cluster {idx_cluster}",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
st.pyplot(fig3)
plt.close()

st.markdown("---")

# ── Heatmap profil semua cluster ──────────────────────────────────
st.subheader("Heatmap Profil Semua Cluster")

profil_norm = profil.copy().astype(float)
for col in profil_norm.columns:
    mn, mx = profil_norm[col].min(), profil_norm[col].max()
    if mx != mn:
        profil_norm[col] = (profil_norm[col] - mn) / (mx - mn)

fig4, ax4 = plt.subplots(figsize=(11, 4))
sns.heatmap(
    profil_norm,
    annot=profil,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    annot_kws={"size": 9},
    ax=ax4
)
ax4.set_title(
    "Heatmap Profil Cluster (Nilai Ternormalisasi, Anotasi Nilai Asli)",
    fontsize=11, fontweight="bold"
)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
st.pyplot(fig4)
plt.close()
