import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

st.markdown("""
<style>
.stButton>button {
    background-color: #1E88E5;
    color: white;
    border-radius: 8px;
}
.stDateInput input {
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)


st.set_page_config(
    page_title="Analisis Sentimen Tokoh Publik Purbaya di Media Sosial X",
    page_icon=":chart:",
    layout="wide",  
)

st.header('Analisis Sentimen Tokoh Publik Purbaya di Media Sosial X')
st.markdown("""---""")

data = pd.read_excel('labeled_tweets_2025-09-08_to_2025-12-30.xlsx')

# mengubah nilai kolom dan menghapus sentimen yang kosong
mapping = {1: 'Positive', 2: 'Negative'}
df = data.dropna(subset=['Label'])
df = df.loc[df['Label'] != 3]
df['Label'] = df['Label'].map(mapping)

# mengurutkan nomer index
df = df.reset_index(drop=True)
df.index = df.index + 1

# menghapus data duplikat
df = df.drop_duplicates(subset=['CleanSVM'])

nav1, nav2 = st.columns(2)
with nav1:
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    start = df['Date'].min()
    finish = df['Date'].max()
    
    date_range = st.date_input(
        'Range Time',
        value=(start, finish),
        min_value=start,
        max_value=finish,
        format="DD/MM/YYYY"
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = start
        end_date = finish
    
with nav2:
    jenis_sentimen = st.multiselect("Category", options = df["Label"].unique(), default = df["Label"].unique())

# filter Tgl
output = (df['Date'] >= start_date) & (df['Date'] <= end_date)

# filter sentiment
df_selection1 = df.loc[output]

# filter sentiment
df_selection = df.query("Label == @jenis_sentimen").loc[output]

# Insert containers separated into tabs:
tab1, tab2 = st.tabs(["Data", "Summary"])

if "page" not in st.session_state:
    st.session_state.page = 1

if "rows_per_page" not in st.session_state:
    st.session_state.rows_per_page = 10

if "nav_action" not in st.session_state:
    st.session_state.nav_action = None
    
if "last_rows_per_page" not in st.session_state:
    st.session_state.last_rows_per_page = st.session_state.rows_per_page

if st.session_state.rows_per_page != st.session_state.last_rows_per_page:
    st.session_state.page = 1
    st.session_state.last_rows_per_page = st.session_state.rows_per_page
    st.rerun()

with tab1:
    rows_per_page = st.session_state.rows_per_page
    total_rows = len(df_selection)
    total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)

    if st.session_state.nav_action == "first":
        st.session_state.page = 1
    elif st.session_state.nav_action == "prev":
        st.session_state.page = max(1, st.session_state.page - 1)
    elif st.session_state.nav_action == "next":
        st.session_state.page = min(total_pages, st.session_state.page + 1)
    elif st.session_state.nav_action == "last":
        st.session_state.page = total_pages
    
    st.session_state.nav_action = None  # reset
    
    st.session_state.page = max(1, min(st.session_state.page, total_pages))
    page = st.session_state.page  # ✅ re-read AFTER update

    if st.session_state.page > total_pages:
        st.session_state.page = total_pages
        
    start_idx = (page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    
    df_page = df_selection.iloc[start_idx:end_idx].copy()
    df_page.reset_index(drop=True, inplace=True)
    df_page.index = df_page.index + start_idx + 1
    
    st.dataframe(
        df_page,
        use_container_width=True,
        height=400
    )

    col0, col1a, col1b, col2a, col2b, col3, col4, col5, col6, col7 = st.columns([2,1,0.8,0.5,0.8,0.5,0.5,1.5,0.5,0.5])
    with col0:
        st.markdown(f"**Display {start_idx + 1}–{min(end_idx, total_rows)} of {total_rows} data**")
    with col1a:
        st.markdown("**Rows per page**")
    with col1b:
        new_rows_per_page = st.selectbox(
            "",
            [5, 10, 25, 50],
            index=[5, 10, 25, 50].index(st.session_state.rows_per_page),
            key="rows_per_page_select",
            label_visibility="collapsed"
        )
        
        if new_rows_per_page != st.session_state.rows_per_page:
            st.session_state.rows_per_page = new_rows_per_page
            st.session_state.page = 1
            st.rerun()

    with col2a:
        st.markdown("**Page**")
    with col2b:
        st.selectbox(
            "",
            options=list(range(1, total_pages + 1)),
            key="page",
            label_visibility="collapsed"
        )
    with col3:
        if st.button("⏮", disabled=page == 1):
            st.session_state.nav_action = "first"
            st.rerun()
    with col4:
        if st.button("◀", disabled=page == 1):
            st.session_state.nav_action = "prev"
            st.rerun()
    with col5:
        st.markdown(
            f"<h5 style='text-align:center'>Page {st.session_state.page} of {total_pages}</h5>",
            unsafe_allow_html=True
        )
    with col6:
        if st.button("▶", disabled=page == total_pages):
            st.session_state.nav_action = "next"
            st.rerun()
    with col7:
        if st.button("⏭", disabled=page == total_pages):
            st.session_state.nav_action = "last"
            st.rerun()
            
    st.markdown("""---""")
with tab2:
    pos = df_selection1['Label'].loc[df_selection1['Label'] == 'Positive']
    neg = df_selection1['Label'].loc[df_selection1['Label'] == 'Negative']
    count = len(df_selection1)
    
    b1, b2, b3 = st.columns([0.45,0.45,0.45])
    b1.metric("", len(pos), "+ Positive")
    b2.metric("", len(neg), "- Negative")
    b3.metric("Jumlah", count)

    # garis 
    st.markdown("""---""")

nav3, nav4 = st.columns(2)
with nav3:
    # Visualisasi hasil sentiment
    sentiment_counts = df_selection1['Label'].value_counts()

    color_map = {
        "Positive": "#2E7D32",  # deep green
        "Negative": "#C62828"   # deep red
    }
    
    fig_sentiment = go.Figure(
        data=[
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.45,
                marker=dict(
                    colors=[color_map[label] for label in sentiment_counts.index],
                    line=dict(color="#FFFFFF", width=2)
                ),
                textinfo="percent",
                hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>Persentase: %{percent}<extra></extra>"
            )
        ]
    )
    
    fig_sentiment.update_layout(
        title={
            "text": "Tweet Category",
            "x": 0.5,
            "xanchor": "center",
            "font": dict(size=18)
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=60, b=40, l=40, r=40),
        height=380
    )
    
    st.plotly_chart(fig_sentiment, use_container_width=True)

with nav4:
    tgl_counts = df_selection1['Date'].value_counts().reset_index()
    tgl_counts.columns = ['Date', 'Count']
    
    # Urutkan berdasarkan tanggal (PENTING supaya tidak acak)
    tgl_counts['Date'] = pd.to_datetime(tgl_counts['Date'])
    tgl_counts = tgl_counts.sort_values('Date')
    
    custom_colors = ['#1E88E5']  # professional blue
    
    fig_tgl = px.bar(
        tgl_counts,
        x='Date',
        y='Count',
        title="Tweet Range",
        color_discrete_sequence=custom_colors
    )
    
    fig_tgl.update_layout(
        xaxis_title="Tanggal",
        yaxis_title="Jumlah Tweet",
        title_x=0.5,
        bargap=0.25,
        height=380,
        margin=dict(t=60, b=40, l=40, r=40)
    )
    
    fig_tgl.update_traces(
        hovertemplate="Tanggal: %{x|%d %b %Y}<br>Jumlah Tweet: %{y}<extra></extra>"
    )
    
    st.plotly_chart(fig_tgl, use_container_width=True)

st.markdown("""---""")

# Prepare data
X = df['CleanSVM']
y = df['Label']

# Define KFold cross-validation with fixed 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Model definition
model = LinearSVC()

# Splitting data once for all folds
splits = list(kf.split(X))

# Function to print and plot metrics
def print_metrics(y_test, y_pred, fold, title_suffix):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy:.3f}")
    m2.metric("Precision", f"{precision:.3f}")
    m3.metric("Recall", f"{recall:.3f}")
    m4.metric("F1-Score", f"{f1:.3f}")
    
    with st.expander("Lihat Classification Report"):
        st.text(classification_report(y_test, y_pred))
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Score": [accuracy, precision, recall, f1]
        })
        
        fig = px.bar(
            metrics_df,
            x="Metric",
            y="Score",
            title="Evaluasi Kinerja",
            text="Score",
            color="Metric"
        )
        fig.update_layout(yaxis_range=[0,1])
        st.plotly_chart(fig, use_container_width=True)

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Fold {fold_choice}")
    st.pyplot(plt)

# Tabs for TF, TF-IDF, IndoBERTweet
tab1, tab2, tab3 = st.tabs(["TF", "TF-IDF", "IndoBERTweet"])

with tab1:
    countVectorizer = CountVectorizer()
    tf = countVectorizer.fit_transform(X).toarray()

    fold_choice = st.selectbox("Pilih K-fold TF", [1, 2, 3, 4, 5])
    
    fold = fold_choice
    train_index, test_index = splits[fold - 1]
    X_train, X_test = tf[train_index], tf[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print_metrics(y_test, y_pred, fold, "Count Vectorizer (TF)")

with tab2:
    tfidfVectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
    tfidf = tfidfVectorizer.fit_transform(X).toarray()

    fold_choice = st.selectbox("Pilih K-fold TF-IDF", [1, 2, 3, 4, 5])
    
    fold = fold_choice
    train_index, test_index = splits[fold - 1]
    X_train, X_test = tfidf[train_index], tfidf[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print_metrics(y_test, y_pred, fold, "TF-IDF")

with tab3:
    fold_choice = st.selectbox("Pilih K-fold IndoBERTweet", [1, 2, 3, 4, 5])

    # Load hasil fold
    file_path = f"results/indobertweet_fold_{fold_choice}_predictions.csv"
    df_fold = pd.read_csv(file_path)

    y_test = df_fold["y_true"]
    y_pred = df_fold["y_pred"]

    print_metrics(
        y_test,
        y_pred,
        fold_choice,
        "IndoBERTweet"
    )

