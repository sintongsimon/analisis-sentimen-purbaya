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
    page_title="Analisis Sentimen Tokoh Publik Purbaya",
    page_icon=":chart:",
    layout="wide",  # Use "wide" layout for a full-size dashboard
)

st.header('Analisis Sentimen Tokoh Publik Purbaya')
st.markdown("""---""")

data = pd.read_excel('labeled_tweets_merged_2025-09-08_to_2025-12-31.csv')

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
    df['created_at'] = pd.to_datetime(df['created_at']).dt.date
    start = df['created_at'].min()
    finish = df['created_at'].max()
    start_date, end_date = st.date_input('Range Time', (start, finish), start, finish, format="DD/MM/YYYY")
with nav2:
    jenis_sentimen = st.multiselect("Category", options = df["Label"].unique(), default = df["Label"].unique())

# filter Tgl
output = (df['created_at'] >= start_date) & (df['created_at'] <= end_date)

# filter sumber, tamggal dan sentiment
df_selection = df.query("Label == @jenis_sentimen").loc[output]

# Insert containers separated into tabs:
tab1, tab2 = st.tabs(["Data", "Summary"])
with tab1:
    df_selection
with tab2:
    pos = df_selection['Label'].loc[df_selection['Label'] == 'Positif']
    neg = df_selection['Label'].loc[df_selection['Label'] == 'Negatif']
    count = len(df_selection)
    
    b1, b2, b3 = st.columns([0.45,0.45,0.45])
    b1.metric("Jumlah Komentar", len(pos), "+ Positif")
    b2.metric("Jumlah Komentar", len(neg), "- Negatif")
    b3.metric("Jumlah", count)

    # garis 
    st.markdown("""---""")

nav3, nav4 = st.columns(2)
with nav3:
    # Visualisasi hasil sentiment
    color_custom = ['#e14b32', '#3ca9ee']
    Sentimen = df_selection['Label'].value_counts()
    fig_sentiment = go.Figure()

    neg_df = df_selection[df_selection['Label'] == 'Negatif']
    pos_df = df_selection[df_selection['Label'] == 'Positif']
        
    if not neg_df.empty:
        color = ['#e14b32']
        fig_sentiment.add_trace(go.Pie(labels=['Negatif'], values=neg_df['Label'].value_counts(), 
                                        marker_colors=color, textinfo='label+percent', 
                                        hoverinfo='label+value', hole=0.3))
    if not pos_df.empty:
        color = ['#3ca9ee']
        fig_sentiment.add_trace(go.Pie(labels=['Positif'], values=pos_df['Label'].value_counts(), 
                                        marker_colors=color, textinfo='label+percent', 
                                        hoverinfo='value', hole=0.3))
    if not neg_df.empty and not pos_df.empty:
        fig_sentiment.add_trace(go.Pie(labels=['Negatif','Positif'], values=Sentimen,
                                      marker_colors=color_custom, textinfo='label+percent',
                                      hoverinfo='value', hole=0.3))
        
    fig_sentiment.update_layout(title="Persentase Sentimen Twitter")
    st.plotly_chart(fig_sentiment, use_container_width=True)


with nav4:
    tgl_counts = df_selection['created_at'].value_counts().reset_index()
    tgl_counts.columns = ['created_at', 'Count']
    custom_colors = ['#dc6e55']
    fig_tgl = px.area(tgl_counts, x='created_at', y='Count', title="Rentang Waktu Komentar", color_discrete_sequence=custom_colors)
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

