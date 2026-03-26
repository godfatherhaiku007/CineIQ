# Importing Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.sparse import vstack as sparse_vstack

warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title            = "CineIQ - Movie Intelligence",
    page_icon             = "🎬",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# Custom CSS Styling for Dark Theme with Gold Accents
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700;800&family=Jost:wght@300;400;500&display=swap');

:root {
    --gold   : #C9A84C;
    --dark   : #0A0A0F;
    --card   : #12121A;
    --border : #1E1E2E;
    --text   : #E2E2E8;
    --muted  : #6B6B80;
}

html, body, [class*="css"] {
    font-family      : 'Jost', sans-serif;
    background-color : var(--dark);
    color            : var(--text);
}

section[data-testid="stSidebar"] {
    background   : #0D0D14 !important;
    border-right : 1px solid var(--border);
}

.cineiq-title {
    font-family    : 'Cormorant Garamond', serif;
    font-size      : 3.8rem;
    font-weight    : 800;
    color          : var(--gold);
    line-height    : 1;
    letter-spacing : -1px;
}

.cineiq-sub {
    font-size      : 0.72rem;
    color          : var(--muted);
    letter-spacing : 4px;
    text-transform : uppercase;
    margin-top     : 4px;
    margin-bottom  : 32px;
}

.kpi-card {
    background    : var(--card);
    border        : 1px solid var(--border);
    border-top    : 3px solid var(--gold);
    border-radius : 8px;
    padding       : 20px 24px;
    text-align    : center;
}

.kpi-value {
    font-family : 'Cormorant Garamond', serif;
    font-size   : 2.4rem;
    font-weight : 700;
    color       : var(--gold);
    line-height : 1;
}

.kpi-label {
    font-size      : 0.68rem;
    color          : var(--muted);
    letter-spacing : 2.5px;
    text-transform : uppercase;
    margin-top     : 6px;
}

.filter-bar {
    background    : var(--card);
    border        : 1px solid var(--border);
    border-radius : 8px;
    padding       : 16px 20px;
    margin-bottom : 24px;
}

.filter-label {
    font-size      : 0.65rem;
    color          : var(--gold);
    letter-spacing : 2px;
    text-transform : uppercase;
    margin-bottom  : 12px;
    font-weight    : 500;
}

.filter-count {
    font-size      : 0.75rem;
    color          : var(--muted);
    letter-spacing : 1px;
    margin-top     : 8px;
}

.section-head {
    font-family    : 'Cormorant Garamond', serif;
    font-size      : 1.7rem;
    font-weight    : 700;
    color          : var(--gold);
    border-bottom  : 1px solid var(--border);
    padding-bottom : 8px;
    margin-bottom  : 20px;
    margin-top     : 8px;
}

.movie-card {
    background    : var(--card);
    border        : 1px solid var(--border);
    border-left   : 3px solid var(--gold);
    border-radius : 8px;
    padding       : 14px 18px;
    margin-bottom : 10px;
}

.movie-card-title {
    font-family : 'Cormorant Garamond', serif;
    font-size   : 1.1rem;
    font-weight : 700;
    color       : var(--text);
}

.movie-card-meta {
    font-size  : 0.80rem;
    color      : var(--muted);
    margin-top : 3px;
}

.rating-pill {
    background    : var(--gold);
    color         : #000;
    font-weight   : 600;
    font-size     : 0.78rem;
    padding       : 2px 10px;
    border-radius : 20px;
    float         : right;
}

.pred-box {
    background    : linear-gradient(135deg, #0D0D00, #1A1600);
    border        : 2px solid var(--gold);
    border-radius : 12px;
    padding       : 36px;
    text-align    : center;
    margin-top    : 20px;
}

.pred-number {
    font-family : 'Cormorant Garamond', serif;
    font-size   : 6rem;
    font-weight : 800;
    line-height : 1;
}

.pred-sub {
    font-size      : 0.70rem;
    color          : var(--muted);
    letter-spacing : 3px;
    text-transform : uppercase;
    margin-top     : 8px;
}

.stButton > button {
    background     : var(--gold) !important;
    color          : #000 !important;
    font-weight    : 600 !important;
    font-family    : 'Jost', sans-serif !important;
    border         : none !important;
    border-radius  : 6px !important;
    padding        : 10px 28px !important;
    letter-spacing : 0.5px;
}

.stButton > button:hover {
    background : #E8C84A !important;
    transform  : translateY(-1px);
}

.stTextInput > div > input,
.stSelectbox > div,
.stNumberInput > div > input {
    background    : var(--card) !important;
    border        : 1px solid var(--border) !important;
    color         : var(--text) !important;
    border-radius : 6px !important;
}

.stTabs [data-baseweb="tab"]  { color: var(--muted); }
.stTabs [aria-selected="true"] {
    color        : var(--gold) !important;
    border-color : var(--gold) !important;
}

hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)

# Loading and Caching the Cleaned Dataset with All Engineered Features
@st.cache_data
def load_data():
    df = pd.read_csv(
        r"omdb_movies_cleaned.csv",
        low_memory=False
    )

    # Recreating All Engineered Features from Task 4
    df['log_imdbVotes']    = np.log1p(df['imdbVotes'])
    df['Released']         = pd.to_datetime(df['Released'], errors='coerce')
    df['release_month']    = df['Released'].dt.month.fillna(0).astype(int)
    df['is_awards_season'] = df['release_month'].apply(
        lambda m: 1 if m in [10, 11, 12] else 0
    )
    df['is_english']       = df['Language'].str.contains(
        'English', case=False, na=False
    ).astype(int)

    rated_map = {'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4, 'NC-17': 5,
                 'Not Rated': 0, 'NR': 0, 'Unrated': 0}
    df['rated_encoded']    = df['Rated'].map(rated_map).fillna(0)

    df['win_rate'] = df.apply(
        lambda r: r['Total_Wins'] / r['Total_Nominations']
        if r['Total_Nominations'] > 0 else 0, axis=1
    )

    director_counts          = df['Director'].value_counts()
    prolific                 = director_counts[director_counts >= 10].index
    df['is_prolific_director'] = df['Director'].isin(prolific).astype(int)

    # One Hot Encoding Top 15 Genres as Binary Columns
    top_genres = ['Drama', 'Comedy', 'Action', 'Thriller', 'Romance',
                  'Crime', 'Horror', 'Adventure', 'Mystery', 'Sci-Fi',
                  'Biography', 'Animation', 'Family', 'History', 'Fantasy']
    for g in top_genres:
        df[f'genre_{g.lower().replace("-", "_")}'] = \
            df['Genre'].str.contains(g, na=False).astype(int)

    df['Primary_Genre'] = df['Genre'].str.split(', ').str[0]
    return df

# Loading Pre-Trained LightGBM Model, Scaler and Feature Names from Saved Files
@st.cache_resource
def load_models():
    lgb_model    = joblib.load("lgb_model.pkl")
    scaler_nn    = joblib.load("scaler_nn.pkl")
    all_features = joblib.load("feature_names.pkl")
    return lgb_model, scaler_nn, all_features

# Building TF-IDF Matrix for Content Based Similarity Matching
@st.cache_resource
def build_tfidf(_df):
    # Removing Common English Words and Movie Specific Filler Words
    movie_filler = [
        'movie', 'film', 'story', 'life', 'one', 'two', 'new', 'find', 'must',
        'world', 'time', 'day', 'back', 'way', 'man', 'woman', 'young', 'old',
        'set', 'based', 'true', 'takes', 'gets', 'comes', 'goes', 'makes',
        'tries', 'finds', 'begins', 'soon', 'group', 'help', 'three', 'series'
    ]
    stop_words = list(ENGLISH_STOP_WORDS) + movie_filler

    _df = _df.copy()
    # Combining Genre, Director, Actors and Plot into One Column for Vectorization
    _df['combined_features'] = (
        _df['Genre'].fillna('')    + ' ' +
        _df['Genre'].fillna('')    + ' ' +
        _df['Director'].fillna('') + ' ' +
        _df['Actors'].fillna('')   + ' ' +
        _df['Plot'].fillna('')
    )

    tfidf        = TfidfVectorizer(stop_words=stop_words, max_features=10000)
    tfidf_matrix = tfidf.fit_transform(_df['combined_features'])
    # Building Title to Index Mapping and Handling Duplicates
    title_idx    = pd.Series(_df.index, index=_df['Title'])
    title_idx    = title_idx[~title_idx.index.duplicated(keep='first')]
    return tfidf, tfidf_matrix, title_idx

# Running K-Means Clustering with K=4 on Numeric Movie Features
@st.cache_resource
def build_clusters(_df):
    cluster_features = ['imdbRating', 'RT_Score', 'Metacritic_Score',
                        'Runtime', 'Oscar_Wins', 'Total_Wins', 'imdbVotes']
    df_c             = _df[cluster_features].dropna().copy()
    scaler           = StandardScaler()
    scaled           = scaler.fit_transform(df_c)
    kmeans           = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_c['Cluster']  = kmeans.fit_predict(scaled)
    return df_c

# Extracting Languages that Appear at Least 50 Times in the Dataset for Filtering
@st.cache_data
def get_dataset_languages(_df_language_col):
    all_langs = _df_language_col.str.split(', ').explode().str.strip().dropna()
    lang_counts = all_langs.value_counts()
    relevant_langs = lang_counts[lang_counts >= 50].index.tolist()
    return sorted(relevant_langs)

# Initialising Session State for User Rating Storage
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}

# Loading All Data and Models into Memory
df                                  = load_data()
lgb_model, scaler_nn, all_features  = load_models()
tfidf, tfidf_matrix, title_idx      = build_tfidf(df)
df_cluster                          = build_clusters(df)
dataset_languages                   = get_dataset_languages(df['Language'])

# Extracting All Unique Genres from the Dataset for Filter Dropdowns
all_genres_list = sorted(
    df['Genre'].str.split(', ').explode().dropna().unique().tolist()
)

# Helper Function to Apply Consistent Dark Theme Styling to All Plotly Charts
def styled_chart(fig, title=""):
    fig.update_layout(
        title         = dict(text=title, font=dict(color='#C9A84C', size=14)),
        plot_bgcolor  = 'rgba(0,0,0,0)',
        paper_bgcolor = 'rgba(0,0,0,0)',
        font_color    = '#E2E2E8',
        margin        = dict(t=40, b=20, l=10, r=10),
        legend        = dict(bgcolor='rgba(0,0,0,0)'),
        xaxis         = dict(gridcolor='#1E1E2E'),
        yaxis         = dict(gridcolor='#1E1E2E'),
        dragmode      = 'zoom',
    )
    # Enabling Interactive Toolbar with Zoom, Pan, Select, Lasso and Hover
    config = {
        'displayModeBar'        : True,
        'scrollZoom'            : True,
        'modeBarButtonsToAdd'   : ['select2d', 'lasso2d'],
        'modeBarButtonsToRemove': ['sendDataToCloud'],
        'displaylogo'           : False,
        'toImageButtonOptions'  : {'format': 'png', 'scale': 2},
    }
    return fig, config

# Helper Function to Apply Global Slicer Filters to Any Dataframe - Works Like Power BI Slicers
def apply_global_filters(data, decade_val, genre_val, rating_range):
    filtered = data.copy()
    if decade_val != "All":
        filtered = filtered[filtered['Decade'] == decade_val]
    if genre_val != "All":
        filtered = filtered[filtered['Genre'].str.contains(genre_val, na=False)]
    filtered = filtered[
        (filtered['imdbRating'] >= rating_range[0]) &
        (filtered['imdbRating'] <= rating_range[1])
    ]
    return filtered

# Sidebar Navigation with App Branding
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:28px 0 36px 0;'>
        <div style='font-family:"Cormorant Garamond",serif; font-size:2rem;
                    font-weight:800; color:#C9A84C;'>CineIQ</div>
        <div style='font-size:0.62rem; color:#6B6B80; letter-spacing:4px;
                    text-transform:uppercase; margin-top:4px;'>
                    Movie Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Overview",
         "🔍  Search & Recommend",
         "📊  EDA Explorer",
         "🎬  Predict Rating",
         "🏆  Leaderboard"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem; color:#3A3A50; text-align:center;'>"
        "OMDB Dataset · 25,180 Films</div>",
        unsafe_allow_html=True
    )

# Page 1 - Overview Dashboard with Global Filters, KPIs and Summary Charts
if page == "🏠  Overview":

    st.markdown('<div class="cineiq-title">CineIQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cineiq-sub">Movie Intelligence Dashboard · OMDB Dataset</div>',
        unsafe_allow_html=True
    )

    # Global Slicer Filters - Changing These Filters Updates All Charts Below Like Power BI
    st.markdown(
        '<div class="filter-label">🎛️  Dashboard Filters — Select to Filter All Visuals</div>',
        unsafe_allow_html=True
    )
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        ov_decade = st.selectbox(
            "Decade",
            ["All"] + sorted(df['Decade'].dropna().unique().tolist()),
            key="ov_decade"
        )
    with fc2:
        ov_genre = st.selectbox(
            "Genre",
            ["All"] + all_genres_list,
            key="ov_genre"
        )
    with fc3:
        ov_rating = st.slider(
            "IMDb Rating Range",
            1.0,
            10.0,
            (1.0, 10.0),
            step=0.1,
            key="ov_rating"
        )

    # Applying Slicer Filters to Create Filtered Dataset for All Charts
    df_ov = apply_global_filters(df, ov_decade, ov_genre, ov_rating)

    # Showing Filter Context - How Many Movies Match Current Filters
    st.markdown(
        f'<div class="filter-count">Showing <b>{len(df_ov):,}</b> of '
        f'{len(df):,} movies based on selected filters</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Row Using Filtered Data
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (f"{len(df_ov):,}",                       "Movies Found"),
        (f"{df_ov['imdbRating'].mean():.2f}"
         if len(df_ov) > 0 else "—",              "Avg IMDb Rating"),
        (f"{int(df_ov['Runtime'].mean())} min"
         if len(df_ov) > 0 else "—",              "Avg Runtime"),
        (f"{df_ov['Director'].nunique():,}"
         if len(df_ov) > 0 else "—",              "Unique Directors"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], kpis):
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{val}</div>'
            f'<div class="kpi-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    # Movies Per Decade Chart Using Filtered Data
    with col_l:
        st.markdown('<div class="section-head">Movies Per Decade</div>',
                    unsafe_allow_html=True)
        if len(df_ov) > 0:
            dec         = df_ov['Decade'].value_counts().sort_index().reset_index()
            dec.columns = ['Decade', 'Count']
            fig         = px.bar(dec, x='Decade', y='Count',
                                 color='Count', color_continuous_scale='YlOrBr',
                                 hover_data={'Count': ':,'})
            fig.update_traces(marker_line_width=0)
            fig.update_coloraxes(showscale=False)
            fig, config = styled_chart(fig)
            st.plotly_chart(fig, use_container_width=True, config=config)
        else:
            st.info("No movies match current filters")

    # Top 10 Genres Chart Using Filtered Data
    with col_r:
        st.markdown('<div class="section-head">Top 10 Genres</div>',
                    unsafe_allow_html=True)
        if len(df_ov) > 0:
            gc          = (df_ov['Genre'].str.split(', ').explode()
                           .value_counts().head(10).reset_index())
            gc.columns  = ['Genre', 'Count']
            fig2        = px.bar(gc, x='Count', y='Genre', orientation='h',
                                 color='Count', color_continuous_scale='YlOrBr',
                                 hover_data={'Count': ':,'})
            fig2.update_coloraxes(showscale=False)
            fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
            fig2, config2 = styled_chart(fig2)
            st.plotly_chart(fig2, use_container_width=True, config=config2)
        else:
            st.info("No movies match current filters")

    # IMDb Rating Distribution Using Filtered Data
    st.markdown('<div class="section-head">IMDb Rating Distribution</div>',
                unsafe_allow_html=True)
    if len(df_ov) > 0:
        fig3 = px.histogram(df_ov, x='imdbRating', nbins=40,
                            color_discrete_sequence=['#C9A84C'],
                            labels={'imdbRating': 'IMDb Rating'})
        fig3.update_traces(marker_line_width=0,
                           hovertemplate='Rating: %{x}<br>Count: %{y}<extra></extra>')
        fig3, config3 = styled_chart(fig3)
        st.plotly_chart(fig3, use_container_width=True, config=config3)
    else:
        st.info("No movies match current filters")

    # Model Performance Summary - These are Fixed Results Not Affected by Filters
    st.markdown('<div class="section-head">Model Performance Summary</div>',
                unsafe_allow_html=True)
    model_results = pd.DataFrame({
        'Model' : ['Random Forest (Baseline)', 'Neural Network',
                   'Random Forest (Tuned)',    'XGBoost',
                   'LightGBM ✓ Best'],
        'RMSE'  : [0.8460, 0.7419, 0.7163, 0.6883, 0.6846],
        'R²'    : [0.4607, 0.5853, 0.6135, 0.6430, 0.6468]
    }).sort_values('R²', ascending=False)

    fig4 = px.bar(
        model_results, x='R²', y='Model', orientation='h',
        color='R²', color_continuous_scale='YlOrBr', text='R²'
    )
    fig4.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig4.update_coloraxes(showscale=False)
    fig4.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig4, config4 = styled_chart(fig4, "R² Score Comparison — Higher is Better")
    st.plotly_chart(fig4, use_container_width=True, config=config4)


# Page 2 - Search and Recommendation System with Rating Storage
elif page == "🔍  Search & Recommend":

    st.markdown('<div class="cineiq-title">Search & Discover</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="cineiq-sub">Find Movies · Rate Them · Get Recommendations</div>',
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["🔍  Search Movies", "⭐  Get Recommendations"])

    # Search Tab - Find Movies by Title, Director, Genre or Actor Name
    with tab1:
        query = st.text_input(
            "Search by title, director, genre or actor",
            placeholder="e.g.  Christopher Nolan  /  horror  /  Leonardo DiCaprio"
        )
        if query:
            q    = query.lower().strip()
            mask = (
                df['Title'].str.lower().str.contains(q, na=False, regex=False)    |
                df['Director'].str.lower().str.contains(q, na=False, regex=False) |
                df['Genre'].str.lower().str.contains(q, na=False, regex=False)    |
                df['Actors'].str.lower().str.contains(q, na=False, regex=False)
            )
            results = (df[mask]
                       .drop_duplicates('Title')
                       .sort_values('imdbRating', ascending=False))

            st.markdown(f"**Found {len(results)} results** — showing top 20")
            for _, row in results.head(20).iterrows():
                st.markdown(
                    f'<div class="movie-card">'
                    f'<span class="rating-pill">⭐ {row["imdbRating"]}</span>'
                    f'<div class="movie-card-title">'
                    f'{row["Title"]} ({int(row["Year"])})</div>'
                    f'<div class="movie-card-meta">'
                    f'{row["Genre"]} · {row["Director"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # Recommendation Tab - Rate a Movie and Get Similar or Different Suggestions
    with tab2:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            movie_input = st.selectbox(
                "Select a movie you have watched",
                options     = sorted(df['Title'].dropna().unique()),
                index       = None,
                placeholder = "Type to search..."
            )
        with col_b:
            # Typeable Rating Input with Decimal Support Upto 1 Place
            user_rating = st.number_input(
                "Your Rating",
                min_value = 1.0,
                max_value = 10.0,
                value     = 7.0,
                step      = 0.1,
                format    = "%.1f"
            )

        # All Options from 3 to 10 for Number of Recommendations
        n_recs = st.slider(
            "Number of Recommendations", min_value=3, max_value=10, value=5, step=1
        )

        if st.button("Get Recommendations") and movie_input:

            if movie_input not in title_idx.index:
                st.error("Movie not found in index.")
            else:
                # Storing the Rating for Future Reference
                if movie_input in st.session_state.user_ratings:
                    st.session_state.user_ratings[movie_input].append(user_rating)
                else:
                    st.session_state.user_ratings[movie_input] = [user_rating]

                idx = title_idx[movie_input]

                if user_rating >= 7.0:
                    # User Liked the Movie - Computing Cosine Similarity for Similar Movies
                    vec        = tfidf_matrix[idx]
                    sims       = cosine_similarity(vec, tfidf_matrix).flatten()
                    sim_scores = sorted(enumerate(sims),
                                        key=lambda x: x[1], reverse=True)[1:51]
                    indices    = [i[0] for i in sim_scores]
                    recs       = (df.iloc[indices]
                                  .drop_duplicates('Title')
                                  .sort_values('imdbRating', ascending=False)
                                  .head(n_recs))
                    header = f"Because you liked **{movie_input}**, you might enjoy:"

                else:
                    # User Did Not Enjoy - Suggesting Well Rated Movies from Different Genres
                    input_genres = set(df.iloc[idx]['Genre'].split(', '))
                    other        = df[df.index != idx].copy()
                    other['genre_overlap'] = other['Genre'].apply(
                        lambda g: len(set(str(g).split(', ')) & input_genres)
                    )
                    # Filtering Out Obscure Movies with Very Few Votes
                    other = other[other['imdbVotes'] >= 5000]
                    recs  = (other[other['genre_overlap'] == 0]
                             .sort_values('imdbRating', ascending=False)
                             .drop_duplicates('Title')
                             .head(n_recs))
                    header = f"Since you didn't enjoy **{movie_input}**, try:"

                # Displaying Recommendation Results as Movie Cards
                st.markdown(f"<br>{header}", unsafe_allow_html=True)
                for _, row in recs.iterrows():
                    st.markdown(
                        f'<div class="movie-card">'
                        f'<span class="rating-pill">⭐ {row["imdbRating"]}</span>'
                        f'<div class="movie-card-title">'
                        f'{row["Title"]} ({int(row["Year"])})</div>'
                        f'<div class="movie-card-meta">'
                        f'{row["Genre"]} · {row["Director"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # Displaying All Stored Ratings as a History Table
        if st.session_state.user_ratings:
            st.markdown("---")
            st.markdown('<div class="section-head">Your Rating History</div>',
                        unsafe_allow_html=True)
            history = [
                {
                    'Movie'       : t,
                    'Avg Rating'  : round(np.mean(v), 1),
                    'Times Rated' : len(v),
                    'All Ratings' : str(v)
                }
                for t, v in st.session_state.user_ratings.items()
            ]
            st.dataframe(
                pd.DataFrame(history),
                use_container_width=True,
                hide_index=True
            )


# Page 3 - EDA Explorer with Global Slicer Filters Like Power BI
elif page == "📊  EDA Explorer":

    st.markdown('<div class="cineiq-title">EDA Explorer</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="cineiq-sub">Explore 35 Years of Cinema Data</div>',
                unsafe_allow_html=True)

    # Global Slicer Filters for EDA - Changing These Updates All Charts Below
    st.markdown(
        '<div class="filter-label">🎛️  Data Filters — Apply Across All Analyses</div>',
        unsafe_allow_html=True
    )
    ef1, ef2, ef3, ef4 = st.columns(4)
    with ef1:
        eda_decade = st.selectbox(
            "Decade",
            ["All"] + sorted(df['Decade'].dropna().unique().tolist()),
            key="eda_decade"
        )
    with ef2:
        eda_genre = st.selectbox(
            "Genre",
            ["All"] + all_genres_list,
            key="eda_genre"
        )
    with ef3:
        eda_lang = st.selectbox(
            "Language",
            ["All"] + dataset_languages,
            key="eda_lang"
        )
    with ef4:
        eda_rating = st.slider(
            "Rating Range",
            1.0,
            10.0,
            (1.0, 10.0),
            step=0.1,
            key="eda_rating"
        )

    # Applying All Slicer Filters to Create Filtered Dataset
    df_eda = apply_global_filters(df, eda_decade, eda_genre, eda_rating)
    if eda_lang != "All":
        df_eda = df_eda[df_eda['Language'].str.contains(eda_lang, case=False, na=False)]

    # Showing How Many Movies Match the Current Filter Selection
    st.markdown(
        f'<div class="filter-count">Analysing <b>{len(df_eda):,}</b> of '
        f'{len(df):,} movies based on selected filters</div>',
        unsafe_allow_html=True
    )

    # Analysis Type Selection
    analysis = st.selectbox(
        "Choose Analysis",
        ["Rating Trends Over Time", "Genre vs Rating",
         "Runtime Analysis",        "Language & Country",
         "Awards vs Rating",        "Correlation Heatmap",
         "K-Means Cluster Analysis"]
    )

    if len(df_eda) == 0:
        st.warning("No movies match your current filters. Try broadening your selection.")

    elif analysis == "Rating Trends Over Time":
        # Average IMDb Rating by Year Using Filtered Data
        yearly = df_eda.groupby('Year')['imdbRating'].mean().reset_index()
        fig    = px.line(yearly, x='Year', y='imdbRating',
                         markers=True, color_discrete_sequence=['#C9A84C'],
                         hover_data={'imdbRating': ':.2f'})
        fig.update_traces(marker_size=4)
        fig, config = styled_chart(fig, "Average IMDb Rating by Year")
        st.plotly_chart(fig, use_container_width=True, config=config)

        # Rating Distribution by Decade Using Filtered Data
        fig2 = px.box(df_eda, x='Decade', y='imdbRating', color='Decade',
                      color_discrete_sequence=px.colors.sequential.YlOrBr)
        fig2.update_layout(showlegend=False)
        fig2, config2 = styled_chart(fig2, "Rating Distribution by Decade")
        st.plotly_chart(fig2, use_container_width=True, config=config2)

    elif analysis == "Genre vs Rating":
        # Average Rating per Genre with Minimum 100 Movies Threshold
        genre_rating = (
            df_eda.assign(Genre=df_eda['Genre'].str.split(', ')).explode('Genre')
            .groupby('Genre')['imdbRating']
            .agg(['mean', 'count']).reset_index()
        )
        min_count = min(100, genre_rating['count'].max()) if len(genre_rating) > 0 else 100
        genre_rating = (genre_rating[genre_rating['count'] >= min(100, min_count)]
                        .sort_values('mean', ascending=False).head(20))
        fig = px.bar(genre_rating, x='Genre', y='mean',
                     color='mean', color_continuous_scale='YlOrBr',
                     labels={'mean': 'Avg Rating'},
                     hover_data={'mean': ':.2f', 'count': ':,'})
        fig.update_coloraxes(showscale=False)
        fig, config = styled_chart(fig, "Average IMDb Rating by Genre")
        st.plotly_chart(fig, use_container_width=True, config=config)

    elif analysis == "Runtime Analysis":
        c1, c2 = st.columns(2)
        with c1:
            # Distribution of Movie Runtime Using Filtered Data
            fig = px.histogram(df_eda, x='Runtime', nbins=40,
                               color_discrete_sequence=['#C9A84C'])
            fig.update_traces(
                hovertemplate='Runtime: %{x} min<br>Count: %{y}<extra></extra>'
            )
            fig, config = styled_chart(fig, "Distribution of Movie Runtime")
            st.plotly_chart(fig, use_container_width=True, config=config)
        with c2:
            # Average Runtime by Decade Using Filtered Data
            rt   = df_eda.groupby('Decade')['Runtime'].mean().reset_index()
            fig2 = px.bar(rt, x='Decade', y='Runtime',
                          color='Runtime', color_continuous_scale='YlOrBr',
                          hover_data={'Runtime': ':.1f'})
            fig2.update_coloraxes(showscale=False)
            fig2, config2 = styled_chart(fig2, "Average Runtime by Decade")
            st.plotly_chart(fig2, use_container_width=True, config=config2)

    elif analysis == "Language & Country":
        # Top 12 Languages Using Filtered Data
        lang         = (df_eda['Language'].str.split(', ').explode()
                        .value_counts().head(12).reset_index())
        lang.columns = ['Language', 'Count']
        fig          = px.bar(lang, x='Language', y='Count',
                              color='Count', color_continuous_scale='YlOrBr',
                              hover_data={'Count': ':,'})
        fig.update_coloraxes(showscale=False)
        fig, config = styled_chart(fig, "Top 12 Languages")
        st.plotly_chart(fig, use_container_width=True, config=config)

        # Top 15 Countries Using Filtered Data
        country         = (df_eda['Country'].str.split(', ').explode()
                           .value_counts().head(15).reset_index())
        country.columns = ['Country', 'Count']
        fig2            = px.bar(country, x='Count', y='Country',
                                 orientation='h', color='Count',
                                 color_continuous_scale='YlOrBr',
                                 hover_data={'Count': ':,'})
        fig2.update_coloraxes(showscale=False)
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig2, config2 = styled_chart(fig2, "Top 15 Countries by Movie Count")
        st.plotly_chart(fig2, use_container_width=True, config=config2)

    elif analysis == "Awards vs Rating":
        # Oscar Wins vs IMDb Rating Scatter Plot Using Filtered Data
        df_oscar = df_eda[df_eda['Oscar_Wins'] > 0]
        if len(df_oscar) > 0:
            fig = px.scatter(
                df_oscar,
                x                      = 'Oscar_Wins',
                y                      = 'imdbRating',
                size                   = 'imdbVotes',
                hover_name             = 'Title',
                hover_data             = {'Year': True, 'Genre': True,
                                          'imdbVotes': ':,', 'imdbRating': ':.1f'},
                color                  = 'imdbRating',
                color_continuous_scale = 'YlOrBr',
                opacity                = 0.7
            )
            fig, config = styled_chart(fig, "Oscar Wins vs IMDb Rating (sized by votes)")
            st.plotly_chart(fig, use_container_width=True, config=config)
        else:
            st.info("No Oscar winning movies match current filters")

    elif analysis == "Correlation Heatmap":
        # Correlation Matrix of Numeric Features Using Filtered Data
        num_cols = ['imdbRating', 'RT_Score', 'Metacritic_Score', 'Runtime',
                    'imdbVotes', 'Oscar_Wins', 'Total_Wins', 'Total_Nominations']
        corr     = df_eda[num_cols].corr()
        fig      = px.imshow(corr, text_auto='.2f',
                             color_continuous_scale='RdYlBu', zmin=-1, zmax=1)
        fig, config = styled_chart(fig, "Feature Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True, config=config)

    elif analysis == "K-Means Cluster Analysis":
        # Cluster Visualization - Filtering Cluster Data Based on Global Filters
        cluster_labels = {
            0 : "Cluster 0 — Poorly Received",
            1 : "Cluster 1 — Well Reviewed Mainstream",
            2 : "Cluster 2 — Average Visibility",
            3 : "Cluster 3 — Elite Blockbusters"
        }
        df_c = df_cluster.copy()
        # Applying Rating Filter to Cluster Data
        df_c = df_c[
            (df_c['imdbRating'] >= eda_rating[0]) &
            (df_c['imdbRating'] <= eda_rating[1])
        ]
        df_c['Cluster Label'] = df_c['Cluster'].map(cluster_labels)

        if len(df_c) > 0:
            fig = px.scatter(
                df_c, x='imdbRating', y='RT_Score',
                color             = 'Cluster Label',
                opacity           = 0.5,
                hover_data        = {'imdbRating': ':.1f', 'RT_Score': ':.0f',
                                     'imdbVotes': ':,'},
                color_discrete_sequence = px.colors.qualitative.Set2
            )
            fig, config = styled_chart(fig, "K-Means Clusters — IMDb Rating vs RT Score")
            st.plotly_chart(fig, use_container_width=True, config=config)

            # Cluster Profile Table Showing Average Values per Segment
            st.markdown('<div class="section-head">Cluster Profiles</div>',
                        unsafe_allow_html=True)
            profile = (df_c.groupby('Cluster Label')
                       .mean(numeric_only=True).round(2)
                       .drop(columns=['Cluster']))
            st.dataframe(profile, use_container_width=True)
        else:
            st.info("No cluster data matches current filters")


# Page 4 - IMDb Rating Prediction Using Trained LightGBM Model
elif page == "🎬  Predict Rating":

    st.markdown('<div class="cineiq-title">Predict IMDb Rating</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="cineiq-sub">Enter Movie Details · Get AI Predicted Rating</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Basic Info**")
        # No Year Restriction - User Can Enter Any Year They Want
        m_year    = st.number_input("Release Year", value=2025)
        m_runtime = st.number_input("Runtime (min)",    60, 400, 110)
        m_rated   = st.selectbox(
            "Content Rating",
            ["Not Rated", "G", "PG", "PG-13", "R", "NC-17"]
        )
        m_english = st.checkbox("English Language Film", value=True)
        m_month   = st.slider("Release Month", 1, 12, 6)

    with col2:
        st.markdown("**Scores & Awards**")
        m_rt    = st.slider("Rotten Tomatoes Score", 0, 100, 65)
        m_meta  = st.slider("Metacritic Score",      0, 100, 60)
        m_votes = st.number_input(
            "Expected IMDb Votes", 100, 3_000_000, 50000, step=1000
        )
        m_oscar = st.number_input("Oscar Wins",        0, 15,  0)
        m_wins  = st.number_input("Total Award Wins",  0, 500, 5)
        m_noms  = st.number_input("Total Nominations", 0, 600, 15)

    # Genre Selection Using Checkboxes for Top 15 Genres
    st.markdown("**Select Genres**")
    top_genres  = ['Drama', 'Comedy', 'Action', 'Thriller', 'Romance',
                   'Crime', 'Horror', 'Adventure', 'Mystery', 'Sci-Fi',
                   'Biography', 'Animation', 'Family', 'History', 'Fantasy']
    genre_cols  = st.columns(5)
    genre_flags = {}
    for i, g in enumerate(top_genres):
        genre_flags[g] = genre_cols[i % 5].checkbox(g)

    if st.button("🎬  Predict IMDb Rating"):

        # Building Prediction Input with All 29 Features Matching Training Set
        rated_map  = {'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4,
                      'NC-17': 5, 'Not Rated': 0}
        win_rate   = m_wins / m_noms if m_noms > 0 else 0

        input_data = {
            'Runtime'              : m_runtime,
            'RT_Score'             : m_rt,
            'Metacritic_Score'     : m_meta,
            'Oscar_Wins'           : m_oscar,
            'Total_Wins'           : m_wins,
            'Total_Nominations'    : m_noms,
            'log_imdbVotes'        : np.log1p(m_votes),
            'Year'                 : m_year,
            'release_month'        : m_month,
            'is_awards_season'     : 1 if m_month in [10, 11, 12] else 0,
            'is_english'           : int(m_english),
            'rated_encoded'        : rated_map.get(m_rated, 0),
            'win_rate'             : win_rate,
            'is_prolific_director' : 0,
        }
        for g in top_genres:
            input_data[f'genre_{g.lower().replace("-", "_")}'] = int(genre_flags[g])

        # Predicting IMDb Rating Using the Best Model (LightGBM)
        X_pred    = pd.DataFrame([input_data])[all_features]
        predicted = round(float(lgb_model.predict(X_pred)[0]), 1)
        predicted = max(1.0, min(10.0, predicted))

        # Displaying Prediction with Color Coded Verdict
        color   = ("#4CAF50" if predicted >= 7
                   else "#FF9800" if predicted >= 5
                   else "#F44336")
        verdict = ("🟢 Likely a Hit"        if predicted >= 7
                   else "🟡 Mixed Reception" if predicted >= 5
                   else "🔴 May Struggle")

        st.markdown(
            f'<div class="pred-box">'
            f'<div class="pred-sub">Predicted IMDb Rating</div>'
            f'<div class="pred-number" style="color:{color}">{predicted}</div>'
            f'<div style="color:#6B6B80; font-size:0.85rem; margin-top:4px;">'
            f'out of 10</div>'
            f'<div style="font-size:1.1rem; font-weight:600; margin-top:16px;">'
            f'{verdict}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Showing Movies with Similar Ratings for Context
        similar = (df[
            (df['imdbRating'] >= predicted - 0.4) &
            (df['imdbRating'] <= predicted + 0.4)
        ].sort_values('imdbVotes', ascending=False).head(5))

        st.markdown(
            f"<br>**Movies with similar ratings (~{predicted}):**",
            unsafe_allow_html=True
        )
        for _, row in similar.iterrows():
            st.markdown(
                f'<div class="movie-card">'
                f'<span class="rating-pill">⭐ {row["imdbRating"]}</span>'
                f'<div class="movie-card-title">'
                f'{row["Title"]} ({int(row["Year"])})</div>'
                f'<div class="movie-card-meta">{row["Genre"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# Page 5 - Leaderboard with Multi-Filter Ranking System
elif page == "🏆  Leaderboard":

    st.markdown('<div class="cineiq-title">Leaderboard</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="cineiq-sub">The Best of Cinema — Filtered & Ranked</div>',
        unsafe_allow_html=True
    )

    # Filter Controls for Leaderboard Ranking
    c1, c2, c3, c4, c5 = st.columns(5)

    min_votes = c1.number_input(
        "Min IMDb Votes", 100, 500000, 10000, step=1000
    )

    decade_filter = c2.selectbox(
        "Decade", ["All"] + sorted(df['Decade'].dropna().unique().tolist())
    )

    # Year Filter - All Individual Years Available
    all_years = sorted(df['Year'].dropna().unique().tolist())
    year_filter = c3.selectbox(
        "Year", ["All"] + [int(y) for y in all_years]
    )

    genre_filter = c4.selectbox(
        "Genre", ["All"] + all_genres_list
    )

    # Language Filter - Only Languages that Appear Frequently in the Dataset
    language_filter = c5.selectbox(
        "Language", ["All"] + dataset_languages
    )

    # Applying All Leaderboard Filters
    filtered = df[df['imdbVotes'] >= min_votes].copy()

    if decade_filter != "All":
        filtered = filtered[filtered['Decade'] == decade_filter]

    if year_filter != "All":
        filtered = filtered[filtered['Year'] == int(year_filter)]

    if genre_filter != "All":
        filtered = filtered[
            filtered['Genre'].str.contains(genre_filter, na=False)
        ]

    if language_filter != "All":
        filtered = filtered[
            filtered['Language'].str.contains(language_filter, case=False, na=False)
        ]

    # Ranking Top 50 Movies by IMDb Rating After Filters
    top = (filtered
           .drop_duplicates('Title')
           .sort_values('imdbRating', ascending=False)
           .head(50))

    st.markdown(
        f"<br>**Showing top {len(top)} movies** "
        f"({len(filtered):,} matched your filters)",
        unsafe_allow_html=True
    )

    # Displaying Ranked Movie Cards with Medal Icons for Top 3
    for rank, (_, row) in enumerate(top.iterrows(), 1):
        medal = ("🥇" if rank == 1 else "🥈" if rank == 2
                 else "🥉" if rank == 3 else f"#{rank}")

        # Truncating Long Language Strings for Display
        lang_display = row.get('Language', '')
        if isinstance(lang_display, str) and len(lang_display) > 40:
            lang_display = lang_display[:40] + "..."

        st.markdown(
            f'<div class="movie-card">'
            f'<span class="rating-pill">⭐ {row["imdbRating"]}</span>'
            f'<div class="movie-card-title">'
            f'{medal}  {row["Title"]} ({int(row["Year"])})</div>'
            f'<div class="movie-card-meta">'
            f'{row["Genre"]} · {row["Director"]} · '
            f'{int(row["imdbVotes"]):,} votes · {lang_display}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
