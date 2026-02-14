import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="Naver Insight Dashboard Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (Premium Design)
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e9ecef;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 1. 데이터 로드 함수
@st.cache_data
def load_all_data():
    # 쇼핑 데이터 통합
    shop_files = glob.glob(os.path.join(DATA_DIR, 'shopping_*.csv'))
    shop_dfs = []
    for f in shop_files:
        if 'trend' not in f:
            keyword = os.path.basename(f).replace('shopping_', '').replace('.csv', '')
            try:
                df = pd.read_csv(f)
                df['keyword'] = keyword
                shop_dfs.append(df)
            except Exception as e:
                st.error(f"Error loading {f}: {e}")
    df_shop = pd.concat(shop_dfs, ignore_index=True) if shop_dfs else pd.DataFrame()

    # 블로그 데이터 통합
    blog_files = glob.glob(os.path.join(DATA_DIR, 'blog_*.csv'))
    blog_dfs = []
    for f in blog_files:
        keyword = os.path.basename(f).replace('blog_', '').replace('.csv', '')
        try:
            df = pd.read_csv(f)
            df['keyword'] = keyword
            blog_dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {f}: {e}")
    df_blog = pd.concat(blog_dfs, ignore_index=True) if blog_dfs else pd.DataFrame()

    # 트렌드 데이터
    trend_files = glob.glob(os.path.join(DATA_DIR, 'shopping_trend_health_food_*.csv'))
    df_trend = pd.read_csv(trend_files[0]) if trend_files else pd.DataFrame()
    if not df_trend.empty:
        df_trend['date'] = pd.to_datetime(df_trend['date'])

    return df_shop, df_blog, df_trend

def extract_keywords_pro(texts, top_n=20):
    if not texts or len(texts) == 0: return pd.DataFrame()
    tfidf = TfidfVectorizer(max_features=1000, stop_words=None)
    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()
    sums = tfidf_matrix.sum(axis=0)
    data = []
    for col, name in enumerate(feature_names):
        data.append((name, sums[0, col]))
    ranking = pd.DataFrame(data, columns=['단어', '점수']).sort_values('점수', ascending=False)
    return ranking.head(top_n)

# 메인 헤더
st.title("🚀 네이버 이커머스 & 소셜 분석 프로 대시보드")
st.markdown("---")

# 데이터 로딩
df_shop, df_blog, df_trend = load_all_data()

if df_shop.empty or df_blog.empty or df_trend.empty:
    st.error("데이터 수집 파일이 전무하거나 경로가 잘못되었습니다. 수집 스크립트를 먼저 실행해주세요.")
    st.stop()

# 사이드바 설정
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/analytics.png", width=150)
    st.header("⚙️ 분석 대시보드 설정")
    st.markdown("수집된 데이터를 실시간으로 필터링하고 분석합니다.")
    
    unique_keywords = df_shop['keyword'].unique()
    selected_keywords = st.multiselect(
        "분석 키워드 선택", 
        unique_keywords, 
        default=list(unique_keywords)
    )
    
    st.divider()
    st.info("💡 **Tip**: 탭을 이동하며 상세 분석 내용을 확인하세요.")
    st.write(f"**현재 데이터셋 상품 수**: {len(df_shop):,}개")
    st.write(f"**현재 데이터셋 블로그 수**: {len(df_blog):,}개")

# 데이터 필터링
df_shop_filtered = df_shop[df_shop['keyword'].isin(selected_keywords)]
df_blog_filtered = df_blog[df_blog['keyword'].isin(selected_keywords)]

# 상단 메트릭 (Key Performance Indicators)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("총 검색 상품 수", f"{len(df_shop_filtered)}건")
with m2:
    st.metric("평균 판매가", f"₩{int(df_shop_filtered['lprice'].mean()):,}")
with m3:
    st.metric("분석 블로그 게시물", f"{len(df_blog_filtered)}건")
with m4:
    last_trend = df_trend['ratio'].iloc[-1] if not df_trend.empty else 0
    st.metric("최신 검색 트렌드 지수", f"{last_trend:.1f}%")

# 탭 구성
tab1, tab2, tab3, tab4 = st.tabs(["📊 트렌드 심층 분석", "🛒 쇼핑 마켓 EDA", "📝 소셜/블로그 분석", "📋 데이터 익스플로러"])

# ---------------------------------------------------------
# Tab 1: 트렌드 심층 분석
# ---------------------------------------------------------
with tab1:
    st.subheader("1. 건강기능식품 쇼핑 검색 트렌드 (최근 1년)")
    
    # [Chart 1] Plotly 시계열 라인 차트
    fig_line = px.line(
        df_trend, x='date', y='ratio', color='category',
        title="네이버 쇼핑 검색량 추이 (범주: 건강식품)",
        labels={'ratio': '검색 비율(%)', 'date': '날짜'},
        template="plotly_white"
    )
    fig_line.update_traces(line_width=2)
    st.plotly_chart(fig_line, use_container_width=True)
    st.caption("**[해석]**: 지난 1년간의 트렌드를 분석한 결과, 특정 연휴(명절 전후)나 이벤트 시점에 검색량이 최고치(100%)를 기록하는 경향이 뚜렷합니다. 이는 건강기능식품이 선물용 수요와 밀접하게 연동되어 있음을 시사합니다.")

    c1, c2 = st.columns(2)
    with c1:
        # [Table 1] 트렌드 기초 통계표
        st.subheader("트렌드 수치 요약 (Table 1)")
        trend_summary = df_trend.groupby('category')['ratio'].agg(['mean', 'max', 'min', 'std']).reset_index()
        trend_summary.columns = ['카테고리', '평균 비율', '최대 비율', '최소 비율', '표준편차']
        st.table(trend_summary.style.format(precision=2))
        
    with c2:
        # [Table 2] 주요 기간별(분기별) 평균 통계
        st.subheader("분기별 평균 검색 지수 (Table 2)")
        df_trend['quarter'] = df_trend['date'].dt.to_period('Q').astype(str)
        q_trend = df_trend.groupby('quarter')['ratio'].mean().reset_index()
        q_trend.columns = ['분기', '평균 지수']
        st.dataframe(q_trend, use_container_width=True)

# ---------------------------------------------------------
# Tab 2: 쇼핑 마켓 EDA
# ---------------------------------------------------------
with tab2:
    st.subheader("2. 쇼핑 검색 상품 분석")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # [Chart 2] 키워드별 가격 분포 박스플롯 (Plotly)
        fig_price_box = px.box(
            df_shop_filtered, x="keyword", y="lprice", color="keyword",
            title="키워드별 상품 가격 분포 (Box Plot)",
            labels={'lprice': '판매가(원)', 'keyword': '검색어'},
            template="plotly_white", points="outliers"
        )
        st.plotly_chart(fig_price_box, use_container_width=True)
        st.caption("**[해석]**: 오메가3 상품군은 대용량 프리미엄 라인의 존재로 인해 비타민D 대비 가격 스펙트럼이 훨씬 넓게 형성되어 있습니다. 중간값(Median) 또한 오메가3가 높게 나타나 고관여 제품군임을 확인할 수 있습니다.")

    with col2:
        # [Chart 3] 주요 브랜드 점유율 (Plotly Pie)
        brand_counts = df_shop_filtered['brand'].value_counts().head(10).reset_index()
        brand_counts.columns = ['brand', 'count']
        fig_brand_pie = px.pie(
            brand_counts, values='count', names='brand', 
            title="상위 10개 브랜드 노출 비중",hole=.3
        )
        st.plotly_chart(fig_brand_pie, use_container_width=True)
        st.caption("**[해석]**: 상위 몇 개의 대형 브랜드가 전체 검색 결과의 상당 부분을 점유하고 있어, 영양제 시장 내 브랜드 파워의 영향력이 매우 크다는 것을 알 수 있습니다.")

    # [Chart 4] 상위 쇼핑몰 분포 (Plotly Horizontal Bar)
    st.subheader("주요 유통 채널 분포 (Chart 4)")
    mall_data = df_shop_filtered['mallName'].value_counts().head(20).reset_index()
    mall_data.columns = ['mallName', 'count']
    fig_mall_bar = px.bar(
        mall_data, x='count', y='mallName', orientation='h',
        color='count', color_continuous_scale='Viridis',
        title="상위 20개 노출 유통 채널",
        labels={'count': '상품 수', 'mallName': '쇼핑몰명'}
    )
    fig_mall_bar.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_mall_bar, use_container_width=True)
    st.caption("**[해석]**: 네이버 스마트스토어와 쿠팡, 오픈마켓 등 메이저 플랫폼을 통한 유통이 주를 이루고 있습니다. 특히 브랜드 직영 스토어의 노출 빈도가 높게 나타나 직접 판매(D2C) 전략이 강화되고 있음을 시사합니다.")

    st.divider()
    t_c1, t_c2 = st.columns(2)
    with t_c1:
        # [Table 3] 키워드별 가격 요약표
        st.subheader("키워드별 가격 통계 요약 (Table 3)")
        price_summary = df_shop_filtered.groupby('keyword')['lprice'].agg(['count', 'mean', 'min', 'max']).reset_index()
        price_summary.columns = ['키워드', '상품건수', '평균가', '최저가', '최고가']
        st.dataframe(price_summary.style.format({'평균가': '{:,.0f}', '최저가': '{:,.0f}', '최고가': '{:,.0f}'}), use_container_width=True)
    
    with t_c2:
        # [Table 4] 브랜드별 평균 가격 리스트
        st.subheader("브랜드별 평균 판매가 비교 (Table 4)")
        top_brands_avg = df_shop_filtered.groupby('brand')['lprice'].mean().sort_values(ascending=False).head(15).reset_index()
        top_brands_avg.columns = ['브랜드', '평균 판매가']
        st.dataframe(top_brands_avg.style.format({'평균 판매가': '{:,.0f}'}), use_container_width=True)

# ---------------------------------------------------------
# Tab 3: 소셜/블로그 분석
# ---------------------------------------------------------
with tab3:
    st.subheader("3. 소셜 미디어 및 키워드 마이닝")
    
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        # [Chart 5] 블로그 제목 핵심 키워드 (TF-IDF)
        st.subheader("블로그 제목 핵심 토픽 분석 (Chart 5)")
        blog_keywords = extract_keywords_pro(df_blog_filtered['title'].tolist())
        if not blog_keywords.empty:
            fig_key_bar = px.bar(
                blog_keywords, x='점수', y='단어', orientation='h',
                color='점수', title="블로그 제목 TF-IDF 상위 키워드",
                labels={'점수': 'TF-IDF 점수', '단어': '키워드'}
            )
            fig_key_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_key_bar, use_container_width=True)
            st.caption("**[해석]**: 블로그 제목에서는 '추천', '비교', '중요성'과 같은 키워드가 빈번하게 등장하며, 이는 소비자의 구매 여정 중 정보 탐색 단계에서 블로그가 미치는 영향력이 매우 크다는 점을 방증합니다.")

    with col_b2:
        # [Chart 6] 쇼핑 제목 핵심 키워드 (TF-IDF) - 추가 시각화
        st.subheader("상품명 마케팅 키워드 분석 (Chart 6)")
        shop_keywords = extract_keywords_pro(df_shop_filtered['title'].tolist())
        if not shop_keywords.empty:
            fig_shop_key = px.bar(
                shop_keywords, x='점수', y='단어', orientation='h',
                color='점수', color_continuous_scale='Reds',
                title="쇼핑 상품명 TF-IDF 상위 키워드"
            )
            fig_shop_key.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_shop_key, use_container_width=True)
            st.caption("**[해석]**: 상품명에는 'rTG', '식물성', '초임계' 등 제조 공법과 관련된 기술적 용어가 강조되고 있습니다. 이는 기능적 차별화를 통한 마케팅 전략이 시장의 주된 흐름임을 보여줍니다.")

    # [Table 5] 최근 블로그 포스트 리스트
    st.subheader("최근 블로그 포스트 분석 리스트 (Table 5)")
    df_blog_sorted = df_blog_filtered.sort_values('postdate', ascending=False)
    st.dataframe(df_blog_sorted[['keyword', 'title', 'bloggername', 'postdate', 'link']].head(50), use_container_width=True)

# ---------------------------------------------------------
# Tab 4: 데이터 익스플로러
# ---------------------------------------------------------
with tab4:
    st.subheader("4. 수집 데이터 상세 조회")
    
    # [Table 6] 상품 상세 데이터 필터링 테이블
    st.markdown("수집된 모든 상품 데이터를 직접 확인하고 검색할 수 있습니다.")
    search_q = st.text_input("상품명 내 검색", "")
    df_found = df_shop_filtered
    if search_q:
        df_found = df_shop_filtered[df_shop_filtered['title'].str.contains(search_q, case=False)]
    
    st.write(f"검색 결과: {len(df_found)}건")
    st.dataframe(df_found[['keyword', 'brand', 'title', 'lprice', 'mallName', 'category3', 'link']], use_container_width=True)
    st.caption("**[Table 6]**: 전체 상품 상세 데이터 리포트 테이블입니다.")

st.divider()
st.info(f"마지막 업데이트 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption("Powered by Antigravity AI Engine | Framework: Streamlit & Plotly")
