import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openai
import os

# --- 页面基本配置 ---
st.set_page_config(
    page_title="AI 股票诊断看板",
    page_icon="📈",
    layout="wide"
)

# --- 辅助指标计算函数（免第三方重型库） ---
def calculate_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# --- 侧边栏：参数与免费模型配置 ---
st.sidebar.header("⚙️ 参数配置")

ticker_input = st.sidebar.text_input("股票代码 (Ticker)", value="NVDA").upper().strip()
time_period = st.sidebar.selectbox("历史周期", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("K线级别", ["1d", "1wk"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI 模型配置 (免费 Gemini)")
api_key = st.sidebar.text_input("Gemini API Key (AIzaSy...)", type="password", value=os.getenv("GEMINI_API_KEY", ""))
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model_name = st.sidebar.selectbox(
    "模型选择", 
    ["gemini-1.5-flash", "gemini-1.5-pro"], 
    index=0
)

user_rumors = st.sidebar.text_area("补充小道消息 / 市场舆情 (选填)", placeholder="例如：论坛传闻大客户推迟订单；机构资金近期大幅流入等...")

# --- 数据获取函数 ---
@st.cache_data(ttl=300)
def load_stock_data(ticker: str, period: str, interval: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        return None, None
    df = calculate_indicators(df)
    return df, stock.info

# --- 页面主区域 ---
st.title("📊 AI 交互式股票看板与诊断中心")

if not ticker_input:
    st.warning("请输入有效的股票代码。")
    st.stop()

with st.spinner("正在获取市场数据..."):
    df, info = load_stock_data(ticker_input, time_period, interval)

if df is None or df.empty:
    st.error(f"无法获取代码 `{ticker_input}` 的数据，请检查代码拼写或交易所后缀（美股如 NVDA，港股如 0700.HK，马股如 1023.KL）。")
    st.stop()

# --- 顶部核心指标展示 ---
latest_close = df['Close'].iloc[-1]
prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest_close
change = latest_close - prev_close
pct_change = (change / prev_close) * 100

pe_val = info.get("trailingPE", "N/A")
pb_val = info.get("priceToBook", "N/A")
market_cap = info.get("marketCap", 0)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("最新收盘价", f"${latest_close:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
col2.metric("市盈率 PE (TTM)", f"{pe_val:.2f}" if isinstance(pe_val, (int, float)) else "N/A")
col3.metric("市净率 PB", f"{pb_val:.2f}" if isinstance(pb_val, (int, float)) else "N/A")
col4.metric("14日 RSI", f"{df['RSI'].iloc[-1]:.2f}" if not df['RSI'].isna().iloc[-1] else "N/A")
col5.metric("市值", f"${market_cap / 1e9:.2f}B" if market_cap else "N/A")

st.markdown("---")

# --- 交互式 K 线与成交量图表 ---
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05, 
    row_heights=[0.75, 0.25]
)

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="K线",
    increasing_line_color='#26a69a',
    decreasing_line_color='#ef5350'
), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index,
