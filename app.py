import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
import openai
import os

# --- 页面基本配置 ---
st.set_page_config(
    page_title="AI 股票诊断看板",
    page_icon="📈",
    layout="wide"
)

# --- 侧边栏：参数与免费模型配置 ---
st.sidebar.header("⚙️ 参数配置")

ticker_input = st.sidebar.text_input("股票代码 (Ticker)", value="NVDA").upper().strip()
time_period = st.sidebar.selectbox("历史周期", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("K线级别", ["1d", "1wk"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI 模型配置 (免费 Groq)")
api_key = st.sidebar.text_input("Groq API Key (gsk_...)", type="password", value=os.getenv("GROQ_API_KEY", ""))
base_url = st.sidebar.text_input("API Base URL", value="https://api.groq.com/openai/v1")
model_name = st.sidebar.selectbox(
    "模型选择", 
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "deepseek-r1-distill-llama-70b"], 
    index=0
)

# 录入小道消息/舆情
user_rumors = st.sidebar.text_area("补充小道消息 / 市场舆情 (选填)", placeholder="例如：论坛传闻大客户推迟订单；某机构近期持续建仓等...")

# --- 数据获取函数 ---
@st.cache_data(ttl=300)
def load_stock_data(ticker: str, period: str, interval: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        return None, None
    
    df['MA20'] = ta.sma(df['Close'], length=20)
    df['MA50'] = ta.sma(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
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

fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA 20', line=dict(color='orange', width=1.2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA 50', line=dict(color='blue', width=1.2)), row=1, col=1)

vol_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['Close'], df['Open'])]
fig.add_trace(go.Bar(
    x=df.index,
    y=df['Volume'],
    name="成交量",
    marker_color=vol_colors,
    opacity=0.8
), row=2, col=1)

fig.update_layout(
    title=f"{ticker_input} 行情与成交量走势",
    xaxis_rangeslider_visible=False,
    height=550,
    margin=dict(l=40, r=40, t=40, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- AI 一键诊断模块 ---
st.markdown("---")
st.subheader("🧠 AI 投资与技术面综合研报")

if st.button("🚀 生成一键诊断报告", type="primary"):
    if not api_key:
        st.warning("请在左侧边栏输入免费申请的 Groq API Key (gsk_...) 后再运行诊断。")
        st.stop()

    with st.spinner("AI 正在解析量价动能、财务基本面与市场情绪..."):
        recent_vol_avg = df['Volume'].iloc[-6:-1].mean()
        latest_vol = df['Volume'].iloc[-1]
        vol_surge = latest_vol / recent_vol_avg if recent_vol_avg > 0 else 1.0

        client = openai.OpenAI(api_key=api_key, base_url=base_url.strip())

        prompt = f"""
你是一名严谨的股票量化与基本面分析专家。请根据以下提供的客观数据对代码【{ticker_input}】进行深度研判。

【行情与技术面指标】
- 最新收盘价: {latest_close:.2f}
- 今日涨跌幅: {pct_change:+.2f}%
- 14日 RSI 强弱指标: {df['RSI'].iloc[-1]:.2f} (判断是否超买/超卖)
- 均线位置: 20日均线 ({df['MA20'].iloc[-1]:.2f}), 50日均线 ({df['MA50'].iloc[-1]:.2f})

【资金与买卖量异动】
- 今日成交量: {latest_vol:,.0f}
- 成交量较前5日均量比值: {vol_surge:.2f}倍 (是否属于放量突破或缩量整理)

【基本面财务概要】
- PE (TTM): {pe_val}
- PB: {pb_val}
- 营收增长率: {info.get("revenueGrowth", "N/A")}
- 净利润率: {info.get("profitMargins", "N/A")}

【输入的小道消息与舆情摘要】
{user_rumors if user_rumors.strip() else "暂无用户输入的额外舆情。"}

请输出一份排版清晰的研报，包含以下结构：
1. **量价与技术面研判**：结合 RSI 与均线支撑阻力，判断目前多空动能及放量/缩量状态。
2. **财务与估值评价**：结合行业常态评估估值合理性与财务健康度。
3. **消息面与情绪分析**：若提供了小道消息，客观剖析传闻真实性风险及其对散户情绪的潜在冲击（若无则简评市场共识）。
4. **操作策略与风控提示**：明确给出短期阻力位、支撑位及严格的止损参考建议。
"""

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            st.success("诊断生成完毕！")
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"调用 AI 接口出错: {e}")
