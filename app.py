import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import os
import json

# ========== 页面配置 ==========
st.set_page_config(
    page_title="智能数据分析Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 自定义 CSS ==========
st.markdown("""
<style>
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-title {
        color: #888;
        font-size: 1.1rem;
        margin-top: 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fc 0%, #e8ecf1 100%);
    }
</style>
""", unsafe_allow_html=True)

# ========== 预设 API 配置 ==========
PROVIDER_PRESETS = {
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "env_key": "DEEPSEEK_API_KEY"
    },
    "MiMo": {
        "base_url": "https://api.mimo.ai/v1",
        "models": ["mimo-v2.5-pro"],
        "env_key": "MIMO_API_KEY"
    },
    "自定义": {
        "base_url": "",
        "models": [],
        "env_key": ""
    }
}

# ========== 标题 ==========
st.markdown('<h1 class="main-title">📊 智能数据分析Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">上传数据文件，AI自动生成专业分析报告 · 支持CSV / Excel</p>', unsafe_allow_html=True)
st.markdown("---")

# ========== 侧边栏配置 ==========
with st.sidebar:
    st.header("⚙️ 配置面板")

    # API 服务商选择
    st.subheader("🔑 API 设置")
    provider = st.selectbox(
        "选择 API 服务商",
        list(PROVIDER_PRESETS.keys()),
        help="选择 DeepSeek 或 MiMo，或选自定义填入自己的 API"
    )
    preset = PROVIDER_PRESETS[provider]

    # API Key
    env_default = os.environ.get(preset["env_key"], "") if preset["env_key"] else ""
    api_key_input = st.text_input(
        "API Key",
        value=env_default,
        type="password",
        help=f"输入 {provider} 的 API Key"
    )

    # Base URL
    base_url = st.text_input(
        "API Base URL",
        value=preset["base_url"],
        help="API 接口地址"
    )

    # 模型选择
    if preset["models"]:
        model_name = st.selectbox(
            "选择模型",
            preset["models"],
            help="选择要使用的模型"
        )
    else:
        model_name = st.text_input(
            "模型名称",
            value="",
            placeholder="输入模型 ID",
            help="填写你要使用的模型 ID"
        )

    st.markdown("---")

    # 分析维度选择
    st.subheader("🎯 分析维度")
    analysis_dimensions = st.multiselect(
        "选择 AI 分析的侧重点",
        ["数据质量评估", "业务洞察", "异常值检测", "趋势分析", "相关性分析", "预测建议"],
        default=["数据质量评估", "业务洞察", "异常值检测"]
    )

    st.markdown("---")

    # 可视化设置
    st.subheader("🎨 可视化设置")
    chart_theme = st.selectbox(
        "图表主题",
        ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
        index=1
    )
    max_histograms = st.slider("直方图最大显示数", 1, 10, 5)

    st.markdown("---")
    st.caption("💡 基于 Streamlit + OpenAI 兼容 API")


# ========== 初始化客户端 ==========
@st.cache_resource
def get_client(api_key, base_url):
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key, base_url=base_url if base_url else None)
    except Exception as e:
        st.error(f"客户端初始化失败: {e}")
        return None

client = get_client(api_key_input, base_url)


# ========== 辅助函数 ==========
def get_data_summary(df):
    """生成数据摘要"""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    summary = {
        "行数": len(df),
        "列数": len(df.columns),
        "列名和类型": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "数值列": numeric_cols,
        "分类列": categorical_cols,
        "缺失值统计": df.isnull().sum().to_dict(),
        "缺失值比例": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }

    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        summary["数值列统计"] = desc.to_dict()

    if categorical_cols:
        cat_info = {}
        for col in categorical_cols[:10]:
            cat_info[col] = {
                "唯一值数量": int(df[col].nunique()),
                "前5高频值": df[col].value_counts().head(5).to_dict()
            }
        summary["分类列统计"] = cat_info

    summary["前3行样例"] = df.head(3).to_dict(orient="records")
    return summary


def call_ai_analysis(summary, user_question=None, dimensions=None, model="deepseek-chat"):
    """调用 AI 生成分析报告（OpenAI 兼容接口）"""
    if client is None:
        return "⚠️ API 客户端未配置，请在侧边栏填写 API Key 和 Base URL。"

    if not model or not model.strip():
        return "⚠️ 请在侧边栏选择或输入模型名称。"

    # 构造维度指令
    dim_prompt = ""
    if dimensions:
        dim_map = {
            "数据质量评估": "评估数据完整性、一致性、准确性，指出缺失值和异常格式问题",
            "业务洞察": "从数据中提取业务含义，发现潜在的商业机会和风险",
            "异常值检测": "识别数据中的异常值和离群点，分析可能原因",
            "趋势分析": "分析数据中的趋势和模式，预测未来走势",
            "相关性分析": "分析变量之间的关联关系，找出关键影响因素",
            "预测建议": "基于数据给出业务预测和行动建议"
        }
        dim_prompt = "\n".join([f"- {d}: {dim_map.get(d, '')}" for d in dimensions])

    # 宽数据集精简 prompt，避免超 token 上限
    n_cols = summary['列数']
    if n_cols > 20:
        # 只发 top-10 列的统计，分类列最多 5 个
        numeric_stats = summary.get('数值列统计', {})
        trimmed_numeric = dict(list(numeric_stats.items())[:10])
        cat_stats = summary.get('分类列统计', {})
        trimmed_cat = dict(list(cat_stats.items())[:5])
        sample_rows = summary.get('前3行样例', [])
        # 每行只保留 top-10 列
        if sample_rows and len(sample_rows[0]) > 10:
            keep_keys = list(trimmed_numeric.keys()) + list(trimmed_cat.keys())
            sample_rows = [{k: v for k, v in row.items() if k in keep_keys} for row in sample_rows]
        data_section = f"""【数据概览（宽数据集已精简）】
- 数据规模: {summary['行数']} 行 × {summary['列数']} 列
- 数值列 ({len(summary.get('数值列', []))}个): {summary.get('数值列', [])[:10]}{'...' if len(summary.get('数值列', [])) > 10 else ''}
- 分类列 ({len(summary.get('分类列', []))}个): {summary.get('分类列', [])[:5]}{'...' if len(summary.get('分类列', [])) > 5 else ''}
- 缺失值: {summary.get('缺失值统计', {})}
- 数值列统计 (Top 10): {json.dumps(trimmed_numeric, ensure_ascii=False, default=str)}
- 分类列统计 (Top 5): {json.dumps(trimmed_cat, ensure_ascii=False, default=str)}
- 样例行: {json.dumps(sample_rows, ensure_ascii=False, default=str)}"""
    else:
        data_section = f"""【数据概览】
- 数据规模: {summary['行数']} 行 × {summary['列数']} 列
- 数值列 ({len(summary.get('数值列', []))}个): {summary.get('数值列', [])}
- 分类列 ({len(summary.get('分类列', []))}个): {summary.get('分类列', [])}
- 缺失值: {summary.get('缺失值统计', {})}
- 缺失值比例(%): {summary.get('缺失值比例', {})}
- 数值列统计: {json.dumps(summary.get('数值列统计', {}), ensure_ascii=False, default=str)}
- 分类列统计: {json.dumps(summary.get('分类列统计', {}), ensure_ascii=False, default=str)}
- 前3行样例: {json.dumps(summary.get('前3行样例', []), ensure_ascii=False, default=str)}"""

    prompt = f"""你是一位资深数据分析师，擅长从数据中提取业务价值。请根据以下数据摘要，提供专业、深入、可执行的分析报告。

{data_section}

请按以下格式输出，每个维度用 ### 标题，内容用编号列表：
"""

    if dim_prompt:
        prompt += f"""
请重点分析以下维度：
{dim_prompt}
"""
    else:
        prompt += """
### 数据质量评估
1. ...
2. ...

### 业务洞察
1. ...
2. ...

### 异常值检测
1. ...
2. ...

### 后续分析建议
1. ...
2. ...
"""

    if user_question:
        prompt += f"\n\n用户额外提问：{user_question}\n请结合数据一并详细回答。"

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=3000,
            temperature=0.7,
            timeout=60,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content
        # 检查是否被截断
        if hasattr(response.choices[0], 'finish_reason') and response.choices[0].finish_reason == "length":
            result += "\n\n⚠️ 注意：报告因 token 上限被截断，建议缩小分析范围或分维度提问。"
        return result
    except Exception as e:
        import logging
        logging.exception("AI 分析调用失败")
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            return "❌ 请求过于频繁，请稍后再试。"
        elif "timeout" in error_msg.lower():
            return "❌ 请求超时，请稍后重试或减少数据量。"
        elif "auth" in error_msg.lower() or "401" in error_msg:
            return "❌ API Key 无效，请检查侧边栏配置。"
        elif "context_length" in error_msg.lower() or "token" in error_msg.lower():
            return "❌ 数据量过大超出模型上下文限制，请减少列数或行数后重试。"
        else:
            return f"❌ AI 分析失败: {error_msg}"


def create_distribution_plots(df, numeric_cols, max_n, theme):
    """生成分布图"""
    plots = []
    for col in numeric_cols[:max_n]:
        fig = px.histogram(
            df, x=col, title=f"📊 {col} 分布",
            nbins=30, template=theme,
            color_discrete_sequence=["#667eea"]
        )
        fig.update_layout(height=400, margin=dict(t=40, b=20))
        plots.append(fig)
    return plots


def create_box_plots(df, numeric_cols, theme):
    """生成箱线图"""
    plots = []
    for col in numeric_cols[:6]:
        fig = px.box(
            df, y=col, title=f"📦 {col} 箱线图",
            template=theme,
            color_discrete_sequence=["#764ba2"]
        )
        fig.update_layout(height=400, margin=dict(t=40, b=20))
        plots.append(fig)
    return plots


def create_correlation_heatmap(df, numeric_cols, theme):
    """生成相关性热力图"""
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    fig.update_layout(
        title="🔥 数值变量相关性热力图",
        height=500, template=theme,
        margin=dict(t=40, b=20)
    )
    return fig


def create_categorical_plots(df, categorical_cols, theme):
    """生成分类变量图表"""
    plots = []
    for col in categorical_cols[:4]:
        vc = df[col].value_counts().head(15)
        fig = px.bar(
            x=vc.index.astype(str), y=vc.values,
            title=f"📋 {col} 频次分布 (Top 15)",
            template=theme,
            labels={'x': col, 'y': '数量'},
            color_discrete_sequence=["#667eea"]
        )
        fig.update_layout(height=400, margin=dict(t=40, b=20))
        plots.append(fig)
    return plots


def create_pie_chart(df, categorical_cols, theme):
    """生成饼图"""
    plots = []
    for col in categorical_cols[:3]:
        if df[col].nunique() <= 10:
            vc = df[col].value_counts()
            fig = px.pie(
                values=vc.values, names=vc.index.astype(str),
                title=f"🥧 {col} 占比分布",
                template=theme,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(height=400, margin=dict(t=40, b=20))
            plots.append(fig)
    return plots


# ========== 主界面 ==========
uploaded_file = st.file_uploader(
    "📁 选择数据文件",
    type=["csv", "xlsx", "xls"],
    help="支持 CSV 和 Excel 格式"
)

if uploaded_file is not None:
    # ---- 文件大小检查 ----
    MAX_FILE_SIZE_MB = 50
    MAX_ROWS = 100_000
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ 文件过大（{file_size_mb:.1f}MB），上限 {MAX_FILE_SIZE_MB}MB。请裁剪数据后重试。")
        st.stop()

    # ---- 读取文件 ----
    try:
        if uploaded_file.name.endswith('.csv'):
            last_parser_error = None
            for enc in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin1']:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
                except pd.errors.ParserError as pe:
                    last_parser_error = pe
                    continue
            else:
                if last_parser_error:
                    st.error(f"CSV 文件格式有误: {last_parser_error}")
                else:
                    st.error("无法识别文件编码，请检查文件格式。")
                st.stop()
        else:
            df = pd.read_excel(uploaded_file)

        if len(df) > MAX_ROWS:
            st.warning(f"⚠️ 数据行数（{len(df):,}）超过上限 {MAX_ROWS:,}，已截取前 {MAX_ROWS:,} 行进行分析。")
            df = df.head(MAX_ROWS)

        st.success(f"✅ 成功加载 **{uploaded_file.name}** — {df.shape[0]} 行 × {df.shape[1]} 列")
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        st.stop()

    st.session_state['df'] = df
    st.session_state['summary'] = get_data_summary(df)
    summary = st.session_state['summary']

    # ========== Tabs 布局 ==========
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 数据预览", "📈 统计概览", "📉 数据可视化", "🤖 AI 分析"
    ])

    # ---- Tab 1: 数据预览 ----
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总行数", f"{summary['行数']:,}")
        with col2:
            st.metric("总列数", summary['列数'])
        with col3:
            missing_total = sum(summary['缺失值统计'].values())
            st.metric("缺失值总数", f"{missing_total:,}")
        with col4:
            st.metric("数值列数", len(summary.get('数值列', [])))

        st.markdown("---")
        st.dataframe(df.head(100), use_container_width=True)
        if len(df) > 100:
            st.info(f"仅显示前 100 行，共 {len(df)} 行数据。")

    # ---- Tab 2: 统计概览 ----
    with tab2:
        stat_col1, stat_col2 = st.columns(2)

        with stat_col1:
            st.subheader("📊 数值列统计")
            if summary.get('数值列统计'):
                stats_df = pd.DataFrame(summary['数值列统计'])
                st.dataframe(stats_df.round(2), use_container_width=True)
            else:
                st.info("没有数值列")

        with stat_col2:
            st.subheader("📋 分类列统计")
            if summary.get('分类列统计'):
                for col, info in summary['分类列统计'].items():
                    with st.expander(f"**{col}** — {info['唯一值数量']} 个唯一值"):
                        top_vals = info['前5高频值']
                        st.bar_chart(pd.Series(top_vals))
            else:
                st.info("没有分类列")

        # 缺失值分析
        st.subheader("🕳️ 缺失值分析")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            missing_pct = (missing / len(df) * 100).round(1)
            missing_df = pd.DataFrame({
                '缺失数量': missing,
                '缺失比例(%)': missing_pct
            })
            fig_missing = px.bar(
                missing_df, x=missing_df.index, y='缺失比例(%)',
                title="各列缺失值比例",
                color='缺失比例(%)',
                color_continuous_scale='Reds',
                template=chart_theme
            )
            fig_missing.update_layout(height=400)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("🎉 数据没有缺失值！")

    # ---- Tab 3: 数据可视化 ----
    with tab3:
        numeric_cols = summary.get('数值列', [])
        categorical_cols = summary.get('分类列', [])

        viz_tabs = st.tabs(["📊 分布图", "📦 箱线图", "🔥 相关性", "🔵 散点图", "📋 分类图", "🥧 饼图"])

        with viz_tabs[0]:
            if numeric_cols:
                plots = create_distribution_plots(df, numeric_cols, max_histograms, chart_theme)
                for fig in plots:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("没有数值列")

        with viz_tabs[1]:
            if numeric_cols:
                plots = create_box_plots(df, numeric_cols, chart_theme)
                for fig in plots:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("没有数值列")

        with viz_tabs[2]:
            if len(numeric_cols) >= 2:
                fig_corr = create_correlation_heatmap(df, numeric_cols, chart_theme)
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("需要至少 2 个数值列")

        with viz_tabs[3]:
            if len(numeric_cols) >= 2:
                sc_col1, sc_col2, sc_col3 = st.columns(3)
                with sc_col1:
                    x_col = st.selectbox("X 轴", numeric_cols, index=0, key="scatter_x")
                with sc_col2:
                    y_col = st.selectbox("Y 轴", numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")
                with sc_col3:
                    color_col = st.selectbox("颜色分组（可选）", ["无"] + categorical_cols, key="scatter_color")

                if x_col != y_col:
                    color_arg = None if color_col == "无" else color_col
                    fig_scatter = px.scatter(
                        df, x=x_col, y=y_col, color=color_arg,
                        title=f"🔵 {x_col} vs {y_col}",
                        template=chart_theme,
                        opacity=0.7
                    )
                    fig_scatter.update_layout(height=500)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("请选择不同的 X/Y 轴列")
            else:
                st.info("需要至少 2 个数值列")

        with viz_tabs[4]:
            if categorical_cols:
                plots = create_categorical_plots(df, categorical_cols, chart_theme)
                for fig in plots:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("没有分类列")

        with viz_tabs[5]:
            if categorical_cols:
                plots = create_pie_chart(df, categorical_cols, chart_theme)
                if plots:
                    for fig in plots:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("没有唯一值 ≤ 10 的分类列适合画饼图")
            else:
                st.info("没有分类列")

    # ---- Tab 4: AI 分析 ----
    with tab4:
        st.subheader("🤖 AI 智能分析报告")
        st.caption(f"当前使用: **{provider}** / **{model_name}**")

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(content)

        user_question = st.chat_input("💬 输入你想问 AI 关于这份数据的问题...")

        quick_col1, quick_col2, quick_col3 = st.columns(3)
        with quick_col1:
            if st.button("📊 一键生成完整报告", type="primary", use_container_width=True):
                st.session_state['_quick_question'] = "请生成完整的数据分析报告"
                st.rerun()
        with quick_col2:
            if st.button("🔍 找出数据中的异常", use_container_width=True):
                st.session_state['_quick_question'] = "请重点分析数据中的异常值和离群点"
                st.rerun()
        with quick_col3:
            if st.button("💡 给出业务建议", use_container_width=True):
                st.session_state['_quick_question'] = "请基于数据给出具体的业务建议和行动方案"
                st.rerun()

        # 合并 chat_input 和快捷按钮的输入
        quick_q = st.session_state.pop('_quick_question', None)
        if quick_q:
            user_question = quick_q

        if user_question:
            st.session_state.chat_history.append(("user", user_question))
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("🧠 AI 正在分析中..."):
                    report = call_ai_analysis(
                        summary,
                        user_question,
                        dimensions=analysis_dimensions,
                        model=model_name
                    )
                st.markdown(report)
                st.session_state.chat_history.append(("assistant", report))

                st.download_button(
                    label="📥 下载分析报告",
                    data=report,
                    file_name="ai_analysis_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        if st.session_state.chat_history:
            if st.button("🗑️ 清空对话历史"):
                st.session_state.chat_history = []
                st.rerun()

else:
    st.markdown("""
    <div style="text-align:center; padding: 60px 0;">
        <h2 style="color:#667eea;">👈 请上传数据文件开始分析</h2>
        <p style="color:#888;">支持 CSV、Excel (.xlsx/.xls) 格式</p>
    </div>
    """, unsafe_allow_html=True)

    intro_col1, intro_col2, intro_col3 = st.columns(3)
    with intro_col1:
        st.markdown("""
        ### 📊 自动统计
        - 行列数、缺失值
        - 数值列描述统计
        - 分类列频次分析
        """)
    with intro_col2:
        st.markdown("""
        ### 📉 智能可视化
        - 分布直方图、箱线图
        - 相关性热力图
        - 散点图、饼图
        """)
    with intro_col3:
        st.markdown("""
        ### 🤖 AI 深度分析
        - DeepSeek / MiMo 双模型
        - 数据质量评估
        - 多轮对话追问
        """)

st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#aaa; font-size:0.85rem;">'
    '智能数据分析Agent v2.0 · Streamlit + OpenAI 兼容 API · Made with ❤️</p>',
    unsafe_allow_html=True
)
