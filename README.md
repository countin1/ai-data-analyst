# AI 数据分析工具

上传 CSV/Excel 文件，自动出分析报告 + AI 深度分析。

## 功能

- 数据预览和描述性统计
- 缺失值分析
- 6 种交互式图表（直方图、箱线图、热力图、散点图、饼图、折线图）
- AI 深度分析（支持 DeepSeek / MiMo / OpenAI 兼容 API）

## 本地运行

```bash
pip install -r requirements.txt
export MIMO_API_KEY=your_key
streamlit run app.py
```

支持的环境变量：

- `MIMO_API_KEY` — MiMo API 密钥
- `DEEPSEEK_API_KEY` — DeepSeek API 密钥

## 部署到 Streamlit Cloud

1. 将代码推送到 GitHub 仓库
2. 去 [share.streamlit.io](https://share.streamlit.io) 连接仓库并部署
3. 在 App settings -> Secrets 中添加 API 密钥：

```toml
MIMO_API_KEY = "your-mimo-key"
DEEPSEEK_API_KEY = "your-deepseek-key"
```

Streamlit Cloud 会自动将 Secrets 中的键暴露为环境变量，应用侧边栏会自动读取已配置的密钥。

## 项目结构

```
.
├── app.py                  # 主应用
├── requirements.txt        # Python 依赖
├── test_data.csv           # 示例数据
├── .streamlit/
│   └── config.toml         # 主题配置
└── README.md
```
