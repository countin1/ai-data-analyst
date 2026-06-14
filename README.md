# AI 数据分析工具

上传 CSV/Excel 文件，自动出分析报告 + AI 深度分析。

## 功能

- 数据预览和描述性统计
- 缺失值分析
- 6 种交互式图表（直方图、箱线图、热力图、散点图、饼图、折线图）
- AI 深度分析（支持 DeepSeek / MiMo / OpenAI 兼容 API）

## 使用

```bash
pip install -r requirements.txt
export MIMO_API_KEY=your_key
streamlit run app.py
```

## 部署到 Streamlit Cloud

1. Fork 本仓库
2. 去 [share.streamlit.io](https://share.streamlit.io) 部署
3. 在 Secrets 中添加 `MIMO_API_KEY`
