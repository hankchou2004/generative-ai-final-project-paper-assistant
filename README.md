# 生成式ai期末專案-智慧論文助手

一個基於 RAG（Retrieval-Augmented Generation）的論文閱讀助手，能協助使用者針對學術論文進行問答、檢索相關內容，並提供較可靠的參考依據。

---

## 專案介紹

本專案是基於原始的 PaperHelper 專案進行修改與延伸開發：

Original Project：
https://github.com/JerryYin777/PaperHelper

在原有架構基礎上，我針對自己的研究與學習需求進行調整與功能擴充，包含：

- PDF 論文嵌入與向量化
- RAG Fusion 檢索方式
- Gemini / OpenAI API 整合
- 論文問答功能
- 回答評估與分數分析頁面
- Streamlit 網頁介面
- 參考內容檢索與生成

本系統透過 Retrieval-Augmented Generation（RAG）技術，降低大型語言模型產生幻覺（Hallucination）的情況，並提升回答與論文內容的相關性。

---

## 功能特色

- 上傳並解析 PDF 論文
- 論文內容向量化與語意搜尋
- RAG 問答系統
- RAG Fusion 檢索
- 顯示相關參考內容
- Streamlit 互動介面
- 回答品質評估功能

---

## 系統流程

系統主要流程如下：

1. 載入 PDF 論文
2. 文字切割（Text Chunking）
3. Embedding 向量生成
4. 建立向量資料庫（FAISS / ChromaDB）
5. 使用 RAG 或 RAG Fusion 進行檢索
6. 將檢索結果交給 LLM 生成回答

---

## Demo

### 系統介面

> 可在此放置系統截圖

```markdown
![demo](image/demo.png)
```

---

## 專案結構

```text
PaperHelper/
│
├── app.py                  # Streamlit 主程式
├── embed_pdf.py            # PDF embedding 與向量化
├── llm_helper.py           # LLM 問答處理
├── agent_helper.py         # Retrieval / Agent logic
├── eval_page.py            # 回答評估頁面
├── requirements.txt
├── README.md
└── pdf/                    # PDF 論文資料夾
```

---

## 安裝方式

### 1. Clone 專案

```bash
git clone https://github.com/hankchou2004/paper-helper.git
cd paper-helper
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 設定 API Key

建立 `.streamlit/secrets.toml`

```toml
OPENAI_API_KEY="your_api_key"
GOOGLE_API_KEY="your_api_key"
```

### 4. 啟動系統

```bash
streamlit run app.py
```

---

## 使用方式

1. 上傳 PDF 論文
2. 建立 Embedding
3. 選擇 RAG 或 RAG Fusion
4. 輸入問題
5. 系統回傳回答與相關參考內容

---

## 技術使用

- Python
- Streamlit
- LangChain
- FAISS
- ChromaDB
- OpenAI API
- Google Gemini API
- Sentence Transformers

---

## 評估方式

本系統提供回答品質評估功能，包含：

- ROUGE Score
- BLEU Score
- BERTScore
- Exact Match

可用於分析 RAG 回答品質與檢索效果。

---

## 注意事項

- 請自行設定 API Key
- 不建議將 `.env` 或 `secrets.toml` 上傳至 GitHub
- 首次建立 embedding 可能需要較長時間
- 建議先完成 PDF embedding 再進行問答

---

## Reference

Original PaperHelper Project：

https://github.com/JerryYin777/PaperHelper
