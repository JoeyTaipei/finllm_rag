# 台灣金融法規 RAG 智能查詢系統

**線上 Demo：** [finllmrag-yv3.streamlit.app](https://finllmrag-yv3.streamlit.app/)

基於檢索增強生成（RAG）技術，針對台灣金融法規合規查詢場景設計的智能問答系統。

## 系統特色

### 解決的核心問題
- **通用 LLM 在地化不足**：專門針對台灣金管會法規、洗錢防制相關法令進行索引
- **幻覺風險控制**：系統資料不足時明確告知，不捏造條文或法規內容
- **可量化評測**：建立金融法規問答 Benchmark，提供信心分數與正規化風險值

### 技術架構

```
使用者查詢
    ↓
查詢轉換（意圖分析）
    ↓
混合檢索（BM25 關鍵字 + 向量語意搜尋）
    ↓
RRF Reranker（倒數排名融合）
    ↓
結構化生成（含信心分數、來源引用、風險評估）
    ↓
LangSmith 追蹤與評測
```

## 技術棧

| 類別 | 工具 |
|---|---|
| 框架 | LangChain LCEL |
| 向量資料庫 | ChromaDB |
| 關鍵字搜尋 | BM25 (rank-bm25) |
| Embedding | sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| 生成模型 | Claude Sonnet (Anthropic API) |
| 追蹤評測 | LangSmith |
| 介面 | Streamlit |
| 資料解析 | pypdf + jieba 中文斷詞 |

## 專案結構

```
taiwan-finllm-rag/
├── src/
│   ├── ingestion/        # PDF 載入、切分、向量化
│   ├── retrieval/        # 混合檢索、RRF Reranker
│   └── generation/       # 結構化生成、Pydantic Schema
├── app/
│   └── streamlit_app.py  # 展示介面
├── scripts/
│   ├── run_eval.py       # LangSmith LLM-as-judge 評測
│   └── evaluate_retrieval.py  # 檢索指標評測
├── data/
│   ├── raw_pdfs/         # 原始法規 PDF
│   └── eval/             # Benchmark 資料集
├── tests/                # pytest 測試
└── outputs/              # 評測結果輸出
```

## 快速開始

### 環境設定

```powershell
# 建立虛擬環境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安裝套件
python -m pip install -r requirements.txt
```

### 設定 API Keys

複製 `.env.example` 為 `.env` 並填入：

```
ANTHROPIC_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=finllm-kgi-demo
LANGCHAIN_TRACING_V2=true
```

### 執行 PDF 索引

將法規 PDF 放入 `data/raw_pdfs/`，然後執行：

```powershell
python -m src.ingestion.pdf_loader
```

### 啟動應用

```powershell
streamlit run app/streamlit_app.py
```

### 執行評測

```powershell
# 檢索品質評測
python scripts/evaluate_retrieval.py

# LLM-as-judge 評測
python -m scripts.run_eval
```

### 執行測試

```powershell
pytest tests/ -v
```

## 評測結果

基於台灣洗錢防制法及銀行法的 5 題 Benchmark（含 1 題 Negative test）：

| 查詢問題 | 正確性 | 引用根據性 | 本地化 | 幻覺風險 | 有用性 |
|---|---|---|---|---|---|
| KYC 客戶資料蒐集 | 0.80 | 0.90 | 0.90 | 0.80 | 0.70 |
| 同一人持股比例限制 | 0.90 | 0.70 | 1.00 | 0.70 | 0.90 |
| 可疑交易申報時限 | 0.50 | 0.70 | 0.90 | 0.60 | 0.60 |
| 帳戶暫停功能條件 | 0.90 | 0.70 | 1.00 | 0.60 | 0.80 |
| 信用卡利率3%（Negative） | 0.90 | 0.90 | 0.90 | 0.80 | 0.80 |

> Q3（可疑交易申報時限）正確性較低，原因為相關法規在現有索引中 chunk 數量不足（7 chunks），系統正確觸發 `over_refusal`，未捏造時限數字。

## 輸出 Schema

每次查詢回傳結構化 JSON：

```json
{
  "answer": "繁體中文回答",
  "sources": [
    {
      "doc_name": "洗錢防制法",
      "article": "第7條",
      "page": 2,
      "supporting_text": "引用原文片段（最多150字）"
    }
  ],
  "confidence": "high | medium | low",
  "missing_info": ["資料不足時說明缺少什麼"],
  "response_type": "grounded | hallucination_risk | over_refusal",
  "normalized_risk": 0.20,
  "risk_note": "此回答僅供內部法遵或授信分析初步參考，實際決策仍需依最新法規與內部規範確認。"
}
```

## 設計理念

> 準確率不可能 100%，目標是建立**可以校正的系統**。

- **Shift Left**：LangSmith 追蹤從開發第一天就嵌入，不是事後補上
- **可駕馭的不確定性**：系統明確區分「知道」、「不確定」、「資料不足」三種狀態
- **可量化治理**：每次回答附 `normalized_risk` 分數，供後續人工審核排序

## 未來擴展方向

- [ ] 增加更多金融法規文件（保險法、證券交易法）
- [ ] 加入使用者回饋機制更新 normalized_risk
- [ ] 擴展為 Agentic 架構，支援跨文件多步推理
- [ ] 與內部監控平台整合（如 MLflow、Azure Monitor）

## 授權

MIT License
