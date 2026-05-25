from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure CWD is project root so relative paths (ChromaDB, outputs/) resolve correctly
os.chdir(Path(__file__).parent.parent)

load_dotenv(encoding="utf-8-sig")

st.set_page_config(
    layout="wide",
    page_title="KGI FinLLM 法規智能查詢系統",
    page_icon="⚖",
)

# ---------------------------------------------------------------------------
# Heavy resource — cached across reruns via Streamlit's process-level cache
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_generator():
    from src.generation.generator import generate  # noqa: PLC0415
    return generate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONF_COLORS = {"high": "#22c55e", "medium": "#eab308", "low": "#ef4444"}
_CONF_LABELS = {"high": "高信心", "medium": "中等信心", "low": "低信心"}
_TYPE_COLORS = {
    "grounded": "#22c55e",
    "hallucination_risk": "#ef4444",
    "over_refusal": "#f97316",
}
_TYPE_LABELS = {
    "grounded": "有根據",
    "hallucination_risk": "幻覺風險",
    "over_refusal": "過度拒答",
}


def _badge(label: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:4px;font-weight:bold;font-size:0.85em">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, object] = {
    "prefill": "",
    "auto_submit": False,
    "last_query": "",
    "last_answer": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("KGI FinLLM\n法規智能查詢系統")
    st.divider()
    st.markdown("**模型資訊**")
    st.markdown("- 模型：`claude-sonnet-4-6`")
    st.markdown("- 檢索策略：Hybrid RRF (BM25 + Vector)")
    st.markdown("- 已索引文件片段：**135**")
    st.divider()
    project = os.getenv("LANGCHAIN_PROJECT", "finllm-kgi-demo")
    st.markdown(f"[LangSmith 專案連結](https://smith.langchain.com)")
    st.caption(f"專案：{project}")

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns([3, 4, 3])

should_generate = False
generate_query = ""

# --- Col 1: Query input ---
with col1:
    st.subheader("查詢輸入")
    st.markdown("**範例問題**")
    examples = [
        "銀行在什麼情況下可以暫停帳戶？",
        "KYC需蒐集哪些資料？",
        "可疑交易申報時限為何？",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.prefill = ex
            st.session_state.auto_submit = True
            st.rerun()

    st.divider()
    user_input = st.text_area(
        "請輸入法規查詢問題",
        value=st.session_state.prefill,
        height=120,
        placeholder="例如：銀行法對同一人持股有何限制？",
    )
    submitted = st.button("查詢", type="primary", use_container_width=True)

    if submitted and user_input.strip():
        should_generate = True
        generate_query = user_input.strip()
        st.session_state.prefill = ""
    elif st.session_state.auto_submit and st.session_state.prefill:
        should_generate = True
        generate_query = st.session_state.prefill
        st.session_state.auto_submit = False
        st.session_state.prefill = ""

# --- Generation (between columns) ---
if should_generate and generate_query != st.session_state.last_query:
    with st.spinner("查詢中，請稍候..."):
        fn = _load_generator()
        st.session_state.last_answer = fn(generate_query)
        st.session_state.last_query = generate_query

answer = st.session_state.last_answer

# --- Col 2: Answer ---
with col2:
    st.subheader("回答")
    if answer is None:
        st.info("請在左側輸入查詢問題，或點選範例問題。")
    else:
        conf_badge = _badge(
            _CONF_LABELS[answer.confidence], _CONF_COLORS[answer.confidence]
        )
        type_badge = _badge(
            _TYPE_LABELS[answer.response_type], _TYPE_COLORS[answer.response_type]
        )
        st.markdown(
            f"{conf_badge}&nbsp;&nbsp;{type_badge}", unsafe_allow_html=True
        )

        st.metric(
            label="正規化風險分數",
            value=f"{answer.normalized_risk:.2f}",
            help="priority_weight × (1 - confidence_score) × feedback_weight",
        )
        st.divider()
        st.markdown(answer.answer)
        st.info(answer.risk_note)

# --- Col 3: Evidence ---
with col3:
    st.subheader("依據來源")
    if answer is None:
        st.caption("查詢後顯示檢索來源。")
    else:
        if answer.sources:
            for i, src in enumerate(answer.sources, start=1):
                title = f"[{i}] {src.doc_name}"
                if src.article:
                    title += f" — {src.article}"
                with st.expander(title):
                    st.markdown(f"**頁面：** {src.page}")
                    st.markdown("**摘錄：**")
                    st.info(src.supporting_text)
            st.markdown(f"已檢索 **{len(answer.sources)}** 個文件片段")
        else:
            st.caption("無法找到相關文件來源。")

        if answer.missing_info:
            st.warning("**缺少資訊：** " + "；".join(answer.missing_info))

# ---------------------------------------------------------------------------
# Bottom expander: Eval benchmark results
# ---------------------------------------------------------------------------

with st.expander("評估基準測試結果（面試展示用）"):
    csv_path = Path("outputs/eval_results.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        display_cols = [
            c
            for c in [
                "inputs.query",
                "feedback.correctness",
                "feedback.citation_groundedness",
                "feedback.localization",
                "feedback.hallucination_risk",
                "feedback.usefulness",
            ]
            if c in df.columns
        ]
        if display_cols:
            df_display = df[display_cols].rename(
                columns={
                    "inputs.query": "查詢問題",
                    "feedback.correctness": "正確性",
                    "feedback.citation_groundedness": "引用根據性",
                    "feedback.localization": "本地化",
                    "feedback.hallucination_risk": "幻覺風險",
                    "feedback.usefulness": "有用性",
                }
            )
            num_cols = [c for c in df_display.columns if c != "查詢問題"]
            col_config: dict = {
                "查詢問題": st.column_config.TextColumn(width="large"),
            }
            for nc in num_cols:
                col_config[nc] = st.column_config.NumberColumn(format="%.2f")
            st.dataframe(df_display, column_config=col_config, use_container_width=True)
        else:
            st.warning("CSV 欄位格式不符，請重新執行 scripts/run_eval.py。")
    else:
        st.info("尚未產生評估結果。請先執行：`.venv\\Scripts\\python -m scripts.run_eval`")
