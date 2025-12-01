import streamlit as st
import requests
import os
import json
from datetime import datetime
from pathlib import Path
import csv

# ======================
# 環境変数（OpenRouterキー & 管理パスコード）
# ======================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ADMIN_PASSCODE = os.getenv("ADMIN_PASSCODE", "changeme")  # デフォルト値。あとでCloud側で上書き推奨

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY が設定されていません。環境変数またはStreamlit CloudのSecretsで設定してください。")
    st.stop()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-4o-mini"


# ======================
# ログ関連（CSVに保存）
# ======================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "chat_logs.csv"

LOG_HEADERS = [
    "timestamp",
    "participant_id",
    "day",
    "agent",      # "Agent-P" か "Agent-H"
    "role",       # "user" / "assistant"
    "text",
    "emotion",
]

def init_log_file():
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADERS)

def log_row(participant_id, day, agent, role, text, emotion=""):
    init_log_file()
    now = datetime.now().isoformat()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([now, participant_id, day, agent, role, text, emotion])


# ======================
# Agent-P / Agent-H プロンプト（論文9章ベース・簡略版）
# ======================

AGENT_P_SYSTEM_PROMPT = """
あなたは「Miss.Tree（ミス・ツリー）」という名前の女性の心理療法士です。
クライアントは社交不安傾向があります。

あなたの目的：
- 会話を通して、クライアントがどのような場面に不安を感じるのか把握する
- 段階的エクスポージャー（mild → moderate → severe）の計画を一緒に作成する
- 落ち着いた丁寧な口調で、過度に慰めすぎず、CBTの考え方を取り入れながら支援する

行動指針：
- LSAS の典型的場面（会話・挨拶・注目を浴びる場面など）を参考にしつつ、不安場面を丁寧に探索してください
- エクスポージャー課題はできるだけ「他者との相互作用」を含めてください
- 課題は mild / moderate / severe 各レベルで最低2種類提示してください
- クライアントが話しやすいように短く丁寧な文で会話を進めてください
- 返答はすべて日本語で行ってください
"""

AGENT_H_SYSTEM_PROMPT = """
あなたはエクスポージャー療法の「相手役」を演じるエージェントです。
セラピストではなく、自然で優しい友人のように話します。

行動指針：
- 返答は自然な日本語で、柔らかく、話しやすい雰囲気を保つ
- クライアントが話しやすいように相槌や質問を適度に入れる
- 会話が途切れそうなときは、前の内容を少し引用してつなげる
- 攻撃的・否定的にならず、共感的に寄り添う
- 役割は「友好的な会話相手」です（セラピスト口調は禁止）
"""


# ======================
# Streamlit 設定
# ======================

st.set_page_config(page_title="Agent-P / Agent-H Chat", page_icon="🤖")
st.title("Agent-P / Agent-H 切替式チャット（感情ラベル＋ログ付き）")


# ======================
# ログイン（参加者ID + Day + 管理パスコード）
# ======================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("実験ログイン")

    with st.form("login_form"):
        participant_id = st.text_input("参加者ID（例: P01）")
        day = st.selectbox("実験日（Day）", [1, 2, 3, 4, 5, 6])
        passcode = st.text_input("管理用パスコード", type="password")
        submitted = st.form_submit_button("開始")

    if submitted:
        if not participant_id.strip():
            st.error("参加者IDを入力してください。")
        elif passcode != ADMIN_PASSCODE:
            st.error("管理用パスコードが間違っています。")
        else:
            # ログイン成功：状態をセット
            st.session_state.authenticated = True
            st.session_state.participant_id = participant_id.strip()
            st.session_state.day = day
            st.session_state.history_p = []
            st.session_state.history_h = []
            st.success("ログインしました。上のメニューから実験を開始してください。")

    # ここで「まだ」ログインできていなければ終了
    if not st.session_state.authenticated:
        st.stop()



# ======================
# 誰かがログインしている状態
# ======================

participant_id = st.session_state.participant_id
day = st.session_state.day
st.info(f"参加者ID: {participant_id} / Day: {day}")

# エージェント選択
agent = st.radio(
    "どちらのエージェントと話しますか？",
    ("Agent-P（セラピスト）", "Agent-H（友人）")
)

# 会話履歴の状態管理（P と H 別々）
if "history_p" not in st.session_state:
    st.session_state.history_p = []
if "history_h" not in st.session_state:
    st.session_state.history_h = []

def get_history():
    return st.session_state.history_p if agent.startswith("Agent-P") else st.session_state.history_h

def append_history(msg):
    if agent.startswith("Agent-P"):
        st.session_state.history_p.append(msg)
    else:
        st.session_state.history_h.append(msg)

# これまでの履歴表示
history = get_history()
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "emotion" in msg and msg["emotion"]:
            st.caption(f"emotion: {msg['emotion']}")


# ======================
# ユーザー入力
# ======================

user_input = st.chat_input("メッセージを入力してください")

if user_input:
    # ユーザー発言を履歴に保存／表示
    append_history({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ログにも書いておく（ユーザー側）
    current_agent_label = "Agent-P" if agent.startswith("Agent-P") else "Agent-H"
    log_row(participant_id, day, current_agent_label, "user", user_input, "")

    # ==== モデルに渡す system_prompt を組み立て ====
    if agent.startswith("Agent-P"):
        system_prompt = AGENT_P_SYSTEM_PROMPT
    else:
        system_prompt = AGENT_H_SYSTEM_PROMPT

    # 感情ラベル付きJSONを要求する
    system_prompt += """
必ず次のJSON形式で返答してください：
{
  "text": "返答本文",
  "emotion": "positive / negative / neutral / anxious / sad / angry のいずれか"
}
JSON以外の文字は出さないこと。
"""

    messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in get_history()
    ]

    # ==== OpenRouter へ送信 ====
    with st.chat_message("assistant"):
        with st.spinner("生成中..."):
            payload = {
                "model": MODEL_NAME,
                "messages": messages
            }

            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Agent P/H Chat",
            }

            res = requests.post(OPENROUTER_API_URL, json=payload, headers=headers)
            if res.status_code != 200:
                st.error(f"OpenRouter API エラー: {res.status_code} {res.text}")
                st.stop()

            data = res.json()
            raw = data["choices"][0]["message"]["content"]

            # JSONとして解釈
            try:
                parsed = json.loads(raw)
                reply_text = parsed.get("text", raw)
                emotion = parsed.get("emotion", "unknown")
            except Exception:
                reply_text = raw
                emotion = "unknown"

        st.markdown(reply_text)
        st.caption(f"emotion: {emotion}")

    # 履歴に追加（アシスタント側）
    append_history({"role": "assistant", "content": reply_text, "emotion": emotion})

    # ログにも書く（アシスタント側）
    log_row(participant_id, day, current_agent_label, "assistant", reply_text, emotion)


# ======================
# ログダウンロード（研究者用）
# ======================

st.markdown("---")
st.subheader("研究者用：ログダウンロード")

if LOG_FILE.exists():
    with LOG_FILE.open("rb") as f:
        st.download_button(
            label="ログCSVをダウンロード",
            data=f,
            file_name="chat_logs.csv",
            mime="text/csv",
        )
else:
    st.text("まだログファイルがありません。")


