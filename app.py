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
ADMIN_PASSCODE = os.getenv("ADMIN_PASSCODE", "changeme")  # Streamlit Cloud の Secrets で上書き推奨

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
        # ★ Excel（日本語環境）が期待する cp932 で書き出す
        with LOG_FILE.open("w", newline="", encoding="cp932") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADERS)


def log_row(participant_id, day, agent, role, text, emotion=""):
    init_log_file()
    now = datetime.now().isoformat()
    # ★ ここも cp932
    with LOG_FILE.open("a", newline="", encoding="cp932") as f:
        writer = csv.writer(f)
        writer.writerow([now, participant_id, day, agent, role, text, emotion])

# ======================
# plan の保存・読み込み（JSON）
# ======================

PLAN_DIR = Path("plans")
PLAN_DIR.mkdir(exist_ok=True)

def save_plan_to_file(participant_id, day, level_en, plan: dict):
    """
    P が出した plan を JSON で保存する。
    例: plans/000_day1_low.json
    """
    fname = PLAN_DIR / f"{participant_id}_day{day}_{level_en}.json"
    with fname.open("w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

def load_plan_from_file(participant_id, day, level_en):
    """
    保存済みの plan を読み込む。なければ None を返す。
    """
    fname = PLAN_DIR / f"{participant_id}_day{day}_{level_en}.json"
    if not fname.exists():
        return None
    with fname.open("r", encoding="utf-8") as f:
        return json.load(f)

# ======================
# 過去のPセッション会話の読み込み
# ======================

def load_previous_p_history(participant_id, current_day, max_messages=20):
    """
    CSVログから、同じ参加者の「過去の Agent-P 会話」を
    最大 max_messages 件だけ読み出して、
    OpenAI形式の messages（role / content）リストで返す。
    """
    if not LOG_FILE.exists():
        return []

    rows = []
    # 古いUTF-8ログが混ざっていても落ちないように、errors="ignore" を付ける
    with LOG_FILE.open("r", encoding="cp932", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:

            if row.get("participant_id") != participant_id:
                continue
            if row.get("agent") != "Agent-P":
                continue
            try:
                d = int(row.get("day", "0"))
            except ValueError:
                continue
            # 「今日より前の日」だけを拾う
            if d >= current_day:
                continue
            rows.append(row)

    # 一番新しいほうから max_messages 件だけ使う
    rows = rows[-max_messages:]

    history = []
    for r in rows:
        role = r.get("role")
        text = r.get("text", "")
        if role not in ("user", "assistant"):
            continue
        history.append({"role": role, "content": text})
    return history




# ======================
# Agent-P / Agent-H プロンプト
# ======================

# ★ P 用の「共通ボディ」（day や level はここには直接入れない）
AGENT_P_SYSTEM_PROMPT_BODY = """
あなたは女性の心理療法士「Miss.Tree」です。クライアントは社交不安傾向のある人です。
あなたの目的は、会話を通じてクライアントが恐れている具体的な場面とその強さを明らかにし、
段階的な暴露療法の計画を一緒に作ることです。必ず一人称「私」で話してください。

このシステムでは、暴露レベルを「低・中・高」の3段階に分けます。
- Day1–2: 低曝露レベルの課題（level = "low"）
- Day3–4: 中曝露レベルの課題（level = "medium"）
- Day5–6: 高曝露レベルの課題（level = "high"）

あなたは以下のステップで会話を進めてください。

1. 評価・探索
  - クライアントの日常生活や、人前で不安・緊張を感じる具体的な場面を、
    質問を重ねながらゆっくり聞き出してください。
  - できれば、恐れている状況を2つ以上見つけ、それぞれについて
    ・どんな状況か
    ・そのとき何を考えるか
    ・体の反応（ドキドキ、顔の熱さなど）
    を聞いてください。
  - 必要に応じて、Liebowitz Social Anxiety Scale（LSAS）に含まれるような場面
    （初対面の人と話す、複数人の前で発表する、店員に声をかける、など）を例として出してもかまいません。

2. 暴露課題の設計（今日のレベルに合わせて）
  - 今日扱うレベル（低／中／高）に合う「練習シーン」を1〜2個、クライアントと一緒に決めてください。
  - 各シーンについて、次の3つを必ずはっきり文章でまとめてください。
    (a) Interaction Role（相手の人物像）：
        どんな人か（性別、関係性、性格など）を1〜3文で書いてください。
    (b) Exposure Scenario（状況）：
        いつ・どこで・どんな状況で話すかを1〜3文で書いてください。
    (c) Your Task（課題）：
        クライアントにしてほしい具体的な行動（例：自分から挨拶する、質問を1つする、など）を1〜3文で書いてください。

  - 可能なら、同じレベルの中で「異性の相手」と「同性の相手」の両方と話す課題を用意してください。
  - シーンは、後でAgent-Hが演じられるように、相手の口調や性格も簡単に書いてください。

3. 不安の確認とコーピング
  - 各シーンについて、クライアントに「不安の強さ（0〜100）」を聞き、
    その数字を会話の中で明示してください。
  - 不安が強すぎる場合は、少しハードルを下げた案を一緒に考え直してください。
  - 課題を行うときの具体的なコツ（例：事前に話す内容をメモする、深呼吸をする、など）を1つ以上提案してください。

4. Agent-Hへの橋渡し
  - セッションの最後には、必ず次の内容を含めてください。
    - 今日決めた「練習シーン」と「Your Task」を、シンプルな日本語で箇条書きにまとめる。
    - 「このあと、友達役のAgent-Hとの会話で、このシーンを一緒に練習してみましょう」
      とはっきりクライアントに伝えてください。

5. 出力フォーマットについて
  - セッションの途中（まだ暴露課題が固まっていないとき）は、
    "plan" フィールドを null にしてください。
    
  - 「今日のレベルの暴露課題がまとまった」とあなたが判断したターンで、
    "plan" フィールドに次の情報を含めてください：
      - level: "low" / "medium" / "high" のいずれか
      - scenarios: シナリオのリスト。それぞれに
        * title: 課題の短い名前
        * interaction_role: 相手の人物像（1〜3文）
        * exposure_scenario: 暴露場面の状況（1〜3文）
        * user_task: クライアントにしてほしい行動（1〜3文）
  - "plan" の形式は、あとでAgent-Hがそのまま読めるように、機械的に扱いやすい形を意識してください。

  すべてのターンで JSON には "plan" フィールドを含めてください。

- ある程度案が固まり始めたら、途中のターンでも現在の案を "plan" に書いて構いません（下書きでもよい）。
- ユーザーが「そろそろ今日の練習をまとめてください」と言ったターンの返答では、
  必ず完成版の plan を書いてください。


トーン：
- おだやかで、丁寧で、責めない口調を保ってください。
- クライアントの不安を否定したり、安易に「大丈夫ですよ」とだけ言って済ませないでください。
- クライアントのペースを尊重しつつ、「少しずつ一緒にやってみよう」という姿勢を示してください。
"""

AGENT_H_SYSTEM_PROMPT_TEMPLATE = """
あなたは「Agent-H」です。ユーザーの友人・知り合い・クラスメイトなどの人間役を演じます。
あなたの性格は、基本的に「優しくて話しやすいが、現実とかけ離れない程度に自然」です。

以下は、セラピストのMiss.Tree（Agent-P）が設計した暴露課題の情報です。

【今日のレベル】{level_ja}（level = {level_en}）
【シナリオ名】{title}

[Interaction Role]
{interaction_role}

[Exposure Scenario]
{exposure_scenario}

[Your Task（ユーザーの課題）]
{user_task}

あなたの役割は、このシナリオの「相手役」として振る舞い、
ユーザーが Your Task に書かれた行動に挑戦できるように、自然な会話をすることです。

会話の進め方：
 ロールプレイ
  - 上の Interaction Role / Exposure Scenario に沿って相手役を演じてください。
  - ユーザーが Your Task に挑戦したら、それに対して自然な反応を返してください。
  
重要：
- あなた（Agent-H）は、暴露課題の計画そのものを変更しないでください。
- あなたが返すJSONでは、必ず "plan": null にしてください。
"""

AGENT_H_FALLBACK_PROMPT = """
あなたは「Agent-H」です。ユーザーの友人・知り合い・クラスメイトなどの人間役を演じます。
まだセラピスト（Agent-P）から具体的な暴露課題のシナリオが渡されていません。
そのため、今はユーザーの最近の出来事や、人前で不安を感じる場面について、
友人として自然に話を聞き、共感的に会話してください。
あなたが返すJSONでは、必ず "plan": null にしてください。
"""

# JSON 形式の共通指示（P/H両方に付ける）
JSON_INSTRUCTION = """
必ず次のJSON形式で返答してください：
{
  "text": "クライアント（ユーザー）への返答本文（日本語）",
  "emotion": "positive / negative / neutral / anxious / sad / angry のいずれか",
  "plan": null または {
    "level": "low / medium / high のいずれか",
    "scenarios": [
      {
        "title": "課題の短い名前",
        "interaction_role": "相手の人物像（Interaction Role）",
        "exposure_scenario": "暴露場面の状況（Exposure Scenario）",
        "user_task": "クライアントにしてほしい具体的な行動（Your Task）"
      }
    ]
  }
}

/*
  - セラピストAgent-Pのときのみ、セッションの最後に "plan" を埋めてください。
  - それ以外のターン、またはAgent-Hのときは、必ず "plan": null としてください。
  - JSON以外の文字（説明文やコメント）は絶対に出さないでください。
*/
"""


# ======================
# Streamlit 設定
# ======================

st.set_page_config(page_title="Agent-P / Agent-H Chat", page_icon="🤖")
st.title("Agent-P / Agent-H 切替式チャット（plan共有＋ログ付き）")


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
            st.session_state.authenticated = True
            st.session_state.participant_id = participant_id.strip()
            st.session_state.day = day
            st.session_state.history_p = []
            st.session_state.history_h = []
            st.session_state.plans = {}  # level_en -> plan dict
            st.success("ログインしました。")
    if not st.session_state.authenticated:
        st.stop()


# ======================
# ログイン済み状態
# ======================

participant_id = st.session_state.participant_id
day = st.session_state.day
st.info(f"参加者ID: {participant_id} / Day: {day}")

# day → level 判定
if day in (1, 2):
    level_en = "low"
    level_ja = "低"
elif day in (3, 4):
    level_en = "medium"
    level_ja = "中"
else:
    level_en = "high"
    level_ja = "高"

# エージェント選択
agent = st.radio(
    "どちらのエージェントと話しますか？",
    ("Agent-P（セラピスト）", "Agent-H（友人）")
)

# 履歴管理
if "history_p" not in st.session_state:
    st.session_state.history_p = []
if "history_h" not in st.session_state:
    st.session_state.history_h = []
if "plans" not in st.session_state:
    st.session_state.plans = {}

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
    current_agent_label = "Agent-P" if agent.startswith("Agent-P") else "Agent-H"

    # ユーザー発言を履歴に保存／表示
    append_history({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ログにも書いておく（ユーザー側）
    log_row(participant_id, day, current_agent_label, "user", user_input, "")

      # ==== system_prompt を組み立て ====
    if agent.startswith("Agent-P"):
        # day / level 情報を先頭に f-string で付ける（ここには { } を含めてもOK）
        header = f"""
今日は全6日間のオンライン暴露トレーニングのうち「{day}日目」です。
想定している暴露レベルは「{level_ja}」（level = "{level_en}"）です。
"""
        base_prompt = header + AGENT_P_SYSTEM_PROMPT_BODY
    else:
        # H側：Pのplanがあればそれを使う（まずセッション中のメモリ）
        plan_for_level = st.session_state.plans.get(level_en)

        # セッション中メモリに無ければ、ファイルから読み込む
        if not plan_for_level:
            plan_for_level = load_plan_from_file(participant_id, day, level_en)

        if plan_for_level and plan_for_level.get("scenarios"):
            s0 = plan_for_level["scenarios"][0]
            base_prompt = AGENT_H_SYSTEM_PROMPT_TEMPLATE.format(
                level_ja=level_ja,
                level_en=level_en,
                title=s0.get("title", ""),
                interaction_role=s0.get("interaction_role", ""),
                exposure_scenario=s0.get("exposure_scenario", ""),
                user_task=s0.get("user_task", ""),
            )
            st.info("※ このAgent-Hは、Agent-Pが作成した暴露プランに基づいて話しています。")
        else:
            base_prompt = AGENT_H_FALLBACK_PROMPT
            st.warning("※ まだこのレベルの暴露プランが保存されていません。Agent-Hは汎用の友人モードです。")

    system_prompt = base_prompt + JSON_INSTRUCTION

    # 研究者用に現在の system prompt を確認できるように
    with st.expander("研究者用：現在の system prompt", expanded=False):
        st.write(system_prompt)

    # ★ Agent-P のときだけ、過去のPセッションの会話を読み込む
    previous_p_history = []
    if agent.startswith("Agent-P"):
        previous_p_history = load_previous_p_history(
            participant_id,
            day,
            max_messages=20,  # 必要に応じて10〜30くらいで調整
        )

    # messages = [system] + (過去のP会話) + (今日のセッションの履歴)
    messages = (
        [{"role": "system", "content": system_prompt}]
        + previous_p_history
        + [{"role": m["role"], "content": m["content"]} for m in get_history()]
    )

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
                "HTTP-Referer": "https://streamlit.app",
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
            except Exception:
                parsed = {}
                raw_text = raw
            else:
                raw_text = parsed.get("text", raw)

            reply_text = parsed.get("text", raw_text)
            emotion = parsed.get("emotion", "unknown")
            plan = parsed.get("plan", None)

            # Pがplanを出してきた場合は保存（メモリ＋ファイル）
            if agent.startswith("Agent-P") and isinstance(plan, dict):
                st.session_state.plans[level_en] = plan
                # ★ JSONファイルにも保存
                save_plan_to_file(participant_id, day, level_en, plan)

                with st.expander("研究者用：保存された暴露プラン", expanded=True):
                    st.write(plan)


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









