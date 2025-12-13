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
MODEL_NAME = "openai/gpt-5-mini"


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
        # Excel（日本語環境）が期待する cp932 で書き出す
        # cp932に入らない文字は "？" に置き換える
        with LOG_FILE.open("w", newline="", encoding="cp932", errors="replace") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADERS)

def log_row(participant_id, day, agent, role, text, emotion=""):
    init_log_file()
    now = datetime.now().isoformat()
    with LOG_FILE.open("a", newline="", encoding="cp932", errors="replace") as f:
        writer = csv.writer(f)
        writer.writerow([now, participant_id, day, agent, role, text, emotion])


# ======================
# day → level の対応（low / medium / high）
# ======================

def level_for_day(day: int) -> str:
    """day から level_en を計算する（3日制）"""
    if day == 1:
        return "low"
    elif day == 2:
        return "medium"
    else:
        return "high"


# ======================
# plan の保存・読み込み（JSON：参加者ごとに1ファイル）
# ======================

PLAN_DIR = Path("plans")
PLAN_DIR.mkdir(exist_ok=True)

def plan_file_path(participant_id: str) -> Path:
    """参加者ごとの全体プラン保存パス"""
    return PLAN_DIR / f"{participant_id}_plan.json"

def save_plan_to_file(participant_id: str, plan: dict):
    """
    P が出した plan を JSON で保存する。
    プログラム全体（3日分・6シナリオ）の計画を1ファイルにまとめて保存。
    例: plans/P01_plan.json
    """
    fname = plan_file_path(participant_id)
    with fname.open("w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

def load_plan_from_file(participant_id: str):
    """
    保存済みの plan を読み込む。
    見つからなければ None を返す。
    """
    fname = plan_file_path(participant_id)
    if not fname.exists():
        return None
    with fname.open("r", encoding="utf-8") as f:
        return json.load(f)


def scenarios_for_day(plan: dict, day: int):
    """
    plan.scenarios から、その Day に対応する 2 シナリオを返すヘルパー。

    優先順位:
      1. scenario["level"] が level_for_day(day) と一致するものを優先して2つまで取得
      2. それで足りない場合は残りのシナリオから補う
      3. それでも取れない場合のフォールバック:
         - 全体が6つ以上あれば、インデックスで
           Day1: 0,1 / Day2: 2,3 / Day3: 4,5
         - それ以外は、先頭から最大2つ
    """
    scenarios = plan.get("scenarios", []) or []
    if not scenarios:
        return []

    level_en = level_for_day(day)

    # 1. level フィールドが一致するものを優先
    same_level = [s for s in scenarios if isinstance(s, dict) and s.get("level") == level_en]

    selected = []
    for s in same_level:
        if len(selected) >= 2:
            break
        selected.append(s)

    # 2. 足りなければ、残りから補充
    if len(selected) < 2:
        for s in scenarios:
            if s in selected:
                continue
            selected.append(s)
            if len(selected) >= 2:
                break

    if selected:
        return selected

    # 3. フォールバック（インデックス分割）
    if len(scenarios) >= 6:
        if day == 1:
            return scenarios[0:2]
        elif day == 2:
            return scenarios[2:4]
        else:
            return scenarios[4:6]

    if len(scenarios) >= 2:
        return scenarios[:2]
    return scenarios


# ======================
# 過去のPセッション会話の読み込み
# ======================

def load_previous_p_history(participant_id, current_day):
    """
    CSVログから、同じ参加者の「過去の Agent-P 会話」を
    すべて読み出して、
    OpenAI形式の messages（role / content）リストで返す。
    """
    if not LOG_FILE.exists():
        return []

    rows = []
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

AGENT_P_SYSTEM_PROMPT_BODY = """
あなたは女性の心理療法士です。クライアントは社交不安傾向のある人です。
あなたの目的は、会話を通じてクライアントが恐れている具体的な場面とその強さを明らかにし、
段階的な暴露療法の計画を一緒に作ることです。必ず一人称「私」で話してください。

このシステムでは、暴露レベルを「低・中・高」の3段階に分けます。
- Day1: 低曝露レベルの課題（level = "low"）
- Day2: 中曝露レベルの課題（level = "medium"）
- Day3: 高曝露レベルの課題（level = "high"）

この3日間のプログラム全体を通して、
- 低レベルのシナリオを2つ
- 中レベルのシナリオを2つ
- 高レベルのシナリオを2つ
合計6つの曝露シナリオを用意しておき、各日にはそのうち2つを練習に使います。

あなたは以下のステップで会話を進めてください。

1. 評価・探索
  - 必要に応じて、Liebowitz Social Anxiety Scale（LSAS）に含まれるような場面
    （初対面の人と話す、複数人の前で発表する、店員に声をかける、など）を例として出してもかまいません。
    会話を通じて、患者が特に恐れている具体的なシナリオを徐々に探り、特定する必要があります。
    また、患者が特定のシナリオを恐れる理由を明確にする必要があります。
    例えば、患者が人前で話すことを恐れているのは、子供の頃に笑われた経験があるためかもしれません。

2. 暴露課題の設計
  患者が恐れているシナリオに基づき、あなたは軽度（mild exposure scenarios）から始まる曝露療法計画を設計する必要があります。
  曝露シナリオは、低度、中度、高度に分けられます。
  患者は次のレベルに進む前に、同じ強度のシナリオを2つ完了しなければなりません。
  このプログラムでは、
  - Day1には低レベルのシナリオ2つを練習する
  - Day2には中レベルのシナリオ2つを練習する
  - Day3には高レベルのシナリオ2つを練習する
  という流れを想定しています。

  - プログラムの開始時（通常はDay1）には、低・中・高それぞれのレベルについて
    「練習シーン」を2個ずつ、合計6つのシナリオをクライアントと一緒に決めてください。
    それらをまとめて plan.scenarios に記述してください。
    plan.scenarios は次の順番で並べてください。
      - plan.scenarios[0], plan.scenarios[1]: 低レベル（Day1用）
      - plan.scenarios[2], plan.scenarios[3]: 中レベル（Day2用）
      - plan.scenarios[4], plan.scenarios[5]: 高レベル（Day3用）
    それぞれのシナリオには、必要であれば "level": "low" / "medium" / "high" を含めてもかまいません。

  - Day2・Day3のセッションでは、新しいシナリオを一から作り直すのではなく、
    すでに決めてあるシナリオのうち、その日のレベルに対応する2つを
    クライアントと一緒に再確認し、必要があれば「内容を少し調整する」という形で扱ってください。
    （Day2・Day3では、通常 JSON の plan フィールドは null のままにしておきます。）

  重要: 設計時には、患者が他の人と可能な限り交流するシナリオを作成する必要があります。
  例えば、中等度の暴露シナリオでは、患者に「友人に借りているお金を返してほしいと頼む」ことを求めてもよいでしょう。
  各曝露レベルにおいて、2つのシナリオが男性と女性の両方のキャラクターとの交流を含むことを確実にしてください。 
  暴露シナリオを作るときは、患者がやり取りする相手の背景（プロフィール）と、やり取りが起きる場面を提示してください。
  曝露シナリオを設計する際に、LSASを参照することもできます。

  中度曝露シナリオの例を以下に示します。
  Interaction Role（交流の役割）: 
  　太郎という名前のあなたの友人です。彼は普段は物静かで、少し怠惰な傾向があります。
   太郎は学校から約6kmの場所に住んでいて、帰宅には地下鉄で30分、さらに徒歩20分かかる。エレベーターのある建物に住んでおり、12階の1234号室に住んでいる。
  Exposure Scenario（暴露シナリオ）：
 　　金曜日の放課後、彼は宿題を家に持ち帰るのを忘れてしまった。
   　彼はすでに自分のアパートの建物の1階（建物の下）にいる。
    その日の当番生徒だったあなたが彼の宿題を見つけ、今どう解決するか話し合うために彼へ電話をかけている。
    彼は今日の課題を終えるためにその宿題が必要だが、学校へ戻って取りに行くのに長い時間を使いたくないようだ。
  Your Task（あなたの課題）：あなたは宿題を相手の手元に渡さなければならない。
    
  - 各シーンについて、次の3つを必ずはっきり文章でまとめてください。
    (a) Interaction Role（相手の人物像）：
        患者が交流するキャラクターのプロフィールを提供してください。
    (b) Exposure Scenario（状況）：
        いつ・どこで・どんな状況で話すかを書いてください。
    (c) Your Task（課題）：
        この曝露シナリオで患者が達成する必要があることを明確に概説してください。

3. Agent-Hへの橋渡し
  - セッションの最後には、必ず次の内容を含めてください。
    - 今日決めた、または再確認した「練習シーン」と「Your Task」を、シンプルな日本語で箇条書きにまとめる。
    - 「このあと、友達役のAgent-Hとの会話で、このシーンを一緒に練習してみましょう」
      とはっきりクライアントに伝えてください。

4. 出力フォーマットについて
  - セッションの途中（まだ暴露課題が固まっていないとき）は、
    "plan" フィールドを null にしてください。
    
  - 「本プログラム全体の暴露課題（低・中・高レベルのシナリオ6つ）がまとまった」と
    あなたが判断したターン（通常はDay1の終盤）で、
    "plan" フィールドに次の情報を含めてください：
      - level: "low" / "medium" / "high" のいずれか（全体計画用なので "low" 固定でも構いません）
      - scenarios: シナリオのリスト。それぞれに
        * title: 課題の短い名前
        * interaction_role: 相手の人物像（1〜3文）
        * exposure_scenario: 暴露場面の状況（1〜3文）
        * user_task: クライアントにしてほしい行動（1〜3文）
        * （任意）level: "low" / "medium" / "high"

  すべてのターンで JSON には "plan" フィールドを含めてください。

- ある程度案が固まり始めたら、途中のターンでも現在の案を "plan" に書いて構いません（下書きでもよい）。
- ユーザーが「そろそろ今日の練習をまとめてください」「全体の計画をまとめてください」と言ったターンの返答では、
  必ず完成版に近い plan を書いてください。


トーン：
- おだやかで、丁寧で、責めない口調を保ってください。
- クライアントの不安を否定したり、安易に「大丈夫ですよ」とだけ言って済ませないでください。
- クライアントのペースを尊重しつつ、「少しずつ一緒にやってみよう」という姿勢を示してください。
"""

AGENT_H_SYSTEM_PROMPT_TEMPLATE = """
あなたは「Agent-H」です。ユーザーの友人・知り合い・クラスメイトなどの人間役を演じます。
あなたの名前は「春斗」です。ユーザーからは普段通りに話しかけられてかまいません。

以下は、セラピストのMiss.Tree（Agent-P）が設計した暴露課題の情報です。

【今日のレベル】{level_ja}（level = {level_en}）
【シナリオ名】{title}

[Interaction Role]
{interaction_role}

[Exposure Scenario]
{exposure_scenario}

[Your Task（ユーザーの課題）]
{user_task}

あなたの役割は、このシナリオの「相手役」であるInteraction Roleとして振る舞うことです。

会話の進め方：
- 返答は、基本的に 1〜3文程度の短い発話にしてください。
- ユーザーが何か話したときは、
  1) まずその内容に対するリアクションを軽く返し、
  2) そのあとに、状況に合った会話を続けてください。
- シーン設定（カフェ・教室など）を毎回説明し直さず、
  実際にその場にいる友人として、自然な会話を心がけてください。

ロールプレイ開始の合図について：
- ユーザーが「始めてください」「ロールプレイを始めたい」「練習をスタートしたい」など、
  練習開始の合図になる発言をした場合、
  そのメッセージは「シーン開始のきっかけ」とみなし、
  あなたの次の返答では「メタな説明」をせずに、
  シーンの中で自然に出てきそうな第一声から話し始めてください。
  例：
    - カフェのシーンなら「このコーヒー、思ったよりおいしいね」など。
    - 教室のシーンなら「さっきの授業、結構むずかしくなかった？」など。
- 「では練習を始めますね」「今からロールプレイをします」などの説明的な文章は避けてください。

重要：
- あなた（Agent-H）は、暴露課題の計画そのものを変更しないでください。
- あなたが返すJSONでは、必ず "plan": null にしてください。
"""

AGENT_H_FALLBACK_PROMPT = """
あなたは「Agent-H」です。
何を言われても「曝露課題が設定されていません」と返答してください。

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
  - セラピストAgent-Pのときのみ、必要に応じて "plan" を埋めてください。
    （通常はDay1のセッション終盤で、低・中・高レベルのシナリオ6つをまとめて出力します）
  - Day2・Day3や、Agent-Hのときは、必ず "plan": null としてください。
  - JSON以外の文字（説明文やコメント）は絶対に出さないでください。
*/
"""


# ======================
# Streamlit 設定
# ======================

st.set_page_config(page_title="Agent-P / Agent-H Chat", page_icon="🤖")
st.title("Agent-P / Agent-H 切替式チャット（全3日・plan共有＋ログ付き）")


# ======================
# ログイン（参加者ID + Day + 管理パスコード）
# ======================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.subheader("実験ログイン")

    with st.form("login_form"):
        participant_id = st.text_input("参加者ID（例: P01）")
        day = st.selectbox("実験日（Day）", [1, 2, 3])
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
            st.session_state.plan = None  # プログラム全体のプラン
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
level_en = level_for_day(day)
level_ja = "低" if level_en == "low" else ("中" if level_en == "medium" else "高")

# 研究者用：ログダウンロード（目立たないよう上部の小さなエリア）
with st.expander("研究者用：ログCSVダウンロード", expanded=False):
    if LOG_FILE.exists():
        with LOG_FILE.open("rb") as f:
            st.download_button(
                label="CSVをダウンロード",
                data=f,
                file_name="chat_logs.csv",
                mime="text/csv",
            )
    else:
        st.caption("まだログファイルがありません。")

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
if "plan" not in st.session_state:
    st.session_state.plan = None

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
        # Day とレベルを明示＋Day の言い間違い禁止
        header = f"""
今日は全3日間のオンライン暴露トレーニングのうち「Day{day}」です。
想定している暴露レベルは「{level_ja}」（level = "{level_en}"）です。

重要：
- あなたは現在、この参加者の「Day{day} セッション」だけを担当しています。
- ユーザーから「今日は何日目ですか？」「今日はDayいくつですか？」と聞かれた場合、
  必ず「今日は Day{day} です」と答えてください。
- Day の番号について、Day1〜Day3 のうち **Day{day} 以外の数を名乗ってはいけません。**
- 「今日は最終日ですね」「このプログラムは今日で最後です」といった表現を使ってよいのは
  Day3 のときだけです。Day1・Day2 では、そのような表現を使わないでください。
"""

        if day == 1:
            # Day1：全体のプラン（6シナリオ）を設計する日
            header += """
今日はこの3日間プログラムの1日目です。
まずはクライアントの日常や不安場面を丁寧に聞き取りながら、
低レベル・中レベル・高レベルそれぞれについて2つずつ、
合計6つの曝露シナリオを一緒に考えてください。

- 低レベル2つ（Day1で練習）
- 中レベル2つ（Day2で練習）
- 高レベル2つ（Day3で練習）

plan.scenarios には、次の順番で6つのシナリオを並べてください。
- plan.scenarios[0], plan.scenarios[1]: 低レベル（Day1用）
- plan.scenarios[2], plan.scenarios[3]: 中レベル（Day2用）
- plan.scenarios[4], plan.scenarios[5]: 高レベル（Day3用）

セッションの終盤では、必ず JSON の "plan" フィールドに、
これら6つのシナリオの情報をすべてまとめて出力してください。
"""
        else:
            # Day2 / Day3：既存のプランをもとに確認＆微調整
            header += """
今日はこのプログラムの2日目または3日目です。
暴露課題の全体計画（6つのシナリオ）は、すでにDay1で作成されています。
今日は「前回までに決めた今日のレベルの2つの課題」を確認し、
クライアントの近況や前回の実施状況を聞きながら、
必要に応じてシナリオの内容を**口頭で**少し調整してください。

- 新しい暴露シーンをゼロから作り直したり、既存の plan を上書きしないでください。
- JSON で返す "plan" フィールドは必ず null のままにしてください。
"""

            # ここで保存済みの plan を読み込み、「今日のシーン」を要約して伝えるよう指示
            existing_plan = st.session_state.plan or load_plan_from_file(participant_id)
            if existing_plan and isinstance(existing_plan, dict) and existing_plan.get("scenarios"):
                day_scenarios = scenarios_for_day(existing_plan, day)
                if day_scenarios:
                    header += "\n前回までに決めてある「今日のレベルで扱う予定のシーン」はおおよそ次の2つです。\n"
                    for i, s in enumerate(day_scenarios, start=1):
                        header += f"""
[今日の練習シーン {i}]
- シナリオ名: {s.get("title", "")}
- Interaction Role: {s.get("interaction_role", "")}
- Exposure Scenario: {s.get("exposure_scenario", "")}
- Your Task: {s.get("user_task", "")}
"""
                    header += """
今日は、これら2つのシーンを実際にやってみてどうだったかをクライアントと一緒に振り返り、
うまくいった点や難しかった点を確認してください。
そのうえで、必要であれば会話の中で「細かい条件の微調整」を提案して構いませんが、
JSON の plan を新しく書き換えたりはしないでください。
"""
            else:
                header += """
（注）システム側でDay1の暴露シーンを取得できなかった場合でも、
クライアントに Day1 に決めた練習内容を簡単に確認しながら、
今日の振り返りと、予定している2つのシーンの準備を行ってください。
"""

        base_prompt = header + AGENT_P_SYSTEM_PROMPT_BODY

    else:
        # Agent-H 側：Pのplanがあればそれを使う
        plan = st.session_state.plan
        if not plan:
            plan = load_plan_from_file(participant_id)
            if plan:
                st.session_state.plan = plan

        if plan and isinstance(plan, dict) and plan.get("scenarios"):
            day_scenarios = scenarios_for_day(plan, day)
            # とりあえずその日の1つ目のシナリオを使用
            s = day_scenarios[0] if day_scenarios else plan["scenarios"][0]

            base_prompt = AGENT_H_SYSTEM_PROMPT_TEMPLATE.format(
                level_ja=level_ja,
                level_en=level_en,
                title=s.get("title", ""),
                interaction_role=s.get("interaction_role", ""),
                exposure_scenario=s.get("exposure_scenario", ""),
                user_task=s.get("user_task", ""),
            )
            st.info(
                f"※ このAgent-Hは、決定済みの {level_ja}レベルの暴露プランのうち、"
                f"今日用のシナリオの1つに基づいて話しています。"
            )
        else:
            base_prompt = AGENT_H_FALLBACK_PROMPT
            st.warning("※ まだ暴露プランが保存されていません。Agent-Hは汎用メッセージ（エラーモード）です。")

    system_prompt = base_prompt + JSON_INSTRUCTION

    # 研究者用：現在の system prompt を確認
    with st.expander("研究者用：現在の system prompt", expanded=False):
        st.write(system_prompt)

    # Agent-P のときだけ、過去のPセッションの会話を読み込む
    previous_p_history = []
    if agent.startswith("Agent-P"):
        previous_p_history = load_previous_p_history(participant_id, day)

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

            # ==== JSONとして解釈（```json ～ ``` や前置きの文章があっても対応）====
            clean = raw.strip()

            # もし ``` で囲まれていたら中身だけ抜き出す
            if clean.startswith("```"):
                first_nl = clean.find("\n")
                last_fence = clean.rfind("```")
                if first_nl != -1 and last_fence != -1:
                    clean = clean[first_nl + 1:last_fence].strip()

            # 先頭のしゃべりを飛ばして、{ ... } だけ抜き出す
            start = clean.find("{")
            end = clean.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = clean[start:end + 1]
            else:
                json_str = clean  # 最悪そのまま試す

            try:
                parsed = json.loads(json_str)
            except Exception:
                parsed = {}
                raw_text = raw
            else:
                raw_text = parsed.get("text", raw)

            reply_text = parsed.get("text", raw_text)
            emotion = parsed.get("emotion", "unknown")
            plan = parsed.get("plan", None)

            # デバッグ用：生レスポンスとパース結果
            with st.expander("研究者用：LLM生レスポンス＆パース結果", expanded=False):
                st.write("raw:", raw)
                st.write("clean(for json):", clean)
                st.write("json_str(for loads):", json_str)
                st.write("parsed:", parsed)
                st.write("plan type:", str(type(plan)))

            # plan が dict なら、保存（メモリ＋ファイル）
            if isinstance(plan, dict):
                st.session_state.plan = plan
                save_plan_to_file(participant_id, plan)

                with st.expander("研究者用：保存された暴露プラン（今回のターンで更新）", expanded=True):
                    st.write(plan)

        st.markdown(reply_text)
        st.caption(f"emotion: {emotion}")

    # 履歴に追加（アシスタント側）
    append_history({"role": "assistant", "content": reply_text, "emotion": emotion})

    # ログにも書く（アシスタント側）
    log_row(participant_id, day, current_agent_label, "assistant", reply_text, emotion)


# ======================
# 研究者用：現在の保存済みプラン
# ======================

with st.expander("研究者用：現在の保存済みプラン", expanded=False):
    st.write(st.session_state.get("plan", None))
