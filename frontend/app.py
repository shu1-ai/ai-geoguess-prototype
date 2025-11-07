import json
import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

# ==============================
# クラス一覧の読み込み
# ==============================
import os

base_dir = os.path.dirname(__file__)
json_path = os.path.join(base_dir, "country_map.json")

with open(json_path, "r", encoding="utf-8") as f:
    COUNTRY_MAP = json.load(f)

classes = list(COUNTRY_MAP.values())
classes_sorted = sorted(classes)
reverse_map = {v: k for k, v in COUNTRY_MAP.items()}

st.title("🌍 GeoGuessアプリ 🌍")

# ==============================
# セッション初期化
# ==============================
for key in ["image_b64", "answer_code", "answer_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ==============================
# モード選択
# ==============================
mode = st.sidebar.radio(
    "モードを選んでください",
    ("画像アップロードで推論", "AIと予測対戦", "戦績表示")
)

# --------------------------------------------------
# 画像アップロードで推論モード
# --------------------------------------------------
if mode == "画像アップロードで推論":
    st.header("📷 ViTによる推論")

    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="アップロードした画像のプレビュー", width=800)

        if st.button("予測開始"):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}

            try:
                res = requests.post("http://127.0.0.1:8000/predict_rollout_topk?topk=3", files=files)
                res.raise_for_status()
                data = res.json()

                st.success("Top候補:")
                for i, item in enumerate(data["top_countries"]):
                    st.write(f"{i+1}. {item['name']} ({item['code']}) - 確率: {item['score']*100:.2f}%")

                st.markdown("### 各ブロック平均アテンション")
                for i, heatmap_b64 in enumerate(data["block_heatmaps"]):
                    st.markdown(f"**Block {i}**")
                    heatmap_bytes = base64.b64decode(heatmap_b64)
                    st.image(Image.open(BytesIO(heatmap_bytes)), width=400)

                st.markdown("### Rollout（累積アテンション）")
                rollout_bytes = base64.b64decode(data["rollout_heatmap"])
                st.image(Image.open(BytesIO(rollout_bytes)), width=400)

            except Exception as e:
                st.error(f"予測に失敗しました: {e}")

# --------------------------------------------------
# 対戦モード
# --------------------------------------------------
elif mode == "AIと予測対戦":
    st.header("⚔️ AIとの予測対戦")

    if st.button("景色を探す"):
        try:
            res = requests.get("http://127.0.0.1:8000/get_random_image")
            res.raise_for_status()
            data = res.json()
            st.session_state["image_b64"] = data["image"]
            st.session_state["answer_code"] = data["country_code"]
            st.session_state["answer_name"] = data["country_name"]
        except Exception as e:
            st.error(f"画像の取得に失敗しました: {e}")
            st.stop()

    # 画像があれば表示
    if st.session_state["image_b64"]:
        img_bytes = base64.b64decode(st.session_state["image_b64"])
        img = Image.open(BytesIO(img_bytes))
        st.image(img, caption="この景色はどこの国？", width=800)

        user_choice = st.selectbox("あなたの予想する国を選んでください", options=classes_sorted)

        if st.button("対戦開始！"):
            payload = {
                "image_b64": st.session_state["image_b64"],
                "user_choice": user_choice,
                "answer_code": st.session_state["answer_code"]
            }

            # --- try ブロックをボタン内で整理 ---
            try:
                res = requests.post("http://127.0.0.1:8000/battle", json=payload)
                res.raise_for_status()
                data = res.json()

                st.markdown("### 🧠 AIのTop3予測")
                for i, t in enumerate(data["ai_top3"]):
                    st.write(f"{i+1}. {t['name']} ({t['code']}) - {t['score']*100:.2f}%")

                st.markdown(f"**正解の国名:** {st.session_state['answer_name']} ({st.session_state['answer_code']})")
                st.markdown(f"**あなたの回答:** {user_choice}")
                st.markdown(f"### 🏁 結果: {data['result']} 🎉")

                # アテンションマップ表示
                st.markdown("### 各ブロックのアテンションマップ")
                for blk in data.get("block_heatmaps", []):
                    st.markdown(f"**Block {blk['block']}**")
                    st.image(Image.open(BytesIO(base64.b64decode(blk["heatmap"]))), width=400)

                st.markdown("### Rollout（累積アテンション）")
                if "rollout_heatmap" in data:
                    st.image(Image.open(BytesIO(base64.b64decode(data["rollout_heatmap"]))), width=400)

            except Exception as e:
                st.error(f"対戦処理に失敗しました: {e}")

    else:
        st.info("まず『景色を探す』ボタンを押してください。")

# -----------------------------
# 戦績表示モード
# -----------------------------
elif mode == "戦績表示":
    st.header("📊 戦績表示")

    try:
        res = requests.get("http://127.0.0.1:8000/get_battle_records")
        res.raise_for_status()
        records = res.json()  # 例: [{"timestamp":..., "user_choice":..., "answer_code":..., "result":...}, ...]

        if records:
            import pandas as pd
            df = pd.DataFrame(records)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            st.dataframe(df.sort_values("timestamp", ascending=False))  # 最新順に表示

            # 統計計算
            total = len(df)
            wins = len(df[df["result"]=="あなたの勝ち！🎉"])
            ai_wins = len(df[df["result"]=="AIの勝ち！🤖"])
            draws = len(df[df["result"]=="引き分け！🤝"])

            user_acc = wins / total * 100
            ai_acc = ai_wins / total * 100
            draw_rate = draws / total * 100

            st.markdown(f"**総対戦数:** {total}")
            st.markdown(f"**ユーザー正解率:** {user_acc:.2f}%")
            st.markdown(f"**AI正解率:** {ai_acc:.2f}%")
            st.markdown(f"**引き分け率:** {draw_rate:.2f}%")

        else:
            st.info("まだ戦績がありません。対戦モードでプレイしてください。")

    except Exception as e:
        st.error(f"戦績取得に失敗しました: {e}")