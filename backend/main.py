from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, json, base64
import torch
import timm
import matplotlib.pyplot as plt
from io import BytesIO
from torchvision import transforms
import pandas as pd
from pathlib import Path
import random
import base64
import os, json, tempfile
from google.cloud import storage

#===============================
# クラウド設定
#===============================

# 環境変数に埋め込まれたJSONキーを一時ファイルに出力して認証
cred_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if cred_json:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(cred_json.encode("utf-8"))
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """GCSの指定ファイルをローカルにダウンロード"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"✅ {source_blob_name} downloaded to {destination_file_name}")

def load_json_from_gcs(bucket_name, source_blob_name):
    """GCS上のJSONを直接ロード（ファイル保存せずに）"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_text()
    print(f"✅ {source_blob_name} loaded as JSON")
    return json.loads(data)

# ===== 実行部分 =====
BUCKET_NAME = "geogu_data"

# モデル・CSVをローカルへ
download_from_gcs(BUCKET_NAME, "country_clipvit_finetune.pth", "country_clipvit_finetune.pth")
download_from_gcs(BUCKET_NAME, "train_subset.csv", "train_subset.csv")
download_from_gcs(BUCKET_NAME, "test_subset.csv", "test_subset.csv")

# country_map.jsonをロード（保存しない）
country_map = load_json_from_gcs(BUCKET_NAME, "country_map.json")

print("🌍 country_mapの一部:", list(country_map.items())[:5])  # デバッグ出力

#===============================
# データベース
#===============================
import sqlite3
import boto3
from datetime import datetime
DB_PATH = "battle_results.db"
S3_BUCKET = "my-battle-app-data"

# S3クライアント（環境変数から取得）
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS battles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_choice TEXT,
            answer_code TEXT,
            result TEXT
        )
    """)
    conn.commit()
    conn.close()

# 起動時にDB初期化
init_db()

#戦績S3送付の定義
def upload_single_record_to_s3(record_dict):
    df = pd.DataFrame([record_dict])
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    filename = f"battle_results/battle_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=filename,
        Body=csv_buffer.getvalue()
    )


# 戦績登録用関数
from datetime import datetime
def save_battle_record(user_choice, answer_code, result):
    
    timestamp = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO battles (timestamp, user_choice, answer_code, result) VALUES (?, ?, ?, ?)",
        (datetime.datetime.now().isoformat(), user_choice, answer_code, result)
    )
    conn.commit()
    conn.close()

    # S3に1レコードだけ追加保存
    upload_single_record_to_s3({
        "timestamp": timestamp,
        "user_choice": user_choice,
        "answer_code": answer_code,
        "result": result
    })

# ==============================
# クラス定義
# ==============================
df = pd.read_csv("train_subset.csv")
classes = sorted(df["country"].dropna().unique().tolist())
num_classes = len(classes)

COUNTRY_MAP = country_map  # GCSからロード済みの辞書を直接使う

# ==============================
# モデルロード
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "vit_base_patch16_clip_224"
model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load("country_clipvit_finetune.pth", map_location=device))
model.to(device)
model.eval()

# ==============================
# Attention抽出
# ==============================
def get_all_attention_maps(model, input_tensor):
    """ 各ブロック・全ヘッド平均のアテンションマップ """
    all_attn = []
    with torch.no_grad():
        x = model.patch_embed(input_tensor)
        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + model.pos_embed
        x = model.pos_drop(x)

        for blk in model.blocks:
            B, N, C = x.shape
            qkv = blk.attn.qkv(x)
            qkv = qkv.reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * (1.0 / (k.shape[-1] ** 0.5))
            attn = attn.softmax(dim=-1)
            # ヘッド平均
            attn_mean = attn.mean(1)
            all_attn.append(attn_mean)
            x = blk(x)
    return all_attn

# ==============================
# Rollout計算
# ==============================
def attention_rollout(attn_maps):
    result = torch.eye(attn_maps[0].size(-1)).to(device)
    for attn in attn_maps:
        attn = attn / attn.sum(dim=-1, keepdim=True)
        result = attn @ result
    return result

# ==============================
# ヒートマップ生成
# ==============================
def generate_heatmap(image, attn_map):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(attn_map, cmap="hot", alpha=0.5, extent=(0, image.width, image.height, 0))
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64

# ==============================
# 前処理
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

# ==============================
# FastAPI設定
# ==============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==============================
# エンドポイント
# ==============================
@app.post("/predict_rollout_topk")
async def predict_rollout_topk(file: UploadFile = File(...), topk: int = 3):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 推論
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=-1)
        top_probs, top_idxs = probs.topk(topk, dim=-1)

        top_countries = []
        for idx, prob in zip(top_idxs[0], top_probs[0]):
            code = classes[idx.item()]
            name = COUNTRY_MAP.get(code, code)
            top_countries.append({"code": code, "name": name, "score": float(prob)})

    # 各ブロック平均アテンション
    attn_maps = get_all_attention_maps(model, input_tensor)

    block_heatmaps = []
    for blk_attn in attn_maps:
        blk_map = blk_attn[0, 0, 1:].reshape(14, 14).cpu().numpy()
        block_heatmaps.append(generate_heatmap(image, blk_map))

    # Rollout
    rollout_map = attention_rollout(attn_maps)
    rollout_mask = rollout_map[0, 0, 1:].reshape(14, 14).cpu().numpy()
    rollout_heatmap = generate_heatmap(image, rollout_mask)

    return {
        "top_countries": top_countries,
        "block_heatmaps": block_heatmaps,
        "rollout_heatmap": rollout_heatmap
    }

# ==============================
# GCS対応：ランダム画像取得（高速版）
# ==============================

df_test = pd.read_csv("test_subset.csv")

@app.get("/get_random_image")
async def get_random_image():
    """test_subset.csv と GCS上の01フォルダを1対1対応としてランダム取得"""
    try:
        # ランダムサンプルを選択
        sample = df_test.sample(1).iloc[0]
        img_id = sample["id"]
        country_code = sample["country"]
        country_name = COUNTRY_MAP.get(country_code, country_code)
        gcs_path = f"01/{img_id}.jpg"

        print(f"🎯 Fetching from GCS: {gcs_path}")

        # GCSから画像を取得
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)

        # ダウンロードとエンコード
        img_bytes = blob.download_as_bytes()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        print(f"✅ Success: {gcs_path}")

        return {
            "image": img_b64,
            "country_code": country_code,
            "country_name": country_name
        }

    except Exception as e:
        print(f"💥 Exception while fetching image: {e}")
        return {"error": str(e)}
# ==============================
# 対戦モード：ユーザー vs AI
# ==============================
reverse_map = {v: k for k, v in COUNTRY_MAP.items()}

@app.post("/battle")
async def battle(payload: dict):
    """ユーザーの選択とAIのTop3を比較して勝敗判定"""
    image_b64 = payload["image_b64"]
    user_choice_name = payload["user_choice"]
    answer_code = payload["answer_code"]

    # --- ユーザー選択をコードに変換 ---
    user_code = reverse_map.get(user_choice_name, user_choice_name)

    # --- 画像デコード → Tensor ---
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # --- AI推論 ---
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=-1)
        top_probs, top_idxs = probs.topk(3, dim=-1)

        ai_top3 = []
        ai_codes = []
        for idx, prob in zip(top_idxs[0], top_probs[0]):
            code = classes[idx.item()]
            ai_codes.append(code)
            name = COUNTRY_MAP.get(code, code)
            ai_top3.append({"code": code, "name": name, "score": float(prob)})

        # --- 勝敗判定（Top1のみ） ---
        ai_top1_code = ai_codes[0]  # AIの1位予測コード

        if user_code == answer_code and user_code != ai_top1_code:
            result = "あなたの勝ち！🎉"
        elif user_code != answer_code and answer_code == ai_top1_code:
            result = "AIの勝ち！🤖"
        elif user_code == answer_code and answer_code == ai_top1_code:
            result = "引き分け！🤝"
        else:
            result = "どちらも不正解😅"
        
        # 勝敗判定後
        save_battle_record(user_choice_name, answer_code, result)

    # --- アテンションマップ取得 ---
    attn_maps = get_all_attention_maps(model, input_tensor)
    block_heatmaps = []
    for i, blk_attn in enumerate(attn_maps):
        blk_map = blk_attn[0, 0, 1:].reshape(14, 14).cpu().numpy()
        heatmap_b64 = generate_heatmap(image, blk_map)
        block_heatmaps.append({"block": i, "heatmap": heatmap_b64})

    rollout_map = attention_rollout(attn_maps)
    rollout_mask = rollout_map[0, 0, 1:].reshape(14, 14).cpu().numpy()
    rollout_heatmap = generate_heatmap(image, rollout_mask)

    return {
        "ai_top3": ai_top3,
        "result": result,
        "block_heatmaps": block_heatmaps,
        "rollout_heatmap": rollout_heatmap
    }

# 戦績取得用エンドポイント
@app.get("/get_battle_records")
def get_battle_records():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT timestamp, user_choice, answer_code, result FROM battles ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return [
        {"timestamp": row[0], "user_choice": row[1], "answer_code": row[2], "result": row[3]}
        for row in rows
    ]

# ==============================
# 全クラス一覧取得（任意）
# ==============================
@app.get("/classes")
async def get_classes():
    """全ての国コード・国名を返す（フロント選択肢用）"""
    items = [{"code": c, "name": COUNTRY_MAP.get(c, c)} for c in classes]
    return {"classes": items}

# ==============================
# 動作確認用ルート
# ==============================
@app.get("/")
def root():
    return {"message": "FastAPI backend is running on Render 🚀"}

# ==============================
# Render上での起動
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))