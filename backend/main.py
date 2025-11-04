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

#===============================
# データベース
#===============================
import sqlite3
from datetime import datetime
DB_PATH = "battle_results.db"

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

# 戦績登録用関数
def save_battle_record(user_choice, answer_code, result):
    import datetime
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO battles (timestamp, user_choice, answer_code, result) VALUES (?, ?, ?, ?)",
        (datetime.datetime.now().isoformat(), user_choice, answer_code, result)
    )
    conn.commit()
    conn.close()

# ==============================
# クラス定義
# ==============================
csv_path = r"C:\Users\nshui\osv5m\labels\train_subset.csv"
df = pd.read_csv(csv_path)
classes = sorted(df["country"].dropna().unique().tolist())
num_classes = len(classes)

country_map_path = r"C:\Users\nshui\Documents\geoguess_proto\data\country_map.json"
with open(country_map_path, "r", encoding="utf-8") as f:
    COUNTRY_MAP = json.load(f)

# ==============================
# モデルロード
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "vit_base_patch16_clip_224"
model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(
    r"C:\Users\nshui\Documents\geoguess_proto\models\country_clipvit_finetune.pth",
    map_location=device
))
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
# ランダム画像取得
# ==============================
# --- グローバルに定義 ---
TEST_IMG_DIR = Path(r"C:\Users\nshui\osv5m\test\01")
df_test = pd.read_csv(r"C:\Users\nshui\osv5m\test\test.csv")

@app.get("/get_random_image")
async def get_random_image():
    global df_test  # グローバル変数を使う宣言
    # 存在する画像だけを抽出
    df_test_copy = df_test.copy()
    df_test_copy["img_path"] = df_test_copy["id"].apply(lambda x: TEST_IMG_DIR / f"{x}.jpg")
    df_available = df_test_copy[df_test_copy["img_path"].apply(lambda p: p.exists())]

    if df_available.empty:
        return {"error": "画像が存在するサンプルがありません"}

    sample = df_available.sample(1).iloc[0]
    img_path = sample["img_path"]

    # 画像をBase64に変換
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    country_code = sample["country"]
    country_name = COUNTRY_MAP.get(country_code, country_code)

    return {
        "image": img_b64,
        "country_code": country_code,
        "country_name": country_name
    }
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