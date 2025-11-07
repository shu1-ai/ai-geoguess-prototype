# プロジェクト名
AI画像分類バトルアプリ（仮）

## 📖 概要
このアプリは、AIモデルを用いた画像分類およびAIとの画像分類対戦のアプリです。
ユーザーがアップロードした画像を分類し、国・地域を推定します。
Streamlit（フロント）＋ FastAPI（バックエンド）＋ SQLite（DB）の三層構造で構築しています。

## 🧠 主な技術スタック
- Python 3.10+
- FastAPI
- Streamlit
- PyTorch (CLIP / ViTモデル)
- SQLite（開発環境）
- PostgreSQL（デプロイ予定）

## Credit

本アプリの実装にあたり、以下のデータを使用しました：

**OpenStreetView-5M**  
The Many Roads to Global Visual Geolocation, CVPR 2024 (Poster)  
Guillaume Astruc*1,2, Nicolas Dufour*1,3, Ioannis Siglidis*1,  
Constantin Aronssohn1, Nacim Bouia, Stephanie Fu1,4, Romain Loiseau1,2,  
Van Nguyen Nguyen1, Charles Raude1, Elliot Vincent1,3, Lintao XU1, Hongyu Zhou1, Loic Landrieu1  

彼らがオープンソースデータとして作成したものを使用しています：  
[https://huggingface.co/datasets/osv5m/osv5m]