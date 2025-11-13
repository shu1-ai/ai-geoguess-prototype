# プロジェクト名
景色から国を予想する画像分類アプリ（仮）

## 概要
このアプリは、**AIが景色から撮影国を推定する画像分類アプリ**です。  
ユーザーがアップロードした画像をAIが解析し、最も可能性の高い国を予測します。 
予測の際には、画像のどの箇所に重点を置いたか、アテンションマップで確認できます。
また、AIとユーザーが同じ画像で国を当て合う「対戦モード」も搭載しています。


## アーキテクチャ構成
- **フロントエンド:** Streamlit  
- **バックエンド:** FastAPI  
- **データベース:** SQLite（PostgreSQLへ移行予定）  
- **ストレージ:** Google Cloud Storage (学習済みモデル・画像データ)

## Credit

本アプリの実装にあたり、以下のデータを使用しました：

**OpenStreetView-5M**  
The Many Roads to Global Visual Geolocation, CVPR 2024 (Poster)  
Guillaume Astruc*1,2, Nicolas Dufour*1,3, Ioannis Siglidis*1,  
Constantin Aronssohn1, Nacim Bouia, Stephanie Fu1,4, Romain Loiseau1,2,  
Van Nguyen Nguyen1, Charles Raude1, Elliot Vincent1,3, Lintao XU1, Hongyu Zhou1, Loic Landrieu1  

彼らがオープンソースデータとして作成したものを使用しています：  
[https://huggingface.co/datasets/osv5m/osv5m]