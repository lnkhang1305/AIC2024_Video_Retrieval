# Download
1. BTC:
+ keyframes
+ media-info  
2. Mail group (aic2024batch01@gmail.com | AIC2024@gmail.com) 
+ clip-features
+ index.ivf
+ qdrant_storage

# Tổ chức thư mục
```
|-- static/ 
|
|-- templates/
|
|-- data/
|-- |-- batch_1/
|-- |-- |-- clip_features_l14
|-- |-- |-- clip_features_b16
|-- |-- |-- clip_features_b32
|-- |-- |-- keyframes
|-- |-- |-- map_frame
|-- |-- batch_2/
|-- |-- |-- clip_features_l14
|-- |-- |-- clip_features_b16
|-- |-- |-- clip_features_b32
|-- |-- |-- keyframes
|-- |-- |-- map_frame
```

# Usage
1. Cài đặt Docker
2. Cài đặt qdrant qua image của docker
```
docker pull qdrant/qdrant
```
3. Host server của docker  (nếu muốn add lại vào DB thì cứ xóa hết trong folder qdrant_storage, host lại như bên dưới sau đó chạy file add_db.py)
```
docker run -p 6333:6333 -v direct_path_to_qdrant_storage:/qdrant/storage qdrant/qdrant
```
4. Chạy file search.py 

## Chạy web
1. Chạy command để cài đặt các thư viện cần thiết
```
pip install - r requirements.txt
```
2. Chạy command để chạy flask
```
python web.py
```
