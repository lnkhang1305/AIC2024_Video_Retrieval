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
|-- clip_features_l14/
|-- |-- Lxx/
|-- |-- |-- Vxxx.npy
|-- clip_features_b16 (Tương tự)
|-- clip_features_b32 (Tương tự)
|
|-- keyframes/
|-- |-- Keyframes/
|-- |-- |-- Lxx/
|-- |-- |-- |-- xxx.jpg
|
|-- map-keyframes/
|-- |-- Lxx_Vxxx.csv
|
|-- media_info/
|-- |-- Lxx_Vxxx.json
```

# Usage
## Host qdrant
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
## Using add_db.py
```
python add_db.py -n name_qdrant_collection -c type_of_clip_model(l14/b32/b16)
```
1. Khi chạy, name_qdrant_collection đặt theo mẫu: {type_of_clip_model}_collection
2. Thư mục chứa cấu hình qdrant:
- l14_storage: model CLIP_L14
- b32_storage: model CLIP_B32
- b16_storage: model CLIP_B16
3. Khi có qdrant_strorage, cần tải toàn bộ keyframes để hiện thị ảnh cho web sau đó host qdrant với đường dẫn trỏ vào qdrant_storage và host web.

## Chạy web
1. Chạy command để cài đặt các thư viện cần thiết
```
pip install - r requirements.txt
```
2. Chạy command để chạy flask
```
python web.py
```
