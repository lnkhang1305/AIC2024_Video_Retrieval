# Download
BTC:
+ keyframes
+ media-info
Mail group (aic2024batch01@gmail.com | AIC2024@gmail.com)
+ clip-features
+ index.ivf

# Usage
1. Cài đặt Docker
2. Cài đặt qdrant qua image của docker
```
docker pull qdrant/qdrant
```
3. Host server của docker 
```
docker run -p 6333:6333 -v direct_path_to_qdrant_storage:/qdrant/storage qdrant/qdrant
```
4. Chạy file search.py 
