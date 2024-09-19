import os
import numpy as np
import pandas as pd
import faiss
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def add_to_db(clip_files, scene_frames, index_files):
    idx = 0
    for clip_file, scene_frame, index_file in zip(clip_files, scene_frames, index_files):
        embeddings = np.load(clip_file)
        index_frames = pd.read_csv(index_file, usecols=['frame_idx'])
        for i, frame_path in enumerate(sorted(os.listdir(scene_frame))):
            # print(embeddings[i].shape)
            index.add(embeddings[i].reshape(1, -1))
            # print(scene_frame+'/'+frame_path)
            client.upsert(
                collection_name="image_collection",
                points=[
                    {
                        "id": idx,
                        "vector": embeddings[i].reshape(1, -1).flatten(),
                        "payload": {
                            "image_path": os.path.join(scene_frame, frame_path), 
                            "video": clip_file.split('\\')[2], 
                            "frame_idx": int(index_frames.iloc[i])}
                    }
                ]
            )
            idx += 1
    faiss.write_index(index, "index.ivf")


dimension = 512
index = faiss.IndexFlatIP(dimension)
client = QdrantClient(url="http://localhost:6333")
client.recreate_collection(
    collection_name='image_collection',
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
name_of_model = "CLIP_L14@336PX" #name of model folder
clip_path = name_of_model if os.path.exists(name_of_model) else os.path.join(r"D:\AI_chalenge_2024\AI_Challenge\model",name_of_model)
clip_files = []
for video_path in sorted(os.listdir(clip_path)):
    for clip in sorted(os.listdir(os.path.join(clip_path,video_path))):
        clip_files.append(os.path.join(clip_path, video_path, clip))

scene_frames = []
frame_path = []
keyframes_path = r"keyframes" if os.path.exists(r"keyframes") else r"D:\AI_chalenge_2024\AI_Challenge\db\videos"
for i in range(1,13):
    frame_path.append(os.path.join(keyframes_path, "Keyframes_L"+ str(i).rjust(2,'0')))
    # print(frame_path)

for kf in frame_path:
    for video_frame in sorted(os.listdir(kf)):
        for scene_frame in sorted(os.listdir(os.path.join(kf,video_frame))):
            scene_frames.append(os.path.join(kf, video_frame, scene_frame))

index_files = []
index_path = r"map-keyframe" if os.path.exists(r"map-keyframe") else r"D:\AI_chalenge_2024\AI_Challenge\db\map-keyframes-b1\map-keyframes"
for index_file in sorted(os.listdir(index_path)):
    index_files.append(os.path.join(index_path, index_file))

add_to_db(clip_files=clip_files, scene_frames=scene_frames,
          index_files=index_files)

