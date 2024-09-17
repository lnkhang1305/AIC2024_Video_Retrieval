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
            index.add(embeddings[i].reshape(1, -1))
            client.upsert(
                collection_name="image_collection",
                points=[
                    {
                        "id": idx,
                        "vector": embeddings[i].reshape(1, -1).flatten(),
                        "payload": {"image_path": scene_frame+'/'+frame_path, "video": clip_file.split('/')[2], "frame_idx": int(index_frames.iloc[i])}
                    }
                ]
            )
            idx += 1
    faiss.write_index(index, "index.ivf")


dimension = 768
index = faiss.IndexFlatIP(dimension)
client = QdrantClient(url="http://localhost:6333")
client.recreate_collection(
    collection_name='image_collection',
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

clip_path = './clip_features'
clip_files = []
for video_path in sorted(os.listdir(clip_path)):
    for clip in sorted(os.listdir(clip_path+'/'+video_path)):
        clip_files.append(clip_path+'/'+video_path+'/'+clip)

scene_frames = []
frame_path = './keyframes'
for video_frame in sorted(os.listdir(frame_path)):
    for scene_frame in sorted(os.listdir(frame_path+'/'+video_frame)):
        scene_frames.append(frame_path+'/'+video_frame+'/'+scene_frame)

index_files = []
index_path = './map-keyframes'
for index_file in sorted(os.listdir(index_path)):
    index_files.append(index_path+'/'+index_file)

add_to_db(clip_files=clip_files, scene_frames=scene_frames,
          index_files=index_files)

