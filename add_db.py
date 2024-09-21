import os
import argparse
import numpy as np
import pandas as pd
import faiss
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def add_to_db(collection_name, clip_files, scene_frames, index_files, idx=0):
    for clip_file, scene_frame, index_file in zip(clip_files, scene_frames, index_files):
        # print(collection_name)
        embeddings = np.load(clip_file)
        index_frames = pd.read_csv(index_file, usecols=['frame_idx'])
        for i, frame_path in enumerate(sorted(os.listdir(scene_frame))):
            # print(embeddings[i].shape)
            index.add(embeddings[i].reshape(1, -1))
            # print(scene_frame+'/'+frame_path)
            client.upsert(
                collection_name=collection_name,
                points=[
                    {
                        "id": idx,
                        "vector": embeddings[i].reshape(1, -1).flatten(),
                        "payload": {"image_path": os.path.join(scene_frame,frame_path), 
                                    "video": scene_frame.split('\\')[-1], 
                                    "frame_idx": int(index_frames.iloc[i])}
                    }
                ]
            )
            idx += 1
    faiss.write_index(index, "index.ivf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=str, help='Name of collection')
    # Ex: ./data/batch_1
    parser.add_argument('-c', type=str, help='Choose clip model (b16/b32/l14)')

    args = parser.parse_args()

    collection_name, clip_feature_model = args.n, args.c
    if clip_feature_model == 'l14':
        dimension = 768
    else:
        dimension = 512
    index = faiss.IndexFlatIP(dimension)
    client = QdrantClient(url="http://localhost:6333")
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension, distance=Distance.COSINE)
        )
    else:
        client.get_collection(collection_name)

    clip_path = './clip_features_' + clip_feature_model
    if not os.path.exists(clip_path):
        clip_path = os.path.join(r"D:\AI_chalenge_2024\AI_Challenge\db\dot2", 'clip_features_' + clip_feature_model)
    clip_files = []
    for video_path in sorted(os.listdir(clip_path)):
        for clip in sorted(os.listdir(os.path.join(clip_path,video_path))):
            clip_files.append(os.path.join(clip_path,video_path,clip))

    scene_frames = []
    frame_path = './keyframes/Keyframes'
    if not os.path.exists(frame_path):
        frame_path = r"D:\AI_chalenge_2024\AI_Challenge\db\dot2\keyframes\Keyframes"
    for video_frame in sorted(os.listdir(frame_path)):
        for scene_frame in sorted(os.listdir(os.path.join(frame_path,video_frame))):
            scene_frames.append(os.path.join(frame_path,video_frame,scene_frame))

    index_files = []
    index_path = r"map-keyframes" if os.path.exists(
        r"map-keyframes") else r"D:\AI_chalenge_2024\AI_Challenge\db\map-keyframes-b1\map-keyframes"
    for index_file in sorted(os.listdir(index_path)):
        index_files.append(os.path.join(index_path, index_file))

    add_to_db(collection_name=collection_name,
              clip_files=clip_files,
              scene_frames=scene_frames,
              index_files=index_files)
