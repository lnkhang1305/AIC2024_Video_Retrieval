import os
import argparse
import numpy as np
import pandas as pd
import faiss
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def add_to_db(collection_name, clip_files, scene_frames, index_files, idx=0):
    for clip_file, scene_frame, index_file in zip(clip_files, scene_frames, index_files):
        embeddings = np.load(clip_file)
        index_frames = pd.read_csv(index_file, usecols=['frame_idx'])
        for i, frame_path in enumerate(sorted(os.listdir(scene_frame))):
            index.add(embeddings[i].reshape(1, -1))
            client.upsert(
                collection_name=collection_name,
                points=[
                    {
                        "id": idx,
                        "vector": embeddings[i].reshape(1, -1).flatten(),
                        "payload": {"image_path": scene_frame+'/'+frame_path, "video": clip_file.split('/')[3], "frame_idx": int(index_frames.iloc[i])}
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
    # w-write; a-append
    mode = input('Mode (w/a):')
    if mode == 'w':
        index = faiss.IndexFlatIP(dimension)
        add_to_db(collection_name=collection_name, clip_files=clip_files, scene_frames=scene_frames,
                  index_files=index_files)
    else:
        index = faiss.read_index('index.ivf')
        collection_info = client.get_collection(collection_name)
        num_vectors = collection_info.points_count
        print(num_vectors)
        add_to_db(collection_name=collection_name, clip_files=clip_files, scene_frames=scene_frames,
                  index_files=index_files, idx=num_vectors+1)
        num_vectors = collection_info.points_count
        print(num_vectors)
