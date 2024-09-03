import torch
import os
import numpy as np
from PIL import Image
import open_clip
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, CollectionParams


def add_image_to_qdrant(image_path, image_id):
    image_vector = np.load(image_path) 
    
    client.upsert(
        collection_name="image_collection",
        points=[
            {
                "id": image_id,
                "vector": image_vector,
                "payload": {"image_path": image_path}
            }
        ]
    )


client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name='image_collection',
    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
)

model, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
id = 1
image_path = "C:\Users\Acer\Desktop\pipeline_test\L02_CLIP_Features\content\L02_CLIP_Features"
for part in sorted(os.listdir(image_path)):
  for image_file in sorted(os.listdir(part)):
    path = image_path+'/'+part+'/'+image_file
    add_image_to_qdrant(path,id)
    id += 1