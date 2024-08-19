import torch
import os
from PIL import Image
import open_clip
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, CollectionParams


def add_image_to_qdrant(image_path, image_id):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0) 
    
    with torch.no_grad():
        image_vector = model.encode_image(image_tensor).cpu().numpy().flatten() 
    
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
for folder in os.listdir('keyframes'):
    for image_name in os.listdir('keyframes'+'/'+folder):
        image_path = 'keyframes' + '/' + folder + '/' + image_name
        add_image_to_qdrant(image_path, id)
        id += 1
