import open_clip
import torch
from qdrant_client import QdrantClient
import matplotlib.pyplot as plt
from PIL import Image


client = QdrantClient(url="http://localhost:6333")


model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')


def search_images_with_text(query_text):
    with torch.no_grad():
        text_vector = model.encode_text(open_clip.tokenize([query_text])).cpu().numpy().flatten()
    
    search_results = client.search(
        collection_name="image_collection",
        query_vector=text_vector,
        limit=5  
    )
    top_5_images = []
    for result in search_results:
        print(f"ID: {result.id}, Image Path: {result.payload['image_path']}, Score: {result.score}")
        top_5_images.append(result)

    _, axs = plt.subplots(2, 3, figsize=(10, 7))
    for i, ax in enumerate(axs.flat[:-1]):
        ax.imshow(Image.open(top_5_images[i].payload['image_path']))
        ax.set_title(f'ID: {top_5_images[i].id}')
        ax.axis('off')

    plt.show()

search_images_with_text("two men are shaking hand and there are two women stading next to them")
