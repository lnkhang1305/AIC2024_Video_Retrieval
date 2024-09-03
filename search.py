import clip
import torch
import faiss
from qdrant_client import QdrantClient
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


client = QdrantClient(url="http://localhost:6333")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/16", device=device)
index = faiss.read_index('index.ivf')


def search_images_with_text(query_text):
    # with torch.no_grad():
    #     text_vector = model.encode_text(open_clip.tokenize([query_text])).cpu().numpy().flatten()
    text_inputs = clip.tokenize(query_text).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_vector = model.encode_text(text_inputs)
    text_vector /= text_vector.norm(dim=-1, keepdim=True)

    # search_results = client.search(
    #     collection_name="image_collection",
    #     query_vector=text_vector,
    #     limit=5
    # )
    text_vector_np = text_vector.cpu().numpy().astype(np.float32)
    k = 5  # Số lượng kết quả trả về
    distances, indices = index.search(text_vector_np, k=k)

    result_ids = [int(idx) for idx in indices[0]]
    results = client.retrieve(
        collection_name='image_collection', ids=result_ids)

    top_5_images = []
    for result in results:
        print(
            f"ID: {result.id}, Path: {result.payload['image_path']},Video : {result.payload['video']},Frame_index: {result.payload['frame_idx']}, Distance: {distances[0][result_ids.index(result.id)]}")
        top_5_images.append(result)

    _, axs = plt.subplots(2, 3, figsize=(10, 7))
    for i, ax in enumerate(axs.flat[:-1]):
        ax.imshow(Image.open(top_5_images[i].payload['image_path']))
        ax.set_title(f'ID: {top_5_images[i].id}')
        ax.axis('off')

    plt.show()


search_images_with_text("A man in black vest")
