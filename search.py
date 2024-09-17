import clip
import torch
import faiss
from qdrant_client import QdrantClient
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import base64
import json
from deep_translator import GoogleTranslator

# ADJUST PARAMETER AND MODEL--------------#
CLIP_MODEL = "ViT-L/14@336px" # ViT-B/16  #
COLLECTION_NAME = "image_collection"      #
# ----------------------------------------#

def init_model():
    device = "cpu"
    # print(device)
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    index = faiss.read_index('index.ivf')
    client = QdrantClient(url="http://localhost:6333")
    return model, index, client


def search_images_with_text(query_text, device, model, index, client):
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
    k = 10  # Số lượng kết quả trả về
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
        path = top_5_images[i].payload['image_path']
        path = path.replace("./keyframes/Keyframes_"+top_5_images[i].payload['video'], "D:\AI_chalenge_2024\AI_Challenge\db\\videos\\Keyframes_"+top_5_images[i].payload['video']+"\\keyframes")
        ax.imshow(Image.open(path))
        ax.set_title(f'ID: {top_5_images[i].id}')
        ax.axis('off')

    plt.show()

def translate_to_EN(query):
    return GoogleTranslator(source='auto', target='en').translate(text=query)
# def extract_keyword(text):

def search_images_from_query(query_text, k, model, index, client):
    buffered = io.BytesIO()
    text_inputs = clip.tokenize(query_text).to("cpu")
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_vector = model.encode_text(text_inputs)
    text_vector /= text_vector.norm(dim=-1, keepdim=True)
    text_vector_np = text_vector.cpu().numpy().astype(np.float32)
    distances, indices = index.search(text_vector_np, k=int(k))
    result_ids = [int(idx) for idx in indices[0]]
    results = client.retrieve(
        collection_name=COLLECTION_NAME, ids=result_ids)
    return_result = []
    for result in results:
        img = Image.open(result.payload['image_path'])
        img.save(buffered, format="JPEG")
        data = {}
        data['ID'] = result.id
        data['Image'] = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data['Video'] = result.payload['video']
        data['Frame_id'] = result.payload['frame_idx']
        print(data)
        return_result.append(json.dumps(data))
    return return_result


# search_images_with_text(translate_to_EN("Một con thuyền chạy được trên băng, màu đen. Con thuyền này chạy bằng động cơ cánh quạt ở bên trên thổi hướng ra phía sau. Con thuyền là phương tiện hỗ trợ cứu hộ một nạn nhân bị rơi xuống hồ băng."))
