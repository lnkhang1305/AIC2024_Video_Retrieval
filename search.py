import clip
import torch
import faiss
from qdrant_client import QdrantClient
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import base64
import json
from deep_translator import GoogleTranslator
import csv
import os
import re


def init_model():
    device = "cpu"
    # print(device)
    model, preprocess = clip.load(r"D:\AI_chalenge_2024\AI_Challenge\model\ViT-B-16.pt", device=device)
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

def get_youtube_video_id_by_url(url):
    regex = r"^((https?://(?:www\.)?(?:m\.)?youtube\.com))/((?:oembed\?url=https?%3A//(?:www\.)youtube.com/watch\?(?:v%3D)(?P<video_id_1>[\w\-]{10,20})&format=json)|(?:attribution_link\?a=.*watch(?:%3Fv%3D|%3Fv%3D)(?P<video_id_2>[\w\-]{10,20}))(?:%26feature.*))|(https?:)?(\/\/)?((www\.|m\.)?youtube(-nocookie)?\.com\/((watch)?\?(app=desktop&)?(feature=\w*&)?v=|embed\/|v\/|e\/)|youtu\.be\/)(?P<video_id_3>[\w\-]{10,20})"
    match = re.match(regex, url, re.IGNORECASE)
    if match:
        return (
            match.group("video_id_1")
            or match.group("video_id_2")
            or match.group("video_id_3")
        )
    else:
        return ""

def search_images_from_query(query_text, k, model, index, client):
    path_to_media_info_folder = r"D:\AI_chalenge_2024\AI_Challenge\db\media-info-b1\media-info"
    text_inputs = clip.tokenize(query_text, truncate=True).to("cpu")
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_vector = model.encode_text(text_inputs)
    text_vector /= text_vector.norm(dim=-1, keepdim=True)
    text_vector_np = text_vector.cpu().numpy().astype(np.float32)
    distances, indices = index.search(text_vector_np, k=int(k))
    result_ids = [int(idx) for idx in indices[0]]
    results = client.retrieve(
        collection_name='image_collection', ids=result_ids)
    return_result = []
    pattern = r'L\d{2}_V\d{3}'
    for result in results:
        data = {}
        data['ID'] = result.id
        data['Video_info'] = re.search(pattern, result.payload['image_path']).group()
        with open(os.path.join(path_to_media_info_folder, data['Video_info'], ".json"), 'r', encoding='utf-8') as media_json_file:
            data['Youtube_id'] = get_youtube_video_id_by_url(json.load(media_json_file)['watch_url'])
        with open(result.payload['image_path'], "rb") as image_file:
            data['Image'] = base64.b64encode(image_file.read()).decode("utf-8")
        data['Video'] = result.payload['video']
        data['Frame_id'] = result.payload['frame_idx']
        return_result.append(json.dumps(data))
    return return_result

def to_csv(str, total_result, filepath):
    model, index, client = init_model()
    with open(filepath, mode='w', newline='') as csv_file:
        csv_data = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        retrieval_data = search_images_from_query(str, total_result, model, index, client)
        for data_json in retrieval_data:
            data = json.loads(data_json)
            csv_data.writerow([data['Video_info'], data['Frame_id']])

# k=100 #number of return results
# file_folder = r"C:\Users\ACER\Downloads\pack1-groupA\pack1-groupA" #path to folder contains queries
# save_result_folder = r"result" #path to folder will save results of queries
# index = 0
# for file_txt in os.listdir(file_folder):
#     print(index)
#     index = index + 1
#     # print(os.path.join(file_folder, file_txt))
#     with open(os.path.join(file_folder, file_txt), 'r', encoding="utf8") as file:
#         query_text = file.read()
#         # print(query_text)
#     file_name = os.path.splitext(file_txt)[0]
#     # print(os.path.join("result",file_name+".csv"))
#     to_csv(translate_to_EN(query_text), k, os.path.join(save_result_folder, file_name+".csv"))
# search_images_with_text(translate_to_EN("Một con thuyền chạy được trên băng, màu đen. Con thuyền này chạy bằng động cơ cánh quạt ở bên trên thổi hướng ra phía sau. Con thuyền là phương tiện hỗ trợ cứu hộ một nạn nhân bị rơi xuống hồ băng."))
