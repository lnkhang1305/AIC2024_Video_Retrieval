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
import re

def init_model():
    device = "cpu"
    # print(device)
    model, preprocess = clip.load(r"ViT-L/14@336px", device=device)
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
        # print(
        #     f"ID: {result.id}, Path: {result.payload['image_path']},Video : {result.payload['video']},Frame_index: {result.payload['frame_idx']}, Distance: {distances[0][result_ids.index(result.id)]}")
        top_5_images.append(result)

    _, axs = plt.subplots(2, 3, figsize=(10, 7))
    for i, ax in enumerate(axs.flat[:-1]):
        path = top_5_images[i].payload['image_path']
        # print(path)
        path = path.replace("./keyframes/Keyframes_"+top_5_images[i].payload['video'], "D:\Python\AIC_2024\pipeline_test\keyframes\Keyframes_"+top_5_images[i].payload['video'])
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
        collection_name='image_collection', ids=result_ids)
    return_result = []
    

    for result in results:
        result.payload['image_path'] = result.payload['image_path'].replace("./keyframes/Keyframes_"+ result.payload['video'] , "D:\Python\AIC_2024\pipeline_test\keyframes\Keyframes_" + result.payload['video'] + "\keyframes")
        print(result.payload['image_path'])
        frame_index = "0" * (3 - len(str(read_index))) + ".jpg"
        img = Image.open(result.payload['image_path'] + '\' + frame_index)
        img.save(buffered, format="JPEG")
        data = {}
        data['ID'] = result.id
        with open(result.payload['image_path'], "rb") as f:
            data['Image'] = base64.b64encode(f.read()).decode("utf-8")
        pattern = r'L\d{2}_V\d{3}'
        data['Video'] = re.search(pattern, result.payload['image_path']).group()
        data['Frame_id'] = result.payload['frame_idx']
        # print(data)
        return_result.append(json.dumps(data))

    return return_result


# search_images_with_text(translate_to_EN("Một con thuyền chạy được trên băng, màu đen. Con thuyền này chạy bằng động cơ cánh quạt ở bên trên thổi hướng ra phía sau. Con thuyền là phương tiện hỗ trợ cứu hộ một nạn nhân bị rơi xuống hồ băng."))

# def to_csv(str, total_result, filepath):
#     model, index, client = init_model()
#     with open(filepath, mode='w') as csv_file:
#         csv_data = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         retrieval_data = search_images_from_query(str, total_result, model, index, client)
#         # print(retrieval_data)
#         for data_json in retrieval_data:
#             data = json.loads(data_json)
#             # print(json.loads(data)['ID'])
#             csv_data.writerow([data['ID'], data['Video'], data['Frame_id']])

# query_text = "Trận bóng đá giữa đội Uzbekistan và đội Triều Tiên. Một đội trong trang phục toàn trắng và đội còn lại trong trang phục toàn xanh dương. Đây là thời điểm đội Triều Tiên được hưởng quả phạt đền 11 mét. Hỏi lúc thực hiện quả phạt đền có bao nhiêu cầu thủ Uzbekistan đang ở trong khung hình?"
# k=1
# file_path = "submission.csv"
# to_csv(translate_to_EN(query_text), k, file_path)