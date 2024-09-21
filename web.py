from flask import Flask, render_template, request, Response
from search import search_images_from_query, translate_to_EN, init_model
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = Flask(__name__)
model_l14, model_b16, model_b32, index, client = init_model()


@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('index.html')


@app.route('/video_retrieval', methods=['GET', 'POST'])
def get_data():
    if(request.form['model'] == "l14"):
        return_data = search_images_from_query(translate_to_EN(request.form['query']), request.form['k'], model_l14, index, client, "l14_collection")
    elif(request.form['model'] == "b16"):
        return_data = search_images_from_query(translate_to_EN(request.form['query']), request.form['k'], model_b16, index, client, "b16_collection")
    else:
        return_data = search_images_from_query(translate_to_EN(request.form['query']), request.form['k'], model_b32, index, client, "b32_collection")
    return Response(json.dumps(return_data),  mimetype='application/json')
#     return Response(json.dumps([
#     json.dumps({
#         "ID": 1,
#         "Video_info": "L02_V030",
#         "Image": "",
#         "Video": "",
#         "Frame_id": "25777",
#         "Youtube_id": "aa8SKMkhKGE"
#     }),
#     json.dumps({
#         "ID": 2,
#         "Video_info": "L03_V040",
#         "Image": "",
#         "Video": "",
#         "Frame_id": "25787",
#         "Youtube_id":"ViAaamT3QmY"
#     })
# ]),  mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
