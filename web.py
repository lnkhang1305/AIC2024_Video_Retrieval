from flask import Flask, render_template, request, Response
from search import search_images_from_query, translate_to_EN, init_model
import json

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def homepage():
    return render_template('index.html')

@app.route('/video_retrieval', methods=['GET','POST'])
def get_data():
    # return_data = search_images_from_query(translate_to_EN(request.form['query']), request.form['k'], model, index, client)
    # return Response(json.dumps(return_data),  mimetype='application/json')
    return Response(json.dumps([
    json.dumps({
        "ID": 1,
        "Video_info": "L02_V030",
        "Image": "",
        "Video": "",
        "Frame_id": "25777",
        "Youtube_id": "aa8SKMkhKGE"
    }),
    json.dumps({
        "ID": 2,
        "Video_info": "L03_V040",
        "Image": "",
        "Video": "",
        "Frame_id": "25787",
        "Youtube_id":"ViAaamT3QmY"
    })
]),  mimetype='application/json')

if __name__ == '__main__':
    model, index, client = init_model()
    app.run(host='0.0.0.0', debug = True)