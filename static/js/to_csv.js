 function exportJSONToCSV(objArray) {
    var array = typeof objArray != 'object' ? JSON.parse(objArray) : objArray;
    var str = ""
    for(data of array){
        json_data = JSON.parse(data)
        str += json_data.Video_info + ',' + json_data.Frame_id +"\r\n"
    }
    var element = document.createElement('a');
    element.href = 'data:text/csv;charset=utf-8,' + encodeURI(str);
    element.target = '_blank';
    element.download = 'export.csv';
    element.click();
  }