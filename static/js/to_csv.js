 function exportJSONToCSV(objArray, col, max_frame=100) {
    var array = typeof objArray != 'object' ? JSON.parse(objArray) : objArray;
    var str = ""
    var index = 0
    for(data of array && index < max_frame){
        json_data = data
        row = []
        for(c of col){
          row.push(json_data[c])
        }
        str += row.join(',') +"\r\n"
        index = index + 1
    }
    var element = document.createElement('a');
    element.href = 'data:text/csv;charset=utf-8,' + encodeURI(str);
    element.target = '_blank';
    element.download = 'export.csv';
    element.click();
  }