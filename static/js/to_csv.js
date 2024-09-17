 function exportJSONToCSV(objArray, col) {
    var array = typeof objArray != 'object' ? JSON.parse(objArray) : objArray;
    var str = ""
    for(data of array){
        json_data = data
        row = []
        for(c of col){
          row.push(json_data[c])
        }
        str += row.join(',') +"\r\n"
    }
    var element = document.createElement('a');
    element.href = 'data:text/csv;charset=utf-8,' + encodeURI(str);
    element.target = '_blank';
    element.download = 'export.csv';
    element.click();
  }