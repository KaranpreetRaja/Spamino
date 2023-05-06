(function(document) {
	'use strict';

	var TableFilter = (function(Arr) {

		var _input;

		function _onInputEvent(e) {
			_input = e.target;
			var tables = document.getElementsByTagName('table');
			Arr.forEach.call(tables, function(table) {
				Arr.forEach.call(table.tBodies, function(tbody) {
					Arr.forEach.call(tbody.rows, _filter);
				});
			});
		}

		function _filter(row) {
			if (row.className.indexOf('indexhead') != -1 || row.className.indexOf('parent') != -1) {
				return;
			}

			var text = row.getElementsByTagName('td')[1].textContent.toLowerCase();
			var val = _input.value.toLowerCase();

			row.style.display = text.indexOf(val) === -1 ? 'none' : 'table-row';
		}

		return {
			init: function() {
				var row = document.getElementsByTagName('tr')[0];
				if (row !== null && row.className.indexOf('indexhead') == -1) {
					row.className += ' indexhead';
				}

				row = document.getElementsByTagName('tr')[1];
				if (row !== null && row.getElementsByTagName('td')[1].textContent === 'Parent Directory') {
					row.className += ' parent';
				}

				document.getElementById('filter').oninput = _onInputEvent;
			}
		};

	})(Array.prototype);

	document.addEventListener('readystatechange', function() {
		if (document.readyState === 'complete') {
			TableFilter.init();
			var filterInput = document.getElementById('filter');
			if ( filterInput.value.trim().length ){
				filterInput.focus();
				filterInput.dispatchEvent(new Event('input'));
			}
		}
	});

	window.addEventListener('keydown', function (e) {
		var filterInput = document.getElementById('filter');
		var isFocused = (document.activeElement === filterInput);
		if ( !isFocused && String.fromCharCode(e.keyCode).match(/(\w|\s)/g) ) {
			filterInput.focus();
		} else {

		}
	});

})(document);

var uri = window.location.pathname.substr(1);
var arr = uri.split('/');
var url = ""
var bread = '<li><strong><a href="/">Home</a></strong></li>';
var cont = 1;
arr.forEach(function(value){
        url = url + '/' + value;
        if(value != ''){
            if(arr.length == cont+1)
                bread += "<li class='active'>"+decodeURI(value)+"</li>";
            else
                bread += "<li><a href='"+url+"'>"+decodeURI(value)+"</a></li>";
        }
        cont++;
});
document.getElementById("breadcrumb").innerHTML = bread;
if (uri.substring(uri.length-1) != '/'){
        var indexes = document.getElementsByClassName('indexcolname'),
        i = indexes.length;
        while (i--){
            var a = indexes[i].getElementsByTagName('a')[0];
            a.href =  uri + '/' + a.getAttribute('href',2);
        }
}
