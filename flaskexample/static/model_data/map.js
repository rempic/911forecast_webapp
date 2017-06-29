var map;
var dataResults;

function initialize() {
    map = new google.maps.Map(document.getElementById('mapdisplay'), {
        zoom: 9,
        center: new google.maps.LatLng(40.010280,-75.295787),
    });

    addMarkers();
}

eqfeed_callback = function (results) {
    dataResults = results;
    }

function addMarkers () {
    for (var i = 0; i < dataResults.features.length; i++) {
        var quake = dataResults.features[i];
        var coors = quake.geometry.coordinates;
        var latLong = new google.maps.LatLng(coors[1], coors[0]);
        var image = "{{ url_for('static', filename='model_data/maps/blue_dot.png') }}";
        var marker = new google.maps.Marker({
            position: latLong,
            map: map,
            icon: image
        });
    }
}

google.maps.event.addDomListener(window, 'load', initialize);