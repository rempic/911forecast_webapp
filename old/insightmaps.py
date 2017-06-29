#import gmaps
#import gmaps.datasets
#import json




def show_map(df_locations_ltln,geojson_file, my_apikey, maker_col, maker_size):
    gmaps.configure(api_key=my_apikey) 

    calls_today_layer = gmaps.symbol_layer(
        df_locations_ltln, fill_color=maker_col, stroke_color=maker_col, scale=maker_size
    )
    
    with open(geojson_file) as f:
        geometry = json.load(f)
        
    geojson_layer = gmaps.geojson_layer(geometry)
    fig = gmaps.figure()
    
    fig.add_layer(calls_today_layer)
    fig.add_layer(geojson_layer)
    return fig
