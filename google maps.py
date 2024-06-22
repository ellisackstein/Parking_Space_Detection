import webbrowser
import urllib.parse
import requests


# address = 'ramban 61 jerusalem'
# webbrowser.open('https://www.google.com/maps/place/' + address)

def open_google_maps(address):
    # Encode the address to make it URL-safe
    encoded_address = urllib.parse.quote(address)

    # Create the Google Maps URL with travel mode set to driving
    google_maps_url = f"https://www.google.com/maps/dir/?api=1&destination={encoded_address}&travelmode=driving"

    # Open the URL in the default web browser
    webbrowser.open(google_maps_url)


# def get_current_location():
    # # Use geocoder to get the current location based on IP address
    # g = ge.ip('me')
    # geopy.geocoders.
    # return g.latlng


def get_eta(api_key, origin_coords, destination):
    # Base URL for the Google Maps Directions API
    base_url = "https://maps.googleapis.com/maps/api/directions/json"

    # Format the origin coordinates
    origin = f"{origin_coords[0]},{origin_coords[1]}"

    # Parameters for the API request
    params = {
        "origin": origin_coords,
        "destination": destination,
        "mode": "driving",
        "key": api_key
    }

    # Make the request to the Directions API
    response = requests.get(base_url, params=params)

    # Parse the response JSON to get the ETA
    if response.status_code == 200:
        directions = response.json()
        if directions['routes']:
            duration = directions['routes'][0]['legs'][0]['duration']['text']
            return duration
        else:
            return "No routes found"
    else:
        return f"Error: {response.status_code}"


# Example usage with an API key and destination address
api_key = "AIzaSyCo2cbObmlr74x7RInQerbI50FjhQCpdCA"
destination = "Ramban 61, Jerusalem"

current_location = 'bialik 17, tel aviv'
eta = get_eta(api_key, current_location, destination)
print(f"Estimated travel time: {eta}")

#address = "Ramban 61, Jerusalem"
#open_google_maps(address)



