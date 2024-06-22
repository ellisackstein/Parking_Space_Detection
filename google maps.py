import webbrowser
import urllib.parse

#address = 'ramban 61 jerusalem'
#webbrowser.open('https://www.google.com/maps/place/' + address)

def open_google_maps(address):
    # Encode the address to make it URL-safe
    encoded_address = urllib.parse.quote(address)

    # Create the Google Maps URL with travel mode set to driving
    google_maps_url = f"https://www.google.com/maps/dir/?api=1&destination={encoded_address}&travelmode=driving"

    # Open the URL in the default web browser
    webbrowser.open(google_maps_url)

address = "Ramban 61, Jerusalem"
open_google_maps(address)