from flask import Flask, render_template
#from example import free_spots_detection_func
from main import run

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('homeScreen.html')


@app.route('/run-script')
def run_script():
    # example
    # address_photos = free_spots_detection_func()

    # our algorithm
    address_photos = run()
    return render_template('parkingDisplay.html', address_photos=address_photos)


def my_function():
    return render_template('parkingDisplay.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
