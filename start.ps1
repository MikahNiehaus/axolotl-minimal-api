# Start the minimal axolotl API server
$env:FLASK_APP = "app.py"
python -m flask run --host=0.0.0.0 --port=5000
