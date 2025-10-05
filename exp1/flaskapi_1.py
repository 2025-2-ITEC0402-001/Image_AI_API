import time
from flask import Flask

app = Flask(__name__)

@app.route("/test")
def sync_test():
    time.sleep(1)
    return {"message": "Flask OK"}
