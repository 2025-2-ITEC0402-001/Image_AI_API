from flask import Flask

def cpu_bound_task():
    total = 0
    for i in range(30_000_000):
        total += i

def create_app():
    app = Flask(__name__)

    @app.route("/test")
    def cpu_test():
        cpu_bound_task()
        return {"message": "Flask CPU Task OK"}

    return app

app = create_app() # 직접 실행을 위한 app 객체