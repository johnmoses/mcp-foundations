""" 
A simple Flask app exposing a to-do list API:
"""

from flask import Flask, request, jsonify

app = Flask(__name__)

tasks = [{"id": 1, "title": "Sample Task", "done": False}]

@app.route('/tasks', methods=['GET'])
def get_tasks():
    return jsonify(tasks)

@app.route('/tasks', methods=['POST'])
def create_task():
    data = request.get_json()
    new_task = {
        "id": len(tasks) + 1,
        "title": data.get("title"),
        "done": False
    }
    tasks.append(new_task)
    return jsonify(new_task), 201

if __name__ == '__main__':
    # Run Flask app on port 5001
    app.run(host='127.0.0.1', port=5001, debug=True)
