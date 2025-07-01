import unittest
from flask_server import app  # Your Flask app module

class FlaskApiTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_get_tasks(self):
        response = self.client.get('/tasks')
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.get_json(), list)

    def test_create_task(self):
        response = self.client.post('/tasks', json={"title": "Test Task"})
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertIn("id", data)
        self.assertEqual(data["title"], "Test Task")

if __name__ == '__main__':
    unittest.main()
