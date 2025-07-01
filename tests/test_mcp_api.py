import unittest
from unittest.mock import patch
from mcp_server import list_tasks, add_task


class TestMcpTools(unittest.TestCase):

    @patch("mcp_server.requests.get")
    def test_list_tasks(self, mock_get):
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = [{"id": 1, "title": "Mock Task"}]

        tasks = list_tasks()
        self.assertIsInstance(tasks, list)
        self.assertEqual(tasks[0]["title"], "Mock Task")

    @patch("mcp_server.requests.post")
    def test_add_task(self, mock_post):
        mock_post.return_value.status_code = 201
        mock_post.return_value.json.return_value = {"id": 2, "title": "New Task"}

        task = add_task("New Task")
        self.assertEqual(task["title"], "New Task")


if __name__ == "__main__":
    unittest.main()
