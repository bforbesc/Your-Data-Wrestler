import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import utils


class OpenAIResponsesApiTest(unittest.TestCase):
    def setUp(self):
        utils.get_openai_response.cache_clear()
        utils.response_cache.clear()

    def test_get_openai_response_uses_responses_api_output_text(self):
        fake_client = Mock()
        fake_client.responses.create.return_value = SimpleNamespace(output_text="Hello from Responses API")

        with patch("utils._get_openai_client", return_value=fake_client), patch.dict(
            "utils.os.environ", {"OPENAI_MODEL": "gpt-4.1-mini"}, clear=False
        ):
            result = utils.get_openai_response("Say hello")

        self.assertEqual(result, "Hello from Responses API")
        fake_client.responses.create.assert_called_once_with(
            model="gpt-4.1-mini",
            input="Say hello",
            temperature=0.7,
            max_output_tokens=500,
        )

    def test_extract_response_text_falls_back_to_output_content(self):
        response = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "First line"},
                        {"type": "output_text", "text": "Second line"},
                    ]
                }
            ]
        }

        self.assertEqual(utils._extract_response_text(response), "First line\nSecond line")


if __name__ == "__main__":
    unittest.main()
