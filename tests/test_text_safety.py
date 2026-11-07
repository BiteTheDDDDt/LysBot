import json
import sqlite3
import unittest

import nonebot

nonebot.init()

from src.plugins.llm_chat.models import ChatResponse, Message
from src.plugins.llm_chat.text_safety import unicode_safe


class UnicodePersistenceTests(unittest.TestCase):
    def test_message_from_json_preserves_pairs_and_repairs_lone_surrogates(self):
        raw = json.loads(
            '{"role":"user","content":"A\\ud83e B\\udd16 C\\ud83e\\udd16","name":"\\ud83e","reasoning_details":{"text":"\\ud83e"}}'
        )
        message = Message(**raw)
        self.assertEqual(message.content, "A\ufffd B\ufffd C\U0001f916")
        self.assertEqual(message.name, "\ufffd")
        self.assertEqual(message.reasoning_details, {"text": "\ufffd"})
        db = sqlite3.connect(":memory:")
        try:
            db.execute("CREATE TABLE messages(content TEXT)")
            db.execute("INSERT INTO messages VALUES(?)", (message.content,))
            self.assertEqual(db.execute("SELECT content FROM messages").fetchone()[0], message.content)
        finally:
            db.close()

    def test_response_and_nested_metadata_remain_utf8_encodable(self):
        response = ChatResponse("hello\ud83e", reasoning_details=[{"text": "\ud83e"}])
        self.assertEqual(str(response), "hello\ufffd")
        json.dumps(response.reasoning_details, ensure_ascii=False).encode("utf-8")
        valid = "中文🤖 café"
        self.assertIs(unicode_safe(valid), valid)


if __name__ == "__main__":
    unittest.main()
