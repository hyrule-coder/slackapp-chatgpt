import os
import logging
import json
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from slack_bolt.adapter.socket_mode import SocketModeHandler
import re
from langchain_openai import ChatOpenAI
import time
from typing import Any
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from datetime import timedelta
from langchain_community.chat_message_histories import MomentoChatMessageHistory

CHAT_UPDATE_INTERVAL_SEC=1

load_dotenv()

#ログ
SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

app = App(
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
    token=os.environ.get("SLACK_BOT_TOKEN"),
    process_before_response=True,
    )


class SlackStreamingCallbackHandler(BaseCallbackHandler):
    last_send_time = time.time()
    message = ""

    def __init__(self, channel, ts):
        self.channel = channel
        self.ts = ts
        self.interval = CHAT_UPDATE_INTERVAL_SEC
        self.update_count = 0

    def on_llm_new_token(self, token: str, **kwargs) ->None:
        self.message += token

        now = time.time()
        if now - self.last_send_time > self.interval:
            app.client.chat_update(
                channel=self.channel, ts=self.ts, text=f"{self.message}\n\nTyping...",
            )
            self.last_send_time = now
            self.update_count += 1

            if self.update_count / 10 > self.interval:
                self.interval = self.interval * 2

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        message_context = "OpenAI APIで生成される情報は不正確または不適切な場合がありますが、当社の見解を述べるものではありません。"
        message_blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": self.message}},
            {"type": "divider"},
            {"type": "context", 
            "elements": [{"type": "mrkdwn", "text": message_context}],
            },
        ]
        app.client.chat_update(
            channel=self.channel, 
            ts=self.ts,
            text=self.message,
            blocks=message_blocks,
        )



@app.event("app_mention")
def handle_mention(event, say):
    channel = event["channel"]
    thread_ts = event["ts"]
    message = re.sub("<@.*?>", "", event["text"])

    id_ts = event["ts"] 
    if "thread_ts" in event:
        id_ts = event["thread_ts"]

    result = say("\n\nTyping...", thread_ts=thread_ts)
    ts = result["ts"] 

    history = MomentoChatMessageHistory.from_client_params(
        id_ts,
        os.environ.get("MOMENTO_CACHE"),
        timedelta(hours=int(os.environ.get("MOMENTO_TTL"))),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a good assistant."),
            (MessagesPlaceholder(variable_name="chat_history")),
            ("user", "{input}"),
        ]
    )

    callback = SlackStreamingCallbackHandler(channel=channel, ts=ts)

    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_API_MODEL"),
        temperature=os.environ.get("OPENAI_API_TEMPERATURE"),
        streaming=True,
        callbacks=[callback],
    )

    chain = prompt | llm | StrOutputParser()

    ai_message = chain.invoke({"input": message, "chat_history": history.messages})

    history.add_user_message(message)
    history.add_ai_message(ai_message)



if __name__ == "__main__":
    SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN")).start()

def handler(event, context):
    logger.info("handler called")
    header = event["headers"]
    logger.info(json.dumps(header))

    if "x-slack-retry-num" in header:
        logger.info("SKIP > x-slack-retry-num: %s", header["x-slack-retry-num"])
        return 200

    # AWS Lambda 環境のリクエスト情報を app が処理できるよう変換してくれるアダプター
    slack_handler = SlackRequestHandler(app=app)
    # 応答はそのまま AWS Lambda の戻り値として返せます
    return slack_handler.handle(event, context)
