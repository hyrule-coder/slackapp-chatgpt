import os
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import re
from langchain_openai import ChatOpenAI
import time
from typing import Any
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

CHAT_UPDATE_INTERVAL_SEC=1

load_dotenv()

app = App(
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET"),
    token=os.environ.get("SLACK_BOT_TOKEN"),
    process_before_response=True,
    )

@app.event("app_mention")
def handle_mention(event, say):
    channel = event["channel"]
    thread_ts = event["ts"]
    thread_ts = event["ts"]
    message = re.sub("<@.*?>", "", event["text"])

    result = say("\n\nTyping...", thread_ts=thread_ts)
    ts = result["ts"] 

    callback = SlackStreamingCallbackHandler(channel=channel, ts=ts)

    llm = ChatOpenAI(
        model=os.environ.get("OPENAI_API_MODEL"),
        temperature=os.environ.get("OPENAI_API_TEMPERATURE"),
        streaming=True,
        callbacks=[callback],
    )
    response = llm.invoke(message)



class SlackStreamingCallbackHandler(BaseCallbackHandler):
    last_send_time = time.time()
    message = ""

    def __init__(self, channel, ts):
        self.channel = channel
        self.ts = ts

    def on_llm_new_token(self, token: str, **kwargs) ->None:
        self.message += token

        now = time.time()
        if now - self.last_send_time > CHAT_UPDATE_INTERVAL_SEC:
            self.last_send_time = now
            app.client.chat_update(
                channel=self.channel,
                ts=self.ts,
                text=f"{self.message}...",
            )

    def on_llm_result(self, response: LLMResult, **kwargs) -> None:
        app.client.chat_update(
            channel=self.channel,
            ts=self.ts,
            text=self.message,
        )


if __name__ == "__main__":
    SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN")).start()
