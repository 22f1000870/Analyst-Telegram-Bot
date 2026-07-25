from dotenv import load_dotenv
import os
import json
import threading
from telegram import Bot
from pathlib import Path
from fastapi import FastAPI,Request
from fastapi.responses import FileResponse,PlainTextResponse
from openai import OpenAI
from datetime import timezone, datetime,timedelta
import uvicorn
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

load_dotenv()

TELEGRAM_BOT_TOKEN= os.getenv("TELEGRAM_BOT_TOKEN")
API_KEY= os.getenv("GEMINI_API_KEY")
if os.getenv("VERCEL_URL"):
    PUBLIC_BASE_URL = f"https://{os.getenv('VERCEL_URL')}"
else:
    PUBLIC_BASE_URL = "http://localhost:8000"

LOG_FILE = "run.jsonl"

client = OpenAI(api_key=API_KEY,base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

app= FastAPI()

@app.get("/")
async def root():
    return {
        "status": "running",
        "base_url": PUBLIC_BASE_URL,
        "log_url": f"{PUBLIC_BASE_URL}/run.jsonl",
    }


@app.get("/run.jsonl")
def log():
    if not os.path.exists(LOG_FILE):
        return PlainTextResponse("")
    return FileResponse(LOG_FILE,media_type="text/plain")

def log_event(event):
    ist=timezone(timedelta(hours=5,minutes=30))
    event["time"]= datetime.now(ist).isoformat()

    with open(LOG_FILE,'a',encoding="utf-8") as f:
        f.write(json.dumps(event,ensure_ascii=False)+"\n")

def extract_json(text):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found")

    json_text = text[start:end + 1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        json_text = json_text.replace("\\", "\\\\")
        return json.loads(json_text)

def answer(question):

    log_url=f"{PUBLIC_BASE_URL}/run.jsonl"

    system_prompt = f"""
You are a data analysis Telegram bot.

The user may provide data inline or reference a public dataset.

Return ONLY one valid JSON object.

The JSON object MUST contain exactly two keys:

{{
  "answer": <the answer in exactly the shape requested by the user>,
  "log_url": "{log_url}"
}}

Example:

Question:
Which state has the highest maternal mortality rate?
Reply with ONLY:
{{"state":"<state name>"}}

Correct response:

{{
  "answer": {{
    "state": "Assam"
  }},
  "log_url": "{log_url}"
}}

output should be one single JSON Object

Do not output markdown.
Do not output explanations.
Do not output any text outside the JSON object.
"""
    log_event({
        "event":"llm_request",
        "question":question
    })

    try:
        response = client.chat.completions.create(
            model="gemini-3.1-flash-lite",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )

        raw_data = response.choices[0].message.content.strip()
        log_event({
                "event":"llm_response",
                "answer":raw_data
            })

    except Exception as error:
        log_event({
            "event": "llm_error",
            "error": str(error),
        })

        return json.dumps({
            "error": "credits problem",
            "log_url": log_url,
        }, ensure_ascii=False, separators=(",", ":"))

   

    try:
        data = extract_json(raw_data)

        if "answer" not in data:
            data = {
                "answer": data,
                "log_url": log_url
            }
        else:
            data["log_url"] = log_url

        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except Exception as error:

        log_event({
            "event":"json_error",
            "error":str(error),
            "answer":raw_data
        })

        data = {
            "error":"Unable to compute answer",
            "log_url":log_url
        }

        return json.dumps(data,ensure_ascii=False,separators=(",",":"))

async def handle_question(update,context):

    user = update.effective_user
    log_event({
        "event": "Telegram message",
        "user": user.username,
        "chat_id": update.message.chat_id,
        "first_name": user.first_name,
        "question": update.message.text
    })

    ans = answer(update.message.text)

    log_event({
        "event": "telegram_reply",
        "chat_id": update.message.chat_id,
        "reply": ans,
    })

    await update.message.reply_text(ans)

def run_web_server():
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


def run_telegram_bot():
    bot_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    bot_app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question)
    )

    bot_app.run_polling(stop_signals=None)



if __name__ == "__main__":
    threading.Thread(
        target=run_telegram_bot,
        daemon=True
    ).start()

    run_web_server()






