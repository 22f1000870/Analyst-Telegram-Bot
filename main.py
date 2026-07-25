from dotenv import load_dotenv
import os
import json
from contextlib import asynccontextmanager
from telegram import Bot
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, PlainTextResponse
from openai import OpenAI
from datetime import timezone, datetime, timedelta
import uvicorn
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_KEY = os.getenv("GEMINI_API_KEY")

# North Flank provides PORT environment variable
PORT = int(os.getenv("PORT", 8000))

LOG_FILE = "run.jsonl"

client = OpenAI(api_key=API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

telegram_app = (
    ApplicationBuilder()
    .token(TELEGRAM_BOT_TOKEN)
    .build()
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await telegram_app.initialize()
    await telegram_app.start()
    
    
    base_url = os.getenv("PUBLIC_URL")  
    if base_url:
        webhook_url = f"{base_url}/webhook"
        await telegram_app.bot.set_webhook(webhook_url)
        print(f"Webhook set to: {webhook_url}")
    else:
        print("PUBLIC_URL not set, webhook not configured automatically")
    
    yield
    
    await telegram_app.stop()
    await telegram_app.shutdown()

app = FastAPI(lifespan=lifespan)

def get_base_url(request: Request) -> str:
    # Check for North Flank's PUBLIC_URL first
    public_url = os.getenv("PUBLIC_URL")
    if public_url:
        return public_url.rstrip("/")
    return str(request.base_url).rstrip("/")

@app.get("/")
async def root(request: Request):
    PUBLIC_BASE_URL = get_base_url(request)
    
    return {
        "status": "running",
        "base_url": PUBLIC_BASE_URL,
        "log_url": f"{PUBLIC_BASE_URL}/run.jsonl",
        "deployment": "North Flank"
    }

@app.get("/health")
async def health():
    """Health check endpoint for North Flank"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat()
    }

@app.get("/run.jsonl")
def log():
    if not os.path.exists(LOG_FILE):
        return PlainTextResponse("")
    return FileResponse(LOG_FILE, media_type="text/plain")

def log_event(event):
    ist = timezone(timedelta(hours=5, minutes=30))
    event["time"] = datetime.now(ist).isoformat()
    
    with open(LOG_FILE, 'a', encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

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

def answer(question, PUBLIC_BASE_URL):
    log_url = f"{PUBLIC_BASE_URL}/run.jsonl"
    
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
        "event": "llm_request",
        "question": question
    })
    
    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash-lite",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        
        raw_data = response.choices[0].message.content.strip()
        log_event({
            "event": "llm_response",
            "answer": raw_data
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
            "event": "json_error",
            "error": str(error),
            "answer": raw_data
        })
        
        data = {
            "error": "Unable to compute answer",
            "log_url": log_url
        }
        
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

async def handle_question(update, context):
    user = update.effective_user
    log_event({
        "event": "Telegram message",
        "user": user.username,
        "chat_id": update.message.chat_id,
        "first_name": user.first_name,
        "question": update.message.text
    })
    
    base_url = context.bot_data.get("base_url", os.getenv("PUBLIC_URL", "http://localhost:8000"))
    ans = answer(update.message.text, base_url)
    
    log_event({
        "event": "telegram_reply",
        "chat_id": update.message.chat_id,
        "reply": ans,
    })
    
    await update.message.reply_text(ans)

telegram_app.add_handler(
    MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_question
    )
)

@app.post("/webhook")
async def telegram_webhook(request: Request):
    try:
        base_url = get_base_url(request)
        telegram_app.bot_data["base_url"] = base_url
        
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
        
        return {"ok": True}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        log_event({
            "event": "webhook_error",
            "error": str(e)
        })
        
        return {"error": str(e)}

@app.get("/setWebhook")
async def set_webhook(request: Request):
    base_url = get_base_url(request)
    url = f"{base_url}/webhook"
    
    success = await telegram_app.bot.set_webhook(url)
    
    return {
        "success": success,
        "url": url,
        "message": "Webhook configured successfully" if success else "Webhook configuration failed"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)