import uvicorn
from .config import setting, positive_int

if __name__ == '__main__':
    uvicorn.run('App.web_server:app', host=setting('OCR_WEB_HOST', '127.0.0.1'),
                port=positive_int('OCR_WEB_PORT', 8018), log_level=setting('LOG_LEVEL', 'info').lower())
