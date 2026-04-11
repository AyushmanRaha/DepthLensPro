.PHONY: install backend frontend dev health

install:
	pip install -r backend/requirements.txt

backend:
	uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload

frontend:
	python3 -m http.server 5500 --directory frontend

dev:
	@echo "Run these in two terminals:"
	@echo "  make backend"
	@echo "  make frontend"

health:
	curl -s http://127.0.0.1:8000/health | python3 -m json.tool
