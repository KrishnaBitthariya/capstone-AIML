#!/bin/bash
# ── NIDS Project Setup & Launch ───────────────────────────────────────────────
# Run once:   bash setup.sh
# Run again:  bash run.sh

set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   NIDS — Network Intrusion Detection System          ║"
echo "║   Capstone Project Setup                             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# 1. Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
echo "✅ venv created"

# 2. Activate & install
echo ""
echo "📥 Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Dependencies installed"

# 3. Train the ML model
echo ""
echo "🤖 Training ML model..."
python ml/train_model.py

# 4. Launch Flask
echo ""
echo "🚀 Starting dashboard → http://127.0.0.1:5000"
echo "   Press Ctrl+C to stop"
echo ""
python app.py
