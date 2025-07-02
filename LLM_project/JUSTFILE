# Production-ready Justfile for Data Analysis Assistant
set shell := ["bash", "-ce"]
set dotenv-load

# --- Configuration ---
venv := ".venv"
host := "0.0.0.0"
port := "8000"
keepalive := "300"
ollama_model := "llama3:8b"
log_dir := "logs"

# --- Main Commands ---
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    
    echo "🚀 Setting up project environment..."
    
    # 1. Create Python virtualenv
    python3 -m venv {{venv}} || { echo "❌ Failed to create virtualenv"; exit 1; }
    
    # 2. Install Python dependencies
    source {{venv}}/bin/activate && \
    pip install --upgrade "pip>=24.0" && \
    pip install \
        --no-cache-dir \
        -r requirements.txt || { echo "❌ Dependency installation failed"; exit 1; }
    
    # 3. Install Ollama if missing
    if ! command -v ollama &> /dev/null; then
        echo "📦 Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh || { echo "❌ Ollama installation failed"; exit 1; }
    fi
    
    # 4. Create log directory
    mkdir -p {{log_dir}}
    
    # 5. Pull the LLM model
    echo "🔍 Pulling {{ollama_model}}..."
    ollama pull {{ollama_model}} || { echo "❌ Model pull failed"; exit 1; }
    
    echo "✅ Setup completed successfully"

run:
    #!/usr/bin/env bash
    set -euo pipefail
    
    # Verify setup
    if [ ! -f "{{venv}}/bin/activate" ]; then
        echo "❌ Error: Virtualenv missing. Run 'just setup' first"
        exit 1
    fi
    
    # Verify Ollama service is running
    if ! pgrep -x "ollama" > /dev/null; then
        echo "⚠️  Ollama service not running. Starting..."
        ollama serve > {{log_dir}}/ollama.log 2>&1 &
        sleep 5
    fi
    
    # Verify model availability
    if ! ollama list | grep -q "llama3"; then
        echo "⚠️  No llama3 model found. Pulling..."
        ollama pull {{ollama_model}} || { echo "❌ Failed to pull model"; exit 1; }
    fi
    
    echo "🚀 Starting services..."
    
    # Create timestamp for log files
    timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # Activate virtualenv
    source {{venv}}/bin/activate
    
    # Start logging server
    python3 logging_server.py > {{log_dir}}/logging_server_${timestamp}.log 2>&1 &
    LOG_SERVER_PID=$!
    echo "📝 Logging server started (PID: $LOG_SERVER_PID)"
    
    # Start Uvicorn with proper process management
    uvicorn app:app \
        --host {{host}} \
        --port {{port}} \
        --timeout-keep-alive {{keepalive}} \
        > {{log_dir}}/uvicorn_${timestamp}.log 2>&1 &
    UVICORN_PID=$!
    echo "🌐 Uvicorn started (PID: $UVICORN_PID)"
    
    # Wait for Uvicorn to start
    sleep 5
    
    # Verify Uvicorn is running
    if ! ps -p $UVICORN_PID > /dev/null; then
        echo "❌ Uvicorn failed to start. Check {{log_dir}}/uvicorn_${timestamp}.log"
        exit 1
    fi
    
    # Start Streamlit
    streamlit run st.py > {{log_dir}}/streamlit_${timestamp}.log 2>&1

clean:
    #!/usr/bin/env bash
    echo "🧹 Cleaning project..."
    rm -rf \
        {{venv}} \
        __pycache__ \
        .pytest_cache \
        *.log \
        {{log_dir}}/*
    echo "✅ Clean complete"

logs:
    #!/usr/bin/env bash
    echo "📋 Recent logs:"
    ls -lt {{log_dir}} | head -n 5

status:
    #!/usr/bin/env bash
    echo "🔄 Service status:"
    pgrep -fla "python3\|uvicorn\|streamlit\|ollama" || echo "No services running"