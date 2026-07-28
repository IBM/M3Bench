#!/bin/bash
set -e

echo "=== M3 Unified Container Starting ==="

# --- Optional: pull data from COS to local disk (DATA_SOURCE=sync) ---
# Default (unset) or DATA_SOURCE=mount -> data is provided by a COS pds mount (or a
# local bind mount) at the paths below; nothing to download here.
# DATA_SOURCE=sync -> download from COS to ephemeral disk before starting servers
# (fallback path; gives full local-fs semantics, e.g. writable ChromaDB for cap 4).
if [ "$DATA_SOURCE" = "sync" ]; then
    echo "DATA_SOURCE=sync -> pulling data from COS to local disk (this can take a while)..."
    python /app/cos_sync.py --prefix databases --dest /app/db
    python /app/cos_sync.py --prefix configs   --dest /app/environment/configs
    if [ "$CAPABILITY_ID" = "4" ]; then
        python /app/cos_sync.py --prefix indexed_documents --dest /app/retrievers/chroma_data
        python /app/cos_sync.py --prefix queries          --dest /app/retrievers/queries || true
    fi
    echo "DATA_SOURCE=sync -> data ready."
fi

# --- Start M3 REST FastAPI on port 8000 ---
echo "Starting M3 REST FastAPI on port 8000..."
cd /app/m3-rest
uvicorn app:app --host 0.0.0.0 --port 8000 &
M3_PID=$!
cd /app

# --- Start Retriever FastAPI on port 8001 (skip if no chroma_data) ---
if [ -d "/app/retrievers/chroma_data" ] && [ -n "$(ls -A /app/retrievers/chroma_data 2>/dev/null)" ]; then
    echo "Starting Retriever FastAPI on port 8001..."
    cd /app/retrievers
    uvicorn server:app --host 0.0.0.0 --port 8001 &
    RETRIEVER_PID=$!
    cd /app
    RETRIEVER_AVAILABLE=true
else
    echo "WARNING: No chroma_data/ found — Retriever server will NOT start."
    RETRIEVER_AVAILABLE=false
fi

# --- Wait for M3 REST FastAPI ---
echo "Waiting for M3 REST FastAPI (port 8000)..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/openapi.json > /dev/null 2>&1; then
        echo "M3 REST FastAPI is ready."
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: M3 REST FastAPI did not start within 60 seconds."
        exit 1
    fi
    sleep 1
done

# --- Wait for Retriever FastAPI (longer timeout — loads embeddings model) ---
if [ "$RETRIEVER_AVAILABLE" = "true" ]; then
    echo "Waiting for Retriever FastAPI (port 8001)..."
    for i in $(seq 1 300); do
        if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
            echo "Retriever FastAPI is ready."
            break
        fi
        if [ "$i" -eq 300 ]; then
            echo "ERROR: Retriever FastAPI did not start within 300 seconds."
            exit 1
        fi
        sleep 1
    done
fi

echo "=== All services running. ==="

# --- Serve mode ---
# SERVE_MODE=http  -> start the MCP stdio->HTTP bridge (Code Engine / remote clients).
#                     The FastAPI backends above stay up (the MCP servers call them);
#                     the bridge exposes MCP over HTTP at /mcp/<domain> on BRIDGE_PORT.
# (unset/default)  -> keep the container alive for `docker exec -i` (local benchmark flow).
if [ "$SERVE_MODE" = "http" ]; then
    echo "SERVE_MODE=http -> starting MCP HTTP bridge on port ${BRIDGE_PORT:-8080} (endpoint: /mcp/<domain>)"
    exec python /app/mcp_http_bridge.py --port "${BRIDGE_PORT:-8080}"
else
    echo "Container ready for exec."
    # Keep container alive for docker exec -i usage
    exec tail -f /dev/null
fi
