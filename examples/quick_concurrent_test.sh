#!/bin/bash

# Quick Concurrent Test for MLX Speculative Server
# This script sends multiple requests simultaneously using curl

SERVER_URL="http://localhost:8000"
CONCURRENCY=10
MAX_TOKENS=50

echo "🚀 Quick Concurrent Test for MLX Speculative Server"
echo "=================================================="
echo "Server: $SERVER_URL"
echo "Concurrency: $CONCURRENCY requests"
echo "Max tokens per request: $MAX_TOKENS"
echo ""

# Check if server is running
echo "🔍 Checking server health..."
if ! curl -s "$SERVER_URL/health" > /dev/null; then
    echo "❌ Server is not running at $SERVER_URL"
    echo "💡 Start the server with: python -m mlx_speculative.cli_enhanced serve --model-path <model>"
    exit 1
fi

echo "✅ Server is running"
echo ""

# Test prompts
prompts=(
    "What is artificial intelligence?"
    "Explain quantum computing."
    "How does machine learning work?"
    "What is the future of AI?"
    "Describe neural networks."
    "What is deep learning?"
    "How do transformers work?"
    "Explain natural language processing."
    "What is computer vision?"
    "How does reinforcement learning work?"
)

# Create temporary directory for results
TEMP_DIR=$(mktemp -d)
echo "📁 Storing results in: $TEMP_DIR"

# Function to send a single request
send_request() {
    local id=$1
    local prompt=$2
    local start_time=$(date +%s.%N)
    
    local response=$(curl -s -X POST "$SERVER_URL/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"$prompt\",
            \"max_tokens\": $MAX_TOKENS,
            \"temperature\": 0.7
        }")
    
    local end_time=$(date +%s.%N)
    local client_elapsed=$(echo "$end_time - $start_time" | bc -l)
    
    # Extract metrics from response
    local server_elapsed=$(echo "$response" | jq -r '.performance.elapsed_time // 0')
    local throughput=$(echo "$response" | jq -r '.performance.throughput // 0')
    local acceptance_rate=$(echo "$response" | jq -r '.speculative_stats.acceptance_rate // 0')
    local speedup=$(echo "$response" | jq -r '.speculative_stats.speedup // 0')
    local tokens=$(echo "$response" | jq -r '.usage.completion_tokens // 0')
    
    # Save result
    echo "$id,$client_elapsed,$server_elapsed,$throughput,$acceptance_rate,$speedup,$tokens" >> "$TEMP_DIR/results.csv"
    
    echo "✅ Request $id completed in ${client_elapsed}s (${throughput} tok/s, ${speedup}x speedup)"
}

# Start concurrent requests
echo "🔥 Starting $CONCURRENCY concurrent requests..."
echo "request_id,client_time,server_time,throughput,acceptance_rate,speedup,tokens" > "$TEMP_DIR/results.csv"

start_time=$(date +%s.%N)

# Launch requests in parallel
for i in $(seq 1 $CONCURRENCY); do
    prompt_index=$((($i - 1) % ${#prompts[@]}))
    prompt="${prompts[$prompt_index]}"
    send_request $i "$prompt" &
done

# Wait for all requests to complete
wait

end_time=$(date +%s.%N)
total_elapsed=$(echo "$end_time - $start_time" | bc -l)

echo ""
echo "📊 CONCURRENT TEST RESULTS"
echo "=========================="
echo "Total time: ${total_elapsed}s"
echo "Requests per second: $(echo "scale=2; $CONCURRENCY / $total_elapsed" | bc -l)"

# Analyze results
if command -v python3 &> /dev/null; then
    echo ""
    echo "📈 Detailed Analysis:"
    python3 -c "
import csv
import statistics

with open('$TEMP_DIR/results.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

if data:
    client_times = [float(row['client_time']) for row in data if float(row['client_time']) > 0]
    server_times = [float(row['server_time']) for row in data if float(row['server_time']) > 0]
    throughputs = [float(row['throughput']) for row in data if float(row['throughput']) > 0]
    acceptance_rates = [float(row['acceptance_rate']) for row in data if float(row['acceptance_rate']) > 0]
    speedups = [float(row['speedup']) for row in data if float(row['speedup']) > 0]
    tokens = [int(row['tokens']) for row in data if int(row['tokens']) > 0]
    
    total_tokens = sum(tokens)
    
    print(f'Successful requests: {len([r for r in data if float(r[\"throughput\"]) > 0])}/{len(data)}')
    print(f'Average client time: {statistics.mean(client_times):.2f}s')
    print(f'Average server time: {statistics.mean(server_times):.2f}s')
    print(f'Average throughput: {statistics.mean(throughputs):.1f} tok/s')
    print(f'Total tokens generated: {total_tokens}')
    print(f'Total throughput: {total_tokens / $total_elapsed:.1f} tok/s')
    print(f'Average acceptance rate: {statistics.mean(acceptance_rates):.1%}')
    print(f'Average speedup: {statistics.mean(speedups):.1f}x')
    print(f'Throughput range: {min(throughputs):.1f} - {max(throughputs):.1f} tok/s')
"
else
    echo "Install python3 for detailed analysis"
fi

echo ""
echo "📄 Raw results saved to: $TEMP_DIR/results.csv"
echo "🎉 Test completed!"

# Cleanup option
echo ""
read -p "🗑️  Delete temporary files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$TEMP_DIR"
    echo "✅ Temporary files deleted"
else
    echo "📁 Results preserved in: $TEMP_DIR"
fi
