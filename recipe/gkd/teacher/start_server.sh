# ==============================================================================
# GKD 教师模型服务端启动脚本
# ==============================================================================
# 用途：启动教师模型推理服务，包括 proxy（代理）和 worker（推理进程）
# 关键参数：
#   - PROXY_FRONTEND_PORT: 客户端连接端口（默认 15555）
#   - PROXY_BACKEND_PORT: worker 连接端口（默认 15556）
#   - BACKEND: 推理引擎后端（默认 vllm）
#   - CKPT_PATH: 教师模型检查点路径
# 使用方法：修改 CKPT_PATH 后执行 bash start_server.sh
# ==============================================================================

export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

BACKEND=vllm
CKPT_PATH="/path/to/TEACHER_MODEL/"

wait_server_ready() {
    server=$1
    ip=$2
    port=$3
    while true; do
        echo "wait $server server ready at $ip:$port..."
        result=`echo -e "\n" | telnet $ip $port 2> /dev/null | grep Connected | wc -l`
        if [ $result -eq 1 ]; then
            break
        else
            sleep 1
        fi
    done
}

ps -ef | grep "python proxy.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9
ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9

nohup python proxy.py &> proxy.log &

wait_server_ready proxy localhost $PROXY_BACKEND_PORT

echo "teacher proxy is ready"

nohup python worker.py --backend $BACKEND --tp-size 1 --n-logprobs 256 --ckpt-path $CKPT_PATH &> worker.log &
echo "start teacher worker"

echo "teacher server is ready"