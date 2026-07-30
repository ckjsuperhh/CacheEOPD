# ==============================================================================
# GKD 教师模型多节点 Worker 加入脚本
# ==============================================================================
# 用途：在多节点部署时，从节点通过此脚本加入教师模型推理集群
# 关键参数：
#   - PROXY_IP: 主节点（proxy 所在节点）的 IP 地址
#   - PROXY_BACKEND_PORT: 主节点 proxy 的后端端口
#   - CKPT_PATH: 教师模型检查点路径
#   - --tp-size 8: 此 worker 的张量并行大小
# 使用方法：修改 PROXY_IP 和 CKPT_PATH 后在从节点执行
# ==============================================================================

export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

PROXY_IP="127.0.0.1"
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

# pkill -f "python proxy.py"
# pkill -f "python worker.py"
ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9

wait_server_ready proxy $PROXY_IP $PROXY_BACKEND_PORT

echo "teacher proxy is ready"

nohup python worker.py --backend $BACKEND --proxy-addr $PROXY_IP:$PROXY_BACKEND_PORT --tp-size 8 --n-logprobs 256 --ckpt-path $CKPT_PATH &> worker.log &

echo "teacher server is ready"
