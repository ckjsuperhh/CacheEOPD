#!/bin/bash
# 分离工作者（Detached Worker）测试运行脚本
# 启动 Ray 集群，依次运行服务端和客户端，然后强制停止 Ray
ray start --head --port=6379
python3 server.py
python3 client.py
ray stop --force