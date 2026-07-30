<!--
中文摘要：分离工作者说明文档。
描述 Ray 分离工作者的客户端-服务端架构和运行方式。
-->
# Detached Worker
## How to run (Only on a single node)
- Start a local ray cluster: 
```bash
ray start --head --port=6379
```
- Run the server
```bash
python3 server.py
```
- On another terminal, Run the client
```bash
python3 client.py
```
