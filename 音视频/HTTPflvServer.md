# 一、简介

HTTP-FLV (HTTP-based Flash Video) 是一种基于HTTP协议的流媒体传输协议，主要用于实时视频直播场景。



## 1. 概念

HTTP-FLV是一种将FLV(Flash Video)格式的视频通过HTTP协议传输的技术。它结合了FLV容器格式和HTTP协议的优势：

- 使用HTTP协议，可以穿透大多数防火墙
- 采用FLV格式，兼容性好
- 支持流式传输，延迟较低(通常2-3秒)

比 HLS 延迟低，比 RTMP 延迟高。



## 2. 工作原理

### 2.1 工作流程

1. 客户端发起HTTP请求
2. 服务器返回HTTP响应头(Connection: keep-alive)
3. 服务器持续发送FLV格式的视频数据
4. 客户端持续接收并播放



### 2.2 FLV格式

FLV文件由Header和Body组成：

- **Header**(9字节)：签名(FLV) + 版本 + 类型标志 + 数据偏移

- 

  Body

  ：由一系列Tag组成，每个Tag包含：

  - 前一个Tag的大小(4字节)
  - Tag类型(1字节：音频/视频/脚本)
  - 数据大小(3字节)
  - 时间戳(3字节+扩展1字节)
  - 流ID(3字节)
  - 实际数据



## 3. 服务器架构

### 3.1 架构组成

```
+---------------+    +----------------+    +---------------+
| 视频采集/转码 | -> | 流媒体服务器 | -> | HTTP-FLV服务器 |
+---------------+    +----------------+    +---------------+
       ↓                                     ↓
+---------------+                    +---------------+
| 视频源(摄像头)|                    | 客户端播放器 |
+---------------+                    +---------------+
```



### 3.2 核心模块设计

1. **连接管理模块**：处理客户端HTTP连接
2. **流媒体分发模块**：管理直播流和订阅关系
3. **缓存模块**：缓存最近的媒体数据
4. **协议转换模块**：将RTMP等协议转换为HTTP-FLV



## 4. HTTP-FLV 服务器实现

### 4.1 Nginx

Nginx可以通过`nginx-rtmp-module`模块支持HTTP-FLV：

```nginx
rtmp {
    server {
        listen 1935;
        application live {
            live on;
            meta copy;
        }
    }
}

http {
    server {
        listen 80;
        location /live {
            flv_live on;
            chunked_transfer_encoding on;
            add_header 'Access-Control-Allow-Origin' '*';
            add_header 'Access-Control-Allow-Credentials' 'true';
        }
    }
}
```

### 4.2 Node.js

```js
const http = require('http');
const fs = require('fs');

// FLV Header (9 bytes)
const flvHeader = Buffer.from([
    0x46, 0x4C, 0x56, 0x01, 0x05, 0x00, 0x00, 0x00, 0x09
]);

const server = http.createServer((req, res) => {
    // 检查请求路径
    if (req.url === '/live.flv') {
        // 设置响应头
        res.writeHead(200, {
            'Content-Type': 'video/x-flv',
            'Connection': 'keep-alive',
            'Transfer-Encoding': 'chunked'
        });
        
        // 发送FLV Header
        res.write(flvHeader);
        
        // 模拟发送视频数据
        const timer = setInterval(() => {
            const tag = generateFLVTag();
            res.write(tag);
        }, 30);
        
        // 客户端断开连接时清理
        req.on('close', () => {
            clearInterval(timer);
        });
    }
});

function generateFLVTag() {
    // 这里简化了FLV Tag生成逻辑
    const tag = Buffer.alloc(11 + 5); // 11字节头 + 5字节数据
    // 填充Tag数据...
    return tag;
}

server.listen(8000);
```



# 二、简易实现 HTTP-FLV

## 1. 主函数结构

```c
int main() {
    // 服务器配置
    int port = 8080;
    const char* filename = "../data/test.flv";
    LOGI("httpflvServer http://127.0.0.1:%d/test.flv", port);
```

## 2. Win socket 初始化

```c
WSADATA wsaData;
if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    LOGE("WSAStartup error");
    return -1;
}
```

## 3. 创建和配置服务器 socket

```c
SOCKET serverFd;
SOCKADDR_IN server_addr;
server_addr.sin_family = AF_INET;
server_addr.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
server_addr.sin_port = htons(port);
serverFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
```

## 4. 绑定和监听

```c
if (bind(serverFd, (SOCKADDR*)&server_addr, sizeof(SOCKADDR)) == SOCKET_ERROR) {
    LOGE("socket bind error");
    return -1;
}
if (listen(serverFd, SOMAXCONN) < 0) {
    LOGE("socket listen error");
    return -1;
}
```

## 5. 准备 HTTP 相应头

```c
constexpr char http_headers[] = \
    "HTTP/1.1 200 OK\r\n" \
    "Access-Control-Allow-Origin: * \r\n" \
    "Content-Type: video/x-flv\r\n" \
    "Content-Length: -1\r\n" \
    "Connection: Keep-Alive\r\n" \
    "Expires: -1\r\n" \
    "Pragma: no-cache\r\n" \
    "\r\n";
int http_headers_len = strlen(http_headers);
```

## 6. 处理客户端连接

```c
while (true) {
    LOGI("等待新连接...");
    int len = sizeof(SOCKADDR);
    SOCKADDR_IN accept_addr;
    int clientFd = accept(serverFd, (SOCKADDR*)&accept_addr, &len);
```

## 7. 处理客户端请求

```c
if (clientFd == SOCKET_ERROR) {
    LOGE("accept connection error");
    break;
}
LOGI("发现新连接 clientFd=%d", clientFd);
unsigned char buf[5000];
char bufRecv[2000] = { 0 };
```

## 8. 打开 FLV 文件并发送数据

```c
FILE* fp;
fp = fopen(filename, "rb");
if (!fp) {
    LOGE("fopen %s fail!", filename);
}
else {
    int times = 0;
    while (true) {
        times++;
        
        if (times == 1) {
            // 第一次循环：接收客户端请求并发送HTTP头
            int bufRecvSize = recv(clientFd, bufRecv, 2000, 0);
            LOGI("bufRecvSize=%d,bufRecv=%s", bufRecvSize, bufRecv);
            send(clientFd, http_headers, http_headers_len, 0);
        }
        else {
            // 后续循环：发送FLV文件内容
            Sleep(10);
            int bufLen = fread(buf, 1, sizeof(buf), fp);
            int ret = send(clientFd, (char*)buf, bufLen, 0);
            
            if (ret <= 0) {
                break;
            }
        }
    }
}
```



# 三、抓包分析

```
Hypertext Transfer Protocol
    GET /test.flv HTTP/1.1\r\n
    User-Agent: Lavf/58.29.100\r\n
    Accept: */*\r\n
    Range: bytes=0-\r\n
    Connection: close\r\n
    Host: 127.0.0.1:8080\r\n
    Icy-MetaData: 1\r\n
    \r\n
    [Full request URI: http://127.0.0.1:8080/test.flv]

```

