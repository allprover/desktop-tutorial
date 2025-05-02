---
HLS 流媒体协议
typora-copy-images-to: img
---





# 一、工作原理

1. **媒体分割**：将音视频内容分割成一系列小的TS (Transport Stream)文件片段
2. **索引文件**：生成M3U8格式的播放列表文件，包含媒体片段的URL和元数据
3. **HTTP传输**：通过标准HTTP服务器分发这些文件
4. **客户端播放**：播放器按顺序下载并播放这些片段



# 二、文件结构

- **主播放列表(Master Playlist)**：`.m3u8`文件，包含所有可用码率的流
- **媒体播放列表(Media Playlist)**：每个码率对应的播放列表
- **媒体片段**：`.ts`文件(MPEG-2 Transport Stream)或`.mp4`片段



# 三、优缺点

**优点**：

- 使用标准HTTP端口，易于部署
- 适应不同网络条件
- 支持直播和点播
- 广泛的设备支持

**缺点**：

- 延迟较高(通常10-30秒)
- 相比RTMP等协议效率较低
- 需要将内容预先切片





# 四、FFmpeg 生成 m3u8 切片

```bash
ffmpeg -i input.mp4 -c:v libx264 -c:a copy -f hls -hls_time 10 -hls_list_size 0  input/index.m3u8

      -hls_time n: 设置每片的长度，默认值为2,单位为秒
      -hls_list_size n:设置播放列表保存的最多条目，设置为0会保存有所片信息，默认值为5
      -hls_wrap n:设置多少片之后开始覆盖，如果设置为0则不会覆盖，默认值为0
          这个选项能够避免在磁盘上存储过多的片，而且能够限制写入磁盘的最多的片的数量
```

![image-20250501155856123](C:\Users\13227\Documents\GitHub\desktop-tutorial\音视频\img\image-20250501155856123.png)





# 五、HLSSever

## 1. 主函数

**网络初始化**

```c
// Windows 网络初始化
WSAStartup(MAKEWORD(2, 2), &wsaData); 

// 创建TCP套接字
SOCKET serverFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

// 绑定到所有网卡的8080端口
server_addr.sin_addr.S_un.S_addr = htonl(INADDR_ANY); 
server_addr.sin_port = htons(8080);
bind(serverFd, (SOCKADDR*)&server_addr, sizeof(SOCKADDR));

// 开始监听
listen(serverFd, SOMAXCONN);  // SOMAXCONN是系统允许的最大队列长度
```

**连接处理逻辑**

```c
while (true) {
    // 接受新连接
    SOCKADDR_IN accept_addr;
    int clientFd = accept(serverFd, (SOCKADDR*)&accept_addr, &len);

    // 创建连接处理器（当前是单线程阻塞模式）
    Connection conn(clientFd);
    conn.start();
}
```



## 2. Connection

```cpp
int Connection::start() {
    // 1. 接收HTTP请求数据
    char bufRecv[2000];
    recv(mClientFd, bufRecv, 2000, 0);

    // 2. 解析URI (如 /index.m3u8)
    char uri[100];
    while (line = strtok(bufRecv, "\n")) {
        if (strstr(line, "GET")) {
            sscanf(line, "GET %s HTTP/1.1\r\n", &uri);
        }
    }

    // 3. 拼接本地文件路径
    std::string filename = "../data/input" + std::string(uri);

    // 4. 读取文件内容到缓冲区
    FILE* fp = fopen(filename.data(), "rb");
    int bufLen = fread(buf, 1, sizeof(buf), fp);

    // 5. 构造HTTP响应头
    char http_headers[2000];
    if (strcmp("/index.m3u8", uri) == 0) {
        // M3U8文件特殊头
        sprintf(http_headers, "HTTP/1.1 200 OK\r\nContent-Type: application/vnd.apple.mpegurl...");
    } else {
        // TS文件通用头
        sprintf(http_headers, "HTTP/1.1 200 OK\r\nContent-Type: video/mp2t...");
    }

    // 6. 发送响应
    send(mClientFd, http_headers, strlen(http_headers), 0);
    send(mClientFd, buf, bufLen, 0);

    return 0;
}
```



# 六、FFplay 播放

我们运行hlsSever服务器后，运行以下指令。

```
ffplay -i http://127.0.0.1:8080/index.m3u8
```



# 七、抓包分析

![image-20250501161548040](C:\Users\13227\Documents\GitHub\desktop-tutorial\音视频\img\image-20250501161548040.png)

ffplay发送http请求：

```
Frame 5: 188 bytes on wire (1504 bits), 188 bytes captured (1504 bits) on interface \Device\NPF_Loopback, id 0
Null/Loopback
Internet Protocol Version 4, Src: 127.0.0.1, Dst: 127.0.0.1
Transmission Control Protocol, Src Port: 61511, Dst Port: 8080, Seq: 1, Ack: 1, Len: 144
Hypertext Transfer Protocol
    GET /index.m3u8 HTTP/1.1\r\n
    User-Agent: Lavf/58.29.100\r\n
    Accept: */*\r\n
    Range: bytes=0-\r\n
    Connection: close\r\n
    Host: 127.0.0.1:8080\r\n
    Icy-MetaData: 1\r\n
    \r\n
    [Response in frame: 11]
    [Full request URI: http://127.0.0.1:8080/index.m3u8]
```

而后服务器收到后，输出一些信息：

```
Frame 11: 454 bytes on wire (3632 bits), 454 bytes captured (3632 bits) on interface \Device\NPF_Loopback, id 0
Null/Loopback
Internet Protocol Version 4, Src: 127.0.0.1, Dst: 127.0.0.1
Transmission Control Protocol, Src Port: 8080, Dst Port: 61511, Seq: 1283, Ack: 145, Len: 410
[4 Reassembled TCP Segments (1692 bytes): #7(210), #9(536), #10(536), #11(410)]
Hypertext Transfer Protocol
    HTTP/1.1 200 OK\r\n
    Access-Control-Allow-Origin: * \r\n
    Connection: keep-alive\r\n
    Content-Length: 1482\r\n
    Content-Type: application/vnd.apple.mpegurl; charset=utf-8\r\n
    Keep-Alive: timeout=30, max=100\r\n
    Server: hlsServer\r\n
    \r\n
    [Request in frame: 5]
    [Time since request: 0.000998000 seconds]
    [Request URI: /index.m3u8]
    [Full request URI: http://127.0.0.1:8080/index.m3u8]
    File Data: 1482 bytes
Media Type
```

Connection 的代码中：

```c
if (0 == strcmp("/index.m3u8", uri)) {
    sprintf(http_headers, "HTTP/1.1 200 OK\r\n"
        "Access-Control-Allow-Origin: * \r\n"
        "Connection: keep-alive\r\n"
        "Content-Length: %d\r\n"
        "Content-Type: application/vnd.apple.mpegurl; charset=utf-8\r\n"
        "Keep-Alive: timeout=30, max=100\r\n"
        "Server: hlsServer\r\n"
        "\r\n",
        bufLen);
}
```

特殊报头 m3u8，两者正好是匹配的。

然后ffplay继续持续请求 index*.ts：

![image-20250501163827462](C:\Users\13227\Documents\GitHub\desktop-tutorial\音视频\img\image-20250501163827462.png)

```
Hypertext Transfer Protocol
    HTTP/1.1 200 OK\r\n
    Access-Control-Allow-Origin: * \r\n
    Connection: close\r\n
    Content-Length: 410216\r\n
    Content-Type: video/mp2t; charset=utf-8\r\n
    Keep-Alive: timeout=30, max=100\r\n
    Server: hlsServer\r\n
    \r\n
    [Request in frame: 16]
    [Time since request: 0.046967000 seconds]
    [Request URI: /index0.ts]
    [Full request URI: http://127.0.0.1:8080/index0.ts]
    File Data: 410216 bytes
```

其中 File Data 就是 h264 视频数据了。

```c
else {
    sprintf(http_headers, "HTTP/1.1 200 OK\r\n"
        "Access-Control-Allow-Origin: * \r\n"
        "Connection: close\r\n"
        "Content-Length: %d\r\n"
        "Content-Type: video/mp2t; charset=utf-8\r\n"
        "Keep-Alive: timeout=30, max=100\r\n"
        "Server: hlsServer\r\n"
        "\r\n",
        bufLen);
}
```

