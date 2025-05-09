# 一、简介

## 1. 特点

- **低延迟**：通常在1-3秒之间
- **基于TCP**：默认使用1935端口
- **支持多种格式**：FLV、MP4、F4V等
- **灵活的消息分块**：支持消息分块传输
- **多种控制命令**：支持连接、创建流、发布、播放等控制命令





## 2. 架构

### 2.1 组成

```
+---------------+    +----------------+    +---------------+
| 视频采集设备 | -> | 编码器/转码器 | -> | RTMP服务器    |
+---------------+    +----------------+    +---------------+
                                 ↓
                         +---------------+
                         | 客户端播放器  |
                         +---------------+
```



### 2.2 模块

1. **连接管理模块**：处理客户端连接/断开
2. **协议解析模块**：解析RTMP协议数据
3. **流媒体管理模块**：管理直播流和订阅关系
4. **媒体分发模块**：向订阅者分发媒体数据
5. **录制模块**：可选录制功能



## 3. 核心功能

### 3.1 握手过程

1. **C0+C1**：客户端发送协议版本和随机数据
2. **S0+S1+S2**：服务器回应协议版本和随机数据
3. **C2**：客户端验证服务器响应



### 3.2 连接控制

- **Connect**：建立与应用程序的连接
- **Call**：调用远程方法
- **Create Stream**：创建逻辑流通道
- **Delete Stream**：删除流



### 3.3 媒体控制

- **Publish**：发布媒体流
- **Play**：播放媒体流
- **Pause**：暂停播放
- **Seek**：跳转到指定位置



### 3.4 数据封装

RTMP将媒体数据封装为消息(Message)，消息又分为多个块(Chunk)传输：

- **音频消息**：包含音频数据
- **视频消息**：包含视频数据
- **数据消息**：包含元数据或脚本数据



# 二、C++ 实现 RTMP

## 1. 主函数

### 1.1 初始化端口

RTMP 默认端口：1935

```c
int main(int argc, char **argv)
{
    int port = 1935;  // RTMP默认端口
    printf("rtmpServer rtmp://127.0.0.1:%d\n", port);
```



### 1.2 事件循环初始化

```
xop::EventLoop eventLoop;
```

- 创建事件循环对象，用于处理网络I/O事件



### 1.3 RTMP 服务器创建与配置

```c++
auto rtmp_server = xop::RtmpServer::Create(&eventLoop);
rtmp_server->SetChunkSize(60000);
//rtmp_server->SetGopCache(); // enable gop cache
```

- 创建RTMP服务器实例，关联到事件循环
- 设置块大小(Chunk Size)为60000字节
- 注释掉的`SetGopCache`表示可以启用GOP缓存（关键帧缓存）



### 1.4 事件回调

```c++
rtmp_server->SetEventCallback([](std::string type, std::string stream_path) {
    printf("[Event] %s, stream path: %s\n\n", type.c_str(), stream_path.c_str());
});
```

- 设置Lambda表达式作为事件回调
- 当服务器事件发生时（如新连接、新流等），打印事件类型和流路径



### 1.5 启动服务器

```c++
if (!rtmp_server->Start("0.0.0.0", port)) {
    printf("start rtmpServer error\n");
}
```

- 启动服务器，监听所有网络接口(0.0.0.0)的1935端口
- 如果启动失败，打印错误信息



### 1.6 主循环

```c++
while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
```

- 简单的保持程序运行的循环
- 每100毫秒休眠一次，避免空转消耗CPU



### 1.7 资源清理

```c++
rtmp_server->Stop();
return 0;
```



## 2. 事件循环

### 2.1 构造函数和析构函数

```c++
EventLoop::EventLoop(uint32_t num_threads) : index_(1)
{
    num_threads_ = 1;
    if (num_threads > 0) {
        num_threads_ = num_threads;
    }
    this->Loop();
}

EventLoop::~EventLoop()
{
    this->Quit();
}
```

- 构造函数接收线程数参数，默认创建1个线程
- 析构函数自动停止事件循环并清理资源



### 2.2 GetTaskScheduler

```cpp
std::shared_ptr<TaskScheduler> EventLoop::GetTaskScheduler()
{
    std::lock_guard<std::mutex> locker(mutex_);
    if (task_schedulers_.size() == 1) {
        return task_schedulers_.at(0);
    }
    else {
        auto task_scheduler = task_schedulers_.at(index_);
        index_++;
        if (index_ >= task_schedulers_.size()) {
            index_ = 1;
        }        
        return task_scheduler;
    }
    return nullptr;
}
```

- 采用轮询(round-robin)方式分配任务调度器
- 单线程情况下直接返回唯一的调度器
- 多线程时循环分配，实现简单的负载均衡



#### 任务调度器(TaskScheduler)

用于处理I/O事件、定时任务和跨线程触发事件。

typedef std::function<void(void)> TriggerEvent;

**构造函数与初始化**

```cpp
TaskScheduler::TaskScheduler(int id)
    : id_(id)
    , is_shutdown_(false) 
    , wakeup_pipe_(new Pipe())
    , trigger_events_(new xop::RingBuffer<TriggerEvent>(kMaxTriggetEvents))
{
    // Windows网络初始化(只执行一次)
    static std::once_flag flag;
    std::call_once(flag, [] {
        #if defined(WIN32) || defined(_WIN32) 
            WSADATA wsa_data;
            if (WSAStartup(MAKEWORD(2, 2), &wsa_data)) {
                WSACleanup();
            }
        #endif
    });

    // 创建唤醒管道并设置回调
    if (wakeup_pipe_->Create()) {
        wakeup_channel_.reset(new Channel(wakeup_pipe_->Read()));
        wakeup_channel_->EnableReading();
        wakeup_channel_->SetReadCallback([this]() { this->Wake(); });        
    }        
}
```

- **初始化列表**：
  - `id_`：调度器标识
  - `is_shutdown_`：关闭标志
  - `wakeup_pipe_`：用于线程间唤醒的管道
  - `trigger_events_`：环形缓冲区存储触发事件
- **Windows网络初始化**：
  - 使用`std::call_once`确保只初始化一次
  - 调用`WSAStartup`初始化Winsock
- **唤醒管道**：
  - 创建管道用于线程间通信
  - 设置管道读端的Channel和回调函数



**Start()**

```c++
void TaskScheduler::Start()
{
    // Linux信号忽略
    #if defined(__linux) || defined(__linux__) 
        signal(SIGPIPE, SIG_IGN); // 忽略管道破裂信号
        signal(SIGQUIT, SIG_IGN);
        signal(SIGUSR1, SIG_IGN);
        signal(SIGTERM, SIG_IGN);
        signal(SIGKILL, SIG_IGN);
    #endif     
    
    is_shutdown_ = false;
    while (!is_shutdown_) {
        this->HandleTriggerEvent();    // 处理触发事件
        this->timer_queue_.HandleTimerEvent(); // 处理定时器
        int64_t timeout = this->timer_queue_.GetTimeRemaining();
        this->HandleEvent((int)timeout); // 处理I/O事件(带超时)
    }
}
```

1. 处理跨线程触发事件
2. 处理到期的定时器
3. 计算下次定时器触发剩余时间作为I/O多路复用超时
4. 处理I/O事件



**Stop()**

```c++
void TaskScheduler::Stop()
{
    is_shutdown_ = true;
    char event = kTriggetEvent;
    wakeup_pipe_->Write(&event, 1); // 写入唤醒事件
}
```

- 设置关闭标志
- 通过管道写入数据唤醒可能阻塞在I/O多路复用中的线程



**定时器管理**

```cpp
TimerId TaskScheduler::AddTimer(TimerEvent timerEvent, uint32_t msec)
{
    return timer_queue_.AddTimer(timerEvent, msec);
}

void TaskScheduler::RemoveTimer(TimerId timerId)
{
    timer_queue_.RemoveTimer(timerId);
}
```

- 委托给内部的`timer_queue_`实现
- 提供添加和删除定时器接口



**触发事件处理**

```cpp
bool TaskScheduler::AddTriggerEvent(TriggerEvent callback)
{
    if (trigger_events_->Size() < kMaxTriggetEvents) {
        std::lock_guard<std::mutex> lock(mutex_);
        char event = kTriggetEvent;
        trigger_events_->Push(std::move(callback));
        wakeup_pipe_->Write(&event, 1); // 写入管道唤醒线程
        return true;
    }
    return false;
}

void TaskScheduler::HandleTriggerEvent()
{
    do {
        TriggerEvent callback;
        if (trigger_events_->Pop(callback)) {
            callback(); // 执行回调
        }
    } while (trigger_events_->Size() > 0);
}

void TaskScheduler::Wake()
{
    char event[10] = { 0 };
    while (wakeup_pipe_->Read(event, 10) > 0); // 清空管道数据
}
```

move



### 2.3 Loop

```cpp
void EventLoop::Loop()
{
    std::lock_guard<std::mutex> locker(mutex_);
    if (!task_schedulers_.empty()) {
        return;
    }

    // 创建任务调度器
    for (uint32_t n = 0; n < num_threads_; n++) {
        #if defined(__linux) || defined(__linux__) 
            std::shared_ptr<TaskScheduler> task_scheduler_ptr(new EpollTaskScheduler(n));
        #elif defined(WIN32) || defined(_WIN32) 
            std::shared_ptr<TaskScheduler> task_scheduler_ptr(new SelectTaskScheduler(n));
        #endif
        task_schedulers_.push_back(task_scheduler_ptr);
        std::shared_ptr<std::thread> thread(new std::thread(&TaskScheduler::Start, task_scheduler_ptr.get()));
        thread->native_handle();
        threads_.push_back(thread);
    }

    // 设置线程优先级
    const int priority = TASK_SCHEDULER_PRIORITY_REALTIME;
    for (auto iter : threads_) {
        #if defined(WIN32) || defined(_WIN32) 
            // Windows下设置线程优先级
        #endif
    }
}
```

- 平台相关的调度器选择：Linux用epoll，Windows用select
- 创建指定数量的工作线程
- 设置线程优先级（Windows平台）



#### Epoll

**功能**

- 通过 Linux 的 `epoll` 机制监听多个文件描述符（如 Socket）的 I/O 事件，触发对应的回调处理。
- 支持动态添加、修改、删除监听通道（`Channel`）。

**核心方法**

构造方法，`epoll` 实例的文件描述符，通过 `epoll_create(1024)` 初始化。

```c
#include <sys/epoll.h>

EpollTaskScheduler::EpollTaskScheduler(int id)
	: TaskScheduler(id)
{
#if defined(__linux) || defined(__linux__) 
    epollfd_ = epoll_create(1024);
 #endif
    this->UpdateChannel(wakeup_channel_);
}
```

UpdateChannel，添加或更新通道的事件监听（`EPOLL_CTL_ADD/MOD`）

```cpp
void EpollTaskScheduler::UpdateChannel(ChannelPtr channel)
{
	std::lock_guard<std::mutex> lock(mutex_);
#if defined(__linux) || defined(__linux__) 
	int fd = channel->GetSocket();
	if(channels_.find(fd) != channels_.end()) {
		if(channel->IsNoneEvent()) {
			Update(EPOLL_CTL_DEL, channel);
			channels_.erase(fd);
		}
		else {
			Update(EPOLL_CTL_MOD, channel);
		}
	}
	else {
		if(!channel->IsNoneEvent()) {
			channels_.emplace(fd, channel);
			Update(EPOLL_CTL_ADD, channel);
		}	
	}	
#endif
}
```

RemoveChannel，移除通道的监听（`EPOLL_CTL_DEL`）

```cpp
void EpollTaskScheduler::RemoveChannel(ChannelPtr& channel)
{
    std::lock_guard<std::mutex> lock(mutex_);
#if defined(__linux) || defined(__linux__) 
	int fd = channel->GetSocket();

	if(channels_.find(fd) != channels_.end()) {
		Update(EPOLL_CTL_DEL, channel);
		channels_.erase(fd);
	}
#endif
}
```

HandleEvent，阻塞等待事件触发（`epoll_wait`），并调用对应的 `Channel->HandleEvent`

```cpp
bool EpollTaskScheduler::HandleEvent(int timeout)
{
#if defined(__linux) || defined(__linux__) 
	struct epoll_event events[512] = {0};
	int num_events = -1;

	num_events = epoll_wait(epollfd_, events, 512, timeout);
	if(num_events < 0)  {
		if(errno != EINTR) {
			return false;
		}								
	}

	for(int n=0; n<num_events; n++) {
		if(events[n].data.ptr) {        
			((Channel *)events[n].data.ptr)->HandleEvent(events[n].events);
		}
	}		
	return true;
#else
    return false;
#endif
}
```

epoll 操作封装

```cpp
void Update(int operation, ChannelPtr& channel) {
    struct epoll_event event = {0};
    if (operation != EPOLL_CTL_DEL) {
        event.data.ptr = channel.get();  // 保存 Channel 指针
        event.events = channel->GetEvents();  // 监听的事件（如 EPOLLIN/EPOLLOUT）
    }
    epoll_ctl(epollfd_, operation, channel->GetSocket(), &event);
}
```



#### Select



### 2.4 Quit

```cpp
void EventLoop::Quit()
{
    std::lock_guard<std::mutex> locker(mutex_);
    for (auto iter : task_schedulers_) {
        iter->Stop();
    }
    for (auto iter : threads_) {
        iter->join();
    }
    task_schedulers_.clear();
    threads_.clear();
}
```

- 停止所有任务调度器
- 等待所有线程结束
- 清理容器资源



### 2.5 管道管理

```cpp
void EventLoop::UpdateChannel(ChannelPtr channel)
void EventLoop::RemoveChannel(ChannelPtr& channel)
```

- 提供通道的更新和删除接口
- 所有操作都路由到第一个任务调度器（主调度器）



#### 2.6 定时器管理

```
TimerId EventLoop::AddTimer(TimerEvent timerEvent, uint32_t msec)
void EventLoop::RemoveTimer(TimerId timerId)
```

- 添加和移除定时器
- 定时事件也由主调度器处理



### 2.7 触发事件

```c
bool EventLoop::AddTriggerEvent(TriggerEvent callback)
```

- 添加触发事件回调
- 用于跨线程事件通知