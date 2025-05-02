-------

​		I/O多路复用它通过一种机制，可以监视多个文件描述符，一旦某个文件描述符可以执行 I/O 操作时，能够通知应用程序进行相应的读写操作。

​		I/O 多路复用技术是为了解决：在并发式 I/O 场景中进程或线程阻塞到某个 I/O 系统调用而出现的技术，使进程不阻塞于某个特定的 I/O 系统调用。 

-----

# 一、select()

调用 select()会一直阻塞，直到某一个或多个文件描述符成为就绪态（可以读或写）。

函数原型：

```c
#include <sys/select.h> 
 
int select( int nfds, 		    //最大描述符编号+1
            fd_set *readfds,    //读是否就绪
            fd_set *writefds, 	//写是否就绪
            fd_set *exceptfds,  //异常是否发生
            struct timeval *timeout);	//阻塞的时间上限
```

fd_set 数据类型是一个文件描述符的集合体。

若把三个 fd_set 类型的参数都设置为 NULL，则类似于 sleep() 。



**文件描述符宏操作**

```c
#include <sys/select.h> 
 
void FD_CLR(int fd, fd_set *set); 			//将文件描述符 fd 从参数 set 所指向的集合中移除；
int FD_ISSET(int fd, fd_set *set); 			//判断文件描述符 fd 是参数 set 所指向的集合中的成员
void FD_SET(int fd, fd_set *set); 			//文件描述符 fd 添加到参数 set 所指向的集合中；
void FD_ZERO(fd_set *set);					//将参数 set 所指向的集合初始化为空； 
```

文件描述符集合有一个最大容量限制，有常量 FD_SETSIZE 来决定，在 Linux 系统下，该常量的值为 1024

在定义一个文件描述符集合之后，必须用 FD_ZERO()宏将其进行初始化操作。



**select() 函数返回值**

- 返回 -1 表示有错误发生
- 返回 0 表示 select() 调用超时，readfds， writefds 以及 exceptfds 所指向的文件描述符集合都会被清空。
- 返回一个正整数表示有一个或多个文件描述符以达到就绪态。



**示例**

​		使用 select()函数来实现 I/O 多路复用操作，同时读取键盘和鼠标。对数据进行了 5 次读取，select()函数的参数 timeout 被设置为 NULL，只关心鼠标或键盘是否有数据可读， 所以将参数 writefds 和 exceptfds 也设置为 NULL。







> 文件描述符 0
>
> - 键盘默认绑定到标准输入（`/dev/stdin`），文件描述符为 `0`。
> - 当用户在终端输入时，数据通过 `read()` 系统调用从文件描述符 `0` 读取。



# 二、poll()

系统调用 poll()与 select()函数很相似，但函数接口有所不同。在 select()函数中，我们提供三个 fd_set 集 合，在每个集合中添加我们关心的文件描述符；而在 poll()函数中，则需要构造一个 struct pollfd 类型的数 组，每个数组元素指定一个文件描述符以及我们对该文件描述符所关心的条件（数据可读、可写或异常情 况）。poll()函数原型如下所示：

```c
#include <poll.h> 
 
int poll(	struct pollfd *fds, 	//指向一个 struct pollfd 类型的数组，数组中的每个元素都会指定									   //一个文件描述符以及我们对该文件描述符所关心的条件
			nfds_t nfds, 			//参数 nfds 指定了 fds 数组中的元素个数
			int timeout);			//如果 timeout 等于-1，则 poll()会一直阻塞
									//如果 timeout 等于 0，poll()不会阻塞
									//如果 timeout 大于 0，则表示设置 poll()函数阻塞时间的上限值
```



**poll函数返回值**

poll()函数返回值含义与 select()函数的返回值是一样的。



**示例**

同样是键鼠多路复用操作。





# 三、epoll()

**基本特性**

仅仅只限于 Linux 平台

- **事件驱动**：仅返回就绪的 fd，无需遍历所有监控的 fd（O(1) 时间复杂度）。
- **内存映射**：使用 `mmap` 减少内核和用户空间的数据拷贝。
- 支持边缘触发（ET）和水平触发（LT）：
  - **LT（默认）**：只要 fd 可读/可写，就会持续通知。
  - **ET**：仅在状态变化时通知一次（需一次性处理完数据，否则可能丢失事件）。



**函数原型**

```c
#include <sys/epoll.h>

int epoll_create(int size);  // 创建 epoll 实例（返回 epfd）
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);  // 增删改监控事件
int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout);  // 等待事件
```

`epoll_event`结构体

```c
struct epoll_event {
    uint32_t events;  // EPOLLIN、EPOLLOUT、EPOLLET（边缘触发）等
    void *data;       // 用户数据（通常存储 fd）
};
```

**示例**

```c
int epfd = epoll_create(1);  // 创建 epoll 实例

struct epoll_event ev;
ev.events = EPOLLIN;  // 监控可读事件
ev.data.fd = sockfd;

epoll_ctl(epfd, EPOLL_CTL_ADD, sockfd, &ev);  // 添加 sockfd 到 epoll

struct epoll_event ready_events[10];
int ret = epoll_wait(epfd, ready_events, 10, 1000);  // 等待 1 秒
for (int i = 0; i < ret; i++) {
    if (ready_events[i].events & EPOLLIN) {
        // ready_events[i].data.fd 可读
    }
}
```



# 四、异步IO

在 I/O 多路复用中，进程通过系统调用 select()或 poll()来主动查询文件描述符上是否可以执行 I/O 操作。

而在异步 I/O 中，当文件描述符上可以执行 I/O 操作时，进程可以请求内核为自己发送一个信号。

**步骤**

- 通过指定 O_NONBLOCK 标志使能非阻塞 I/O。
- 通过指定 O_ASYNC 标志使能异步 I/O。
- 设置异步 I/O 事件的接收进程。
- 为内核发送的通知信号注册一个信号处理函数。
- 以上步骤完成之后，进程就可以执行其它任务了，当 I/O 操作就绪时，内核会向进程发送一个 SIGIO 信号，当进程接收到信号时，会执行预先注册好的信号处理函数，我们就可以在信号处理函数中进 行 I/O 操作。 



# 

