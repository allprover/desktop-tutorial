---
高级文件I/O
---



# 一、非阻塞和阻塞I/O

阻塞其实就是进入了休眠状态，交出了 CPU 控制权。譬如 wait()、pause()、sleep()等函数都会进入阻塞。

对于某些文件类型（读管道文件、网络设备文件和字符设备文件），当对文件进行读操作时，如果数据未准备好、文件当前无数据可读，那么读操作可能会使调用者阻塞，直到有数据可读时才会被唤醒，这就是阻塞式 I/O 常见的一种表现；

如果是非阻塞式 I/O，即使没有数据可读，也不会被阻塞、而是会立马返回错误！

普通文件的读写操作是不会阻塞的，不管读写多少个字节数据，read()或 write()一定会在有限的时间内返回，所以普通文件一定是以非阻塞的方式进行 I/O 操作，这是普通文件本质上决定的。



## 1. I/O读文件

参数 flags 指定 O_NONBLOCK 标志，会使得后续的 I/O 操作以非阻塞进行，相反则默认使用阻塞方式。



## 2. 阻塞I/O的优缺点



## 3. 使用非阻塞I/O实现并发读取



# 二、I/O多路复用

I/O多路复用，可以监视多个文件描述符。当某个文件可以执行 I/O 操作时，能够通知应用程序进行相应的读写操作。

I/O 多路复用技术是为了解决：在并发式 I/O 场景中进程或线程阻塞到某个 I/O 系统调用而出现的技术，使进程不阻塞于某个特定的I/O 系统调用。

由此可知，I/O 多路复用一般用于并发式的非阻塞 I/O，也就是多路非阻塞 I/O，譬如程序中既要读取鼠标、又要读取键盘，多路读取。

## 1. select() 函数

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

#include<fcntl.h>
#include<unistd.h>
#include<sys/select.h>

#define MOUSE "/dev/input/event3"

int main(void){
    char buf[100];
    int fd, ret = 0, flag;
    fd_set rdfds;
    int loops = 5;

    /* 打开鼠标设备文件 */
    fd = open(MOUSE, O_RDONLY | O_NONBLOCK);
    if(-1 == fd){
        perror("open error");
        exit(-1);
    }

    /* 将键盘设置为非阻塞方式*/
    flag = fcntl(0, F_GETFL);   //先获取原来的 flag
    flag |= O_NONBLOCK;         //将 O_NONBLOCK 标准添加到 flag
    fcntl(0, F_SETFL, flag);    //重新设置 flag

    /* 同时读取键盘和鼠标*/
    while(loops--){
        FD_ZERO(&rdfds);
        FD_SET(0, &rdfds);      //键盘属于标准输入，0就是其文件标识符
        FD_SET(fd, &rdfds);     //添加鼠标

        ret = select(fd + 1, &rdfds, NULL, NULL, NULL);

        //两种非正常的返回
        if(0>ret){
            perror("select error");
            goto out;
        }
        else if (0 == ret)
        {
            fprintf(stderr, "select timeout.\n");
            continue;
        }

        /* 检查键盘是否为就绪态*/
        if(FD_ISSET(0, &rdfds)){
            ret = read(0, buf, sizeof(buf));
            if(0<ret)
                printf("键盘： 成功读取<%d>个字节数据\n", ret);
        }

        /* 检查鼠标是否为就绪态 */
        if(FD_ISSET(fd, &rdfds)){
            ret = read(fd, buf, sizeof(buf));
            if(0<ret)
                printf("鼠标: 成功读取<%d>个字节数据\n", ret);
        }
        
    }

out: 
    close(fd);
    exit(ret);
} 


```



## 2. poll() 函数

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <fcntl.h>
#include <unistd.h>
#include <poll.h>

#define MOUSE "/dev/input/event3"

int main(void){
    char buf[100];
    int fd, ret = 0, flag;
    int loops = 5; 
    struct pollfd fds[2];
    
    /* 打开鼠标设备文件*/
    fd = open(MOUSE, O_RDONLY | O_NONBLOCK);
    if(-1 == fd){
        perror("open error");
        exit(-1);
    }
    /* 将键盘设置为非阻塞*/
    flag = fcntl(0, F_GETFL);
    flag |= O_NONBLOCK;
    fcntl(0, F_SETFL, flag);

    /* 同时读取键盘和鼠标 */
    fds[0].fd = 0;
    fds[0].events = POLLIN;     //只关心数据可读
    fds[0].revents = 0;

    fds[1].fd = fd;
    fds[1].events = POLLIN;
    fds[1].revents = 0;

    while(loops--){
        ret = poll(fds, 2, -1);
        if(0>ret){
            perror("poll error");
            goto out;
        }
        else if (0 == ret){
            fprintf(stderr, "poll timeout.\n");
            continue;
        }

        /* 检查键盘是否就绪*/
        if(fds[0].revents & POLLIN){
            ret = read(0, buf, sizeof(buf));
            if(0 < ret){
                printf("键盘: 成功读取<%d>个字节数据\n", ret);
            }
        }

        /* 检查鼠标是否就绪*/
        if(fds[1].revents & POLLIN){
            ret = read(fd, buf, sizeof(buf));
            if(0 < ret){
                printf("鼠标: 成功读取<%d>个字节数据\n", ret);
            }
        }
    }
out:
    close(fd);
    exit(ret);
}
```



# 三、异步I/O

```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <signal.h>

#define MOUSE "/dev/input/event3"

static int fd;

static void sigio_handler(int sig)
{
    static int loops = 5;
    char buf[100] = {0};
    int ret;

    if(SIGIO != sig)
        return;
    ret = read(fd, buf, sizeof(buf));
    if(0<ret)
        printf("鼠标: 成功读取<%d>个字节数据\n", ret);

    loops--;
    if(0>=loops){
        close(fd);
        exit(0);
    }

}

int main(void){
    int flag;

    fd = open(MOUSE, O_RDONLY | O_NONBLOCK);
    if(-1 == fd){
        perror("open error");
        exit(-1);
    }

    /* 使能异步 IO*/
    flag = fcntl(fd, F_GETFL);
    flag |= O_ASYNC;
    fcntl(fd, F_SETFL, flag);

    /* 设置异步IO的所有者*/
    fcntl(fd, __F_SETOWN, getpid());

    /* 为SIGIO信号注册信号处理函数*/
    signal(SIGIO, sigio_handler);

    for(;;)
        sleep(1);
}
```

