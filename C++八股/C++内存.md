---

由全局变量和局部变量，引出内存管理等话题

---



# 一、定义位置与作用域

## 全局变量

定义位置：在所有函数外部定义，例如文件开头或函数之间

作用域：从定义位置开始到整个源文件结束有效，可被同一文件内的所有函数访问。若需跨文件使用，需通过 extern 声明。

```cpp
int globalVar = 10;  // 全局变量
void func() {
    printf("%d", globalVar);  // 可访问
}
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)



## 局部变量

定义位置：在函数内部或复合语句（如{}代码块）内定义，包括形式参数。

作用域：仅限于定义它的函数或代码块内部，离开后无法访问。

```cpp
void func() {
    int localVar = 20;  // 局部变量
    printf("%d", localVar);  // 仅在func()内有效
}
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)



# 二、生命周期与存储位置

## 全局变量

生命周期：从程序启动时初始化，到程序结束时释放，贯穿整个运行周期。

存储位置：位于静态存储区（数据段），内存由编译器静态分配。

初始化：若未显式初始化，默认值为0（整型）或NULL（指针）。



## 局部变量

生命周期：仅在函数调用时创建，函数返回后销毁。

存储位置：位于栈区，内存动态分配且随函数调用自动回收。

初始化：未初始化时值为随机（取决于栈内存残留值），需手动赋值。

> 其实学过C逆向的，很容易理解。函数调用时，需要把各种寄存器入栈，而这些寄存器是决定了栈上的变量参数的。调用返回后，要把寄存器出栈。



# 三、使用特性

## 重名冲突处理

若局部变量与全局变量同名，函数内部优先操作局部变量，全局变量被暂时“隐藏”。

```cpp
int x = 10;  // 全局变量
void func() {
    int x = 5;  // 局部变量
    printf("%d", x);  // 输出5，而非全局的10
}
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

> 因为函数是调用栈的，优先从栈区获取变量值。



## 代码维护性影响

全局变量：便于跨函数共享数据，但滥用可能导致耦合性高、调试困难。
 局部变量：封装性好，函数独立性高，但需通过参数传递数据。


## 内存管理差异

全局变量占用固定内存，可能增加程序体积。

局部变量动态释放，内存利用率更高。



## 使用建议

 全局变量：慎用于需跨模块共享数据的场景，建议通过static限制作用域。

局部变量：优先使用以减少副作用，通过参数和返回值传递数据。



# **其他**

![img](https://i-blog.csdnimg.cn/direct/450cd36f5ad043e1ad6ac6c0acd13c61.png)![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)编辑

全局变量，位于静态存储器（数据段）中，局部变量则位于栈区。



​    初始化为0的全局变量通常会被分配到程序的BSS（Block Started by Symbol）段。BSS段是用于存放未初始化或初始化为0的全局变量和静态变量的一部分内存空间。在程序加载时，系统会自动将BSS段中的变量初始化为0。

​    已经明确初始化为非零值的全局变量会被分配到程序的数据（Data）段。Data段用于存放已经初始化的全局变量和静态变量。

​    总结来说，初始化为0的全局变量通常会被分配到BSS段，而已初始化为非零值的全局变量则会被分配到Data段。



# 四、值传递和指针传递

由于上面讲局部变量的时候，提到了函数。那么扩展值传递和指针传递的区别。其实就是栈区操作的区别了。

## 值传递

主调函数将实参的值复制到栈中（如push eax，假设eax存储参数值）。

被调函数通过[ebp+8]、[ebp+12]等偏移量访问参数的副本。

因为操作的是栈上的副本，修改形参是不影响实参的。

但是由于是在栈上，那么传递的值应为小型数据。



## 指针传递

主调函数将实参的地址压入栈中（如push &a）。

被调函数通过[ebp+8]获取地址，再通过解引用（如mov eax, [ebp+8]）操作实际内存。

因为是获取地址后，对该地址进行操作，操作的就是实际的内存了。

会导致被调函数是能够直接修改主调函数中的变量的。



当然还有引用，其实和指针传递差不多。



# 五、指针函数和函数指针

因为上面扯到了指针和函数，那么就提一下让人头疼的这部分。

## **指针函数**

定义：本质是一个函数，其返回值类型为指针（如int *func()）。
 作用：用于返回动态分配内存的地址、静态变量地址或通过参数传递的有效地址。

**栈区操作：**

若函数内部定义局部变量并返回其地址（如int *a = ...; return a;），

该地址指向栈帧中的局部变量。但由于函数返回后栈帧会被销毁，此时返回的地址成为野指针，访问会导致未定义行为。

安全做法是返回静态区（static变量）、全局区或堆区（malloc分配）的地址。

```cpp
int *unsafeFunc() {
    int x = 10;     // 栈区局部变量
    return &x;      // 危险：返回栈地址
}
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

```cpp
int* addIntegers(int a, int b) {
  int* result = (int*)malloc(sizeof(int));
  *result = a + b;
  return result;
}
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

后者的话，用malloc分配就安全了。 

## 函数指针

定义：本质是一个指针变量，指向函数的入口地址（如int (*ptr)(int, int)）。

作用：通过指针间接调用函数，常用于回调函数、动态绑定等场景。

**栈区操作：**

函数指针变量（如int (*ptr)(int)）存储在栈区，其值为目标函数的入口地址（代码区地址）。栈帧结构与普通函数调用一致。

```cpp
int add(int a, int b) { return a + b; }
int main() {
    int (*ptr)(int, int) = add;  // 函数指针存储于栈区
    ptr(3, 4);                   // 参数压栈后跳转执行
}
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

# 六、内存泄漏

1） 堆内存泄漏：通过malloc/new等内存分配，但忘了free或delete

2） 系统资源泄露：主要指程序使用系统分配的资源（Bitmap、handle、SOCKET）但没有使用相应的函数释放掉

3） 没有将基类的析构函数定义为虚函数



