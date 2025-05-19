---
opencv集成gstreamer交叉编译环境
---



#### 目录结构

```
~/opencv-4.9.0/
  ├── CMakeLists.txt
  ├── build_rk3588/      # 你当前所在的空目录
  └── rk3588_toolchain.cmake
```

#### 工具链文件

```cmake
# 示例内容
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 必须使用绝对路径！
set(CMAKE_C_COMPILER "/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/bin/aarch64-buildroot-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/bin/aarch64-buildroot-linux-gnu-g++")

# Sysroot路径
set(CMAKE_SYSROOT "/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot")
```

### 检查sysroot中的Gstreamer

```
# 在主机上检查sysroot中的GStreamer
ls /home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot/usr/lib/libgstreamer*
ls /home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot/usr/include/gstreamer-1.0
```

### 开启Gstreamer支持

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=../rk3588_toolchain.cmake \
-DWITH_GSTREAMER=ON \
-DGSTREAMER_INCLUDE_DIR=/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot/usr/include/gstreamer-1.0 \
-DGSTREAMER_LIBRARY=/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot/usr/lib/libgstreamer-1.0.so \
-DGSTREAMER_BASE_INCLUDE_DIR=/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot/usr/include/gstreamer-1.0 \
-DGSTREAMER_BASE_LIBRARY=/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot/usr/lib/libgstreamer-base-1.0.so \
-DGSTREAMER_APP_INCLUDE_DIR=/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot/usr/include/gstreamer-1.0 \
-DGSTREAMER_APP_LIBRARY=/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot/usr/lib/libgstreamer-app-1.0.so \
-DGSTREAMER_PLUGINS_BASE_INCLUDE_DIR=/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot/usr/include/gstreamer-1.0 \
-DGSTREAMER_PLUGINS_BASE_LIBRARY=/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot/usr/lib/libgstreamer-plugins-base-1.0.so \
-DWITH_GSTREAMER_0_10=OFF \
-DWITH_FFMPEG=OFF \
..
```

### **验证pkg-config配置**

```
ls /home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot/usr/lib/pkgconfig/gstreamer-1.0.pc
```

### **设置交叉编译的pkg-config环境**

```bash
export PKG_CONFIG_PATH=/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot/usr/lib/pkgconfig
export PKG_CONFIG_SYSROOT_DIR=/home/elf/aarch64-buildroot-linux-gnu_sdk-buildroot/aarch64-buildroot-linux-gnu/sysroot
export PKG_CONFIG_LIBDIR=${PKG_CONFIG_PATH}
```

