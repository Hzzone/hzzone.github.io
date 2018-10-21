安装依赖:

```shell
sudo apt-get install -y build-essential cmake git pkg-config

sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler

sudo apt-get install -y libatlas-base-dev

sudo apt-get install -y --no-install-recommends libboost-all-dev

sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev

sudo apt-get install -y python-pip

sudo apt-get install -y python-dev

sudo apt-get install -y python-numpy python-scipy

sudo apt-get install -y libopencv-dev
```

下载 [cudnn](https://developer.nvidia.com/rdp/cudnn-download)，选择:

![](https://tuchuang-1252747889.cosgz.myqcloud.com/2018-10-21-085609.png)

下载 [CUDA 9.0](https://developer.nvidia.com/cuda-90-download-archive)，选择:

![](https://tuchuang-1252747889.cosgz.myqcloud.com/2018-10-21-085351.png)


安装 CUDA 9.0:

```shell
sudo sh cuda_9.0.176_384.81_linux.run
```
![](https://hzzone.io/images/563A2032-2DD5-46C3-AE6D-49BF4C068FE3.png)

显卡驱动可以在官网下载最新版本的进行安装，老版本在 Ubuntu 上我经常遇到问题，在登录界面一直重复，就是显卡驱动的问题。

然后添加环境变量，在 `zshrc` 或 `bashrc` 中添加:

```shell
export PATH=/usr/local/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
```

然后:

```shell
source ~/.zshrc
```

然后安装 cudnn:

```shell
# 解压
tar -xzvf cudnn-9.0-linux-x64-v7.3.1.20.tgz
cp cuda/include/cudnn.h /usr/local/cuda-9.0/include/
cp cuda/lib64/lib* /usr/local/cuda-9.0/lib64/
```

完成之后，如果没有安装驱动，从官网下载安装，需要注意的提前关闭 Xserver:

```shell
sudo service lightdm stop
```

安装完成之后再启动:

```shell
sudo service lightdm start
```

最后查看 CUDA 版本:

```shell
➜  ~ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Sep__1_21:08:03_CDT_2017
Cuda compilation tools, release 9.0, V9.0.176
```

查看设备信息:

```shell
➜  samples ls
0_Simple     2_Graphics  4_Finance      6_Advanced       common    Makefile
1_Utilities  3_Imaging   5_Simulations  7_CUDALibraries  EULA.txt
➜  samples pwd
/usr/local/cuda-9.0/samples
➜  samples sudo make -j16
....
```

```shell
➜  deviceQuery ./deviceQuery
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GTX 1060 6GB"
  CUDA Driver Version / Runtime Version          9.1 / 9.0
  CUDA Capability Major/Minor version number:    6.1
```
