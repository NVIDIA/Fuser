#!/bin/bash

timeit() {
    echo $@
    /usr/bin/time --format="Time: %e s\nMemory: %M KB" $@
    echo ""
}

timeall() {
    $1 --version
    timeit $1 main.cpp -DTYPE=int -I../src/ -std=c++17 -O3 -o main0.exe
    timeit $1 main.cpp -DTYPE="DynamicType<NoContainers,int>" -I../src/ -std=c++17 -O3 -o main1.exe
    timeit $1 main.cpp -DTYPE="DynamicType<NoContainers,int,double>" -I../src/ -std=c++17 -O3 -o main2.exe
    timeit $1 main.cpp -DTYPE="DynamicType<NoContainers,int,double,int*>" -I../src/ -std=c++17 -O3 -o main3.exe
    timeit $1 main.cpp -DTYPE="DynamicType<NoContainers,int,double,int*,float*>" -I../src/ -std=c++17 -O3 -o main4.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector>,int,double,int*,float*>" -I../src/ -std=c++17 -O3 -o main5.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list>,int,double,int*,float*>" -I../src/ -std=c++17 -O3 -o main6.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*>" -I../src/ -std=c++17 -O3 -o main7.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*>" -I../src/ -std=c++17 -O3 -o main8.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*,std::string>" -I../src/ -std=c++17 -O3 -o main9.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*,std::string,std::string*>" -I../src/ -std=c++17 -O3 -o main10.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*,std::string,std::string*,std::string**>" -I../src/ -std=c++17 -O3 -o main11.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*,std::string,std::string*,std::string**,std::string***>" -I../src/ -std=c++17 -O3 -o main12.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*,std::string,std::string*,std::string**,std::string***,std::string****>" -I../src/ -std=c++17 -O3 -o main13.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*,std::string,std::string*,std::string**,std::string***,std::string****,std::string*****>" -I../src/ -std=c++17 -O3 -o main14.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*,std::string,std::string*,std::string**,std::string***,std::string****,std::string*****,std::string******>" -I../src/ -std=c++17 -O3 -o main15.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*,std::string,std::string*,std::string**,std::string***,std::string****,std::string*****,std::string******,std::string*******>" -I../src/ -std=c++17 -O3 -o main16.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*,std::string,std::string*,std::string**,std::string***,std::string****,std::string*****,std::string******,std::string*******,std::string********>" -I../src/ -std=c++17 -O3 -o main17.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*,std::string,std::string*,std::string**,std::string***,std::string****,std::string*****,std::string******,std::string*******,std::string********,std::string*********>" -I../src/ -std=c++17 -O3 -o main18.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*,std::string,std::string*,std::string**,std::string***,std::string****,std::string*****,std::string******,std::string*******,std::string********,std::string*********,std::string**********>" -I../src/ -std=c++17 -O3 -o main19.exe
    timeit $1 main.cpp -DTYPE="DynamicType<Containers<std::vector,std::list,std::deque>,int,double,int*,float*,double*,std::string,std::string*,std::string**,std::string***,std::string****,std::string*****,std::string******,std::string*******,std::string********,std::string*********,std::string**********,std::string***********>" -I../src/ -std=c++17 -O3 -o main20.exe
    rm -rf *.exe
}

echo "======================= g++ ======================="
timeall g++

echo ""

echo "===================== clang++ ====================="
timeall clang++
