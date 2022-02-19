#!/usr/bin/bash
cd data/
for i in 1 2 3 4 5 6 7 8 9 10
do 
    mkdir week$i
    wget www2.imm.dtu.dk/courses/02506/data/week$i.zip --directory-prefix week$i
    unzip week$i/week$i.zip -d week$i
    rm week$i/week$i.zip
done