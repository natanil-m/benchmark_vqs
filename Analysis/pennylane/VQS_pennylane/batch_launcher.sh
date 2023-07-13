#!/bin/bash

for ((i=5; i<=30; i++))
do
 sh launch_pro.sh "$i" "$1"
done
