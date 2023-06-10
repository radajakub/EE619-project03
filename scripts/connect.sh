#!/bin/bash

server_num=$1

if (( $server_num > 11  )) && (( $server_num < 15 ))
then
    sshpass -p MFoRL2645 ssh 20236065@eelabg${server_num}.kaist.ac.kr
fi
