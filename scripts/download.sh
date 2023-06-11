#!/bin/bash

server_num=$1

if (( $server_num > 11  )) && (( $server_num < 15 ))
then
    server_prefix=20236065@eelabg${server_num}.kaist.ac.kr:EE619-project03/SAC
    local_prefix=./outputs/newest_${server_num}


    sshpass -p MFoRL2645 scp -r ${server_prefix}/experiments ${local_prefix}/
    sshpass -p MFoRL2645 scp -r ${server_prefix}/ee619/training.out ${local_prefix}/training.out

    mkdir -p ${local_prefix}/intermediate_models/
    sshpass -p MFoRL2645 scp -r ${server_prefix}/ee619/trained_model_\*.pt ${local_prefix}/intermediate_models
    sshpass -p MFoRL2645 scp -r ${server_prefix}/ee619/trained_model.pt ${local_prefix}/trained_model.pt
fi
