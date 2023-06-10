#!/bin/bash

server_num=$1

if (( $server_num > 11  )) && (( $server_num < 15 ))
then
    sshpass -p MFoRL2645 scp -r 20236065@eelabg${server_num}.kaist.ac.kr:EE619-project03/SAC/experiments ~/Projects/EE619-project03/outputs/newest_${server_num}
    sshpass -p MFoRL2645 scp -r 20236065@eelabg${server_num}.kaist.ac.kr:EE619-project03/SAC/ee619/training.out ~/Projects/EE619-project03/outputs/newest_${server_num}/training.out
    sshpass -p MFoRL2645 scp -r 20236065@eelabg${server_num}.kaist.ac.kr:EE619-project03/SAC/ee619/trained_model.pt ~/Projects/EE619-project03/outputs/newest_${server_num}/trained_model.pt
    sshpass -p MFoRL2645 scp -r 20236065@eelabg${server_num}.kaist.ac.kr:EE619-project03/SAC/ee619/traind_model_best.pt ~/Projects/EE619-project03/outputs/newest_${server_num}/trained_model_best.pt
fi
