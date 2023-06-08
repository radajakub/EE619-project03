#!/bin/bash

sshpass -p MFoRL2645 scp -r 20236065@eelabg13.kaist.ac.kr:EE619-project03/MINE/experiments ~/Projects/EE619-project03/outputs/newest_13
sshpass -p MFoRL2645 scp -r 20236065@eelabg13.kaist.ac.kr:EE619-project03/MINE/ee619/training.out ~/Projects/EE619-project03/outputs/newest_13/training.out
sshpass -p MFoRL2645 scp -r 20236065@eelabg13.kaist.ac.kr:EE619-project03/MINE/ee619/trained_model.pt ~/Projects/EE619-project03/outputs/newest_13/trained_model.pt
sshpass -p MFoRL2645 scp -r 20236065@eelabg13.kaist.ac.kr:EE619-project03/MINE/ee619/trained_model_intermediate.pt ~/Projects/EE619-project03/outputs/newest_13/trained_model_intermediate.pt
