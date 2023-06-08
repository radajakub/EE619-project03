#!/bin/bash

sshpass -p MFoRL2645 scp -r 20236065@eelabg12.kaist.ac.kr:EE619-project03/MINE/experiments ~/Projects/EE619-project03/outputs/newest_12
sshpass -p MFoRL2645 scp -r 20236065@eelabg12.kaist.ac.kr:EE619-project03/MINE/ee619/training.log ~/Projects/EE619-project03/outputs/newest_12/training.log
sshpass -p MFoRL2645 scp -r 20236065@eelabg12.kaist.ac.kr:EE619-project03/MINE/ee619/trained_model.pt ~/Projects/EE619-project03/outputs/newest_12/trained_model.log
sshpass -p MFoRL2645 scp -r 20236065@eelabg12.kaist.ac.kr:EE619-project03/MINE/ee619/trained_model_intermediate.pt ~/Projects/EE619-project03/outputs/newest_12/trained_model_intermediate.pt
