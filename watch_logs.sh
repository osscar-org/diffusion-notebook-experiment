#!/bin/bash
ssh -t dokku@materialscloud.io enter osscar-diffusion-experiment web tail -f notebook/diffusion_module.log 

