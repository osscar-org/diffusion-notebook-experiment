#!/bin/bash
ssh dokku@materialscloud.io enter osscar-diffusion-experiment web cat notebook/diffusion_module.log > prod-dokku-diffusion_module.log
echo "run the following command now:"
echo "  python logging-analysis/parse_diffusion_logs.py prod-dokku-diffusion_module.log"

