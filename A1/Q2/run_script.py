#!/usr/bin/env python
"""
Wrapper script to run Q2_denoising_autoencoder.py with proper encoding
"""
import sys
import os

# Set encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Change to script directory
os.chdir(r'd:\Sem8\GenAI\Assignments\A1\Q2')

# Execute the main script
exec(open('Q2_denoising_autoencoder.py').read())
