#!/bin/bash
python -V
pip install -r requirements.txt
streamlit run src/dashboard/app.py
