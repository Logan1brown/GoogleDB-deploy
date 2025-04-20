#!/bin/bash
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
python -m pip install -r requirements.txt

echo "Starting Streamlit..."
streamlit run src/dashboard/app.py
