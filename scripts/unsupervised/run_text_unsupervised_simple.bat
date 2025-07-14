@echo off
echo Running Simple Text-based Unsupervised Anomaly Detection...

python scripts/train_text_unsupervised_simple.py --data_dir data --output_dir results_text_unsupervised
 
echo.
echo Text-based unsupervised training complete!
pause 