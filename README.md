# Directory Structure

- `create_data`: Python scripts for creating data
- `data`: Generated raw data
- `pipelines`: Python scripts for ZenML Pipelines
- `steps`: Python scripts for ZenML Steps
- `configs`: Python scripts for ZenML Configs
- `run.py`: Script for running ZenML Pipelines
- `models`: Generated ZenML Models
- `test`: Test scripts
- `docs`: Some documentation



# Operation steps

1. Make sure the Python version is 3.12, then install dependencies

   ```
   pip install -r requirements.txt
   ```

2. Generate training data. The generated data is saved in the `data` directory

   ```
   cd create_data
   python create_cpu_timestamp_data.py
   ```

3. Run the training model script

   ```
   python run.py
   ```

   
