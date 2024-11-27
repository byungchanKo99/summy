# summy
A program that summarizes mentoring videos and writes reports

# How to Start

1. Create and activate virtual environment
   ```
   python3.12 -m venv venv
   source venv/bin/activate
   ```
   Install virtual environment with Python 3.12 version due to pydub library compatibility issues

2. Install dependencies
   ```
   pip install moviepy openai pydub inquirer
   ```
   Note: ffmpeg must be installed.
   ffmpeg homebrew - https://formulae.brew.sh/formula/ffmpeg

3. Insert OpenAI API key
   ```
   client = OpenAI(api_key="insert your key here")
   ```

4. Run Python file
   ```
   python summy.py
   ```
   Then locate your audio file and press enter
