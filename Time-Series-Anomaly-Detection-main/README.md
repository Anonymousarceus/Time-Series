# Smart System Monitor

A tool that watches your data and tells you when something unusual happens.

## What Does This Do?

This program looks at your data over time and finds when things go wrong. It's like having a smart assistant that watches all your numbers and alerts you when something doesn't look normal.

**Think of it like this**: If you track your car's temperature, speed, and fuel usage every minute, this tool would notice if the temperature suddenly spikes or if fuel usage becomes weird compared to normal patterns.

## Key Features

- **Finds Problems Automatically**: No need to manually check everything
- **Tells You What's Wrong**: Points out which specific things are causing issues
- **Easy to Use**: Just give it your data file and get results
- **Smart Learning**: Studies normal patterns first, then spots the odd stuff

## Getting Started

### Step 1: Install the Program
```bash
pip install -r requirements.txt
```

### Step 2: Test It Out
```bash
# Create some test data first
python utils.py

# Then run the detector
python main.py TEP_Train_Test.csv my_results.csv
```

### Step 3: Look at Your Results
```bash
python -c "from utils import analyze_results; analyze_results('my_results.csv')"
```

## How to Use It

### Simple Way
Put your data in a CSV file and run:
```bash
python main.py your_data.csv results.csv
```

### What You Get Back
The program adds these columns to your data:

- **Abnormality_score**: A number from 0 to 100
  - 0-10: Everything looks normal ✅
  - 11-30: A little unusual 🤔
  - 31-60: Something's definitely off ⚠️
  - 61-90: This is a problem! 🚨
  - 91-100: Major issue! 🔥

- **top_feature_1** to **top_feature_7**: The main things causing each problem

## What Kind of Data Do You Need?

- **Format**: Excel or CSV file with numbers
- **Columns**: Multiple measurements (like temperature, pressure, speed, etc.)
- **Rows**: Each row should be one time point (like every minute or hour)
- **Size**: At least 72 rows of data

## How It Works

### The Smart Part
The program uses four different "brains" to spot problems:

1. **Pattern Hunter**: Finds data points that don't fit with the rest
2. **Smart Copier**: Learns to copy normal patterns, fails on weird ones  
3. **Relationship Tracker**: Notices when things that usually go together stop doing so
4. **Team Decision**: Combines all three approaches for the best results

### What Problems Can It Find?

- **Numbers Too High/Low**: When values go outside normal ranges
- **Broken Relationships**: When things that usually change together stop doing so
- **Weird Patterns**: When the sequence of events looks unusual

## File Structure

```
Your Project Folder/
├── main.py                 # The main program
├── anomaly_detector.py     # The smart detection part
├── data_preprocessor.py    # Cleans up your data
├── feature_attribution.py # Figures out what caused problems
├── utils.py               # Helper functions
└── requirements.txt       # List of needed software
```

## Common Problems & Solutions

**"Not enough data" error**
- Make sure your file has at least 72 rows

**"Everything looks normal" (all scores near 0)**
- Your data might be too clean or not have any real problems
- Try it on data where you know something went wrong

**Program runs very slowly**
- Large files take time - try with a smaller sample first

**Weird results**
- Make sure your data has numbers, not text
- Check that each column represents something meaningful

## Tips for Best Results

1. **Use Good Data**: Make sure your measurements are accurate
2. **Include Context**: More related measurements = better detection
3. **Know Your Normal**: The program learns from the first part of your data, so make sure that part represents normal operations
4. **Check Results**: The program is smart, but you know your system best

## Example Use Cases

- **Factory Equipment**: Monitor machines for early warning signs
- **Website Performance**: Track response times and error rates
- **Financial Data**: Spot unusual trading patterns
- **IoT Sensors**: Monitor building temperature, humidity, etc.
- **Vehicle Fleet**: Track fuel usage, engine performance

## Need Help?

1. Make sure your data file is in the same folder as the program
2. Check that all your numbers make sense (no weird text mixed in)
3. Try the test data first to make sure everything works
4. Start with smaller files before processing huge datasets

## What Makes This Special?

- **No Manual Setup**: Just point it at your data
- **Explains Itself**: Tells you not just "something's wrong" but "what's wrong"
- **Learns Your System**: Adapts to your specific normal patterns
- **Production Ready**: Fast enough for real-world use

---

*Built for finding needles in haystacks, so you can fix problems before they become disasters.*