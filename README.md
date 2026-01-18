# EA Trading Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing Expert Advisor (EA) trading performance with interactive charts and PDF report generation.

## 🚀 Features

- **Interactive Data Upload**: Support for CSV and Excel files
- **Multi-tab Analysis**: 
  - Overview with key metrics and insights
  - Crossover analysis with session performance
  - Impulse analysis with threshold tracking
  - Raw data tables with filtering
- **Session Heatmaps**: Performance analysis across trading sessions (SYDNEY, TOKYO, LONDON, NY)
- **Segment Analysis**: Distance and impulse distribution charts
- **Threshold Tracking**: Success rates for reversal thresholds (20%, 30%, 40%, 50%)
- **PDF Report Generation**: Professional reports with insights and recommendations
- **Interactive Filtering**: Date range, direction, session, and segment filters
- **Data Export**: Download filtered data as CSV

## 📋 Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## 🛠️ Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Dashboard

1. Open terminal/command prompt in the project directory
2. Run the Streamlit app:
```bash
streamlit run dashboard.py
```
3. Open your web browser and go to `http://localhost:8501`

## 📁 Data Format

### Crossover Stats CSV
Expected columns:
- StartTime, EndTime, Direction, StartPrice, EndPrice, MaxMinPrice, Distance, MAValue, Segment, Session_Start, Session_Peak, Session_End, Symbol, TF, MAPeriod, MAType

### Impulse Reversal CSV  
Expected columns:
- Time, Direction, EntryPrice, Peak, TriggerPrice, Impulse, Pullback, Segment, Hit_20%, Hit_30%, Hit_40%, Hit_50%, Session_Entry, Session_Peak, Session_Trigger, Symbol, TF, MAPeriod, MAType

## 📊 Dashboard Sections

### 1. Overview Tab
- Key performance metrics
- AI-generated insights
- Session performance heatmaps
- Direction bias analysis

### 2. Crossover Analysis Tab
- Segment distribution charts
- Direction performance comparison
- Time-based analysis
- Performance statistics by segment

### 3. Impulse Analysis Tab
- Threshold success rate analysis
- Impulse vs Pullback scatter plots
- Threshold performance tables
- Success rate metrics

### 4. Data Tables Tab
- Interactive data tables with filtering
- Export functionality
- Real-time filtering options

## 📄 PDF Reports

Generate professional PDF reports with:
- Executive summary with key metrics
- AI-generated insights and recommendations
- Performance charts and analysis
- Data tables and statistics
- Trading recommendations

### Report Types:
- **Summary**: Quick overview (2-3 pages)
- **Detailed**: Comprehensive analysis (10+ pages)  
- **Executive**: Management summary (5-6 pages)

## 🎯 Key Metrics Tracked

### Crossover Analysis:
- Total trades and average distance
- Direction bias (Bullish vs Bearish)
- Session performance
- Segment distribution
- Duration analysis

### Impulse Analysis:
- Threshold hit rates
- Success rate percentages
- Impulse strength analysis
- Pullback percentages
- Session-based performance

## 🔧 Customization

The dashboard is modular and can be extended:
- `utils.py`: Data processing and chart generation
- `pdf_generator.py`: PDF report creation
- `dashboard.py`: Main Streamlit interface

## 📈 Sample Usage

1. Upload your EA-generated CSV files using the sidebar
2. Explore different analysis tabs
3. Apply filters to focus on specific periods or conditions
4. Generate PDF reports for offline analysis
5. Download filtered data for further analysis

## 🛡️ Error Handling

The dashboard includes comprehensive error handling for:
- File format validation
- Data structure verification
- Chart generation errors
- PDF generation issues

## 📞 Support

For issues or questions:
1. Check the data format requirements
2. Ensure all required columns are present
3. Verify file formats (CSV/Excel)
4. Check the console for detailed error messages

## 🔄 Updates

The dashboard automatically processes new data uploads and refreshes all charts and metrics in real-time.