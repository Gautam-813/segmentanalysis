import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
from utils import DataProcessor, ChartGenerator, ReportGenerator, format_number
from pdf_generator import PDFReportGenerator

# Page configuration
st.set_page_config(
    page_title="EA Trading Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'chart_generator' not in st.session_state:
    st.session_state.chart_generator = ChartGenerator(st.session_state.data_processor)
if 'report_generator' not in st.session_state:
    st.session_state.report_generator = ReportGenerator(st.session_state.data_processor, st.session_state.chart_generator)

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 EA Trading Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 File Upload")
        
        # File upload section
        crossover_file = st.file_uploader(
            "Upload Crossover Stats CSV/Excel",
            type=['csv', 'xlsx'],
            key="crossover_upload"
        )
        
        impulse_file = st.file_uploader(
            "Upload Impulse Reversal CSV/Excel", 
            type=['csv', 'xlsx'],
            key="impulse_upload"
        )
        
        # Load data
        crossover_loaded = False
        impulse_loaded = False
        
        if crossover_file is not None:
            try:
                with st.spinner("Loading crossover data..."):
                    st.session_state.data_processor.load_crossover_data(crossover_file)
                    crossover_loaded = True
                st.success("✅ Crossover data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading crossover data: {str(e)}")
        
        if impulse_file is not None:
            try:
                with st.spinner("Loading impulse data..."):
                    st.session_state.data_processor.load_impulse_data(impulse_file)
                    impulse_loaded = True
                st.success("✅ Impulse data loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading impulse data: {str(e)}")
        
        # Filters section
        if crossover_loaded or impulse_loaded:
            st.header("🔍 Filters")
            
            # Date range filter
            if crossover_loaded:
                df = st.session_state.data_processor.crossover_data
                min_date = df['StartTime'].min().date()
                max_date = df['StartTime'].max().date()
                
                date_range = st.date_input(
                    "Select Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            
            # Direction filter
            direction_filter = st.multiselect(
                "Select Direction",
                options=['BULLISH', 'BEARISH'],
                default=['BULLISH', 'BEARISH']
            )
            
            # Session filter
            session_filter = st.multiselect(
                "Select Sessions",
                options=['SYDNEY', 'TOKYO', 'LONDON', 'NY'],
                default=['SYDNEY', 'TOKYO', 'LONDON', 'NY']
            )
        
        # PDF Report Generation
        if crossover_loaded or impulse_loaded:
            st.header("📄 Generate Report")
            
            report_type = st.selectbox(
                "Report Type",
                options=['summary', 'detailed', 'executive'],
                format_func=lambda x: x.title()
            )
            
            if st.button("🔄 Generate PDF Report", type="primary"):
                try:
                    with st.spinner("Generating PDF report..."):
                        pdf_gen = PDFReportGenerator(
                            st.session_state.data_processor,
                            st.session_state.chart_generator,
                            st.session_state.report_generator
                        )
                        pdf_data = pdf_gen.generate_pdf_report(report_type)
                        
                        # Create download button
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_data,
                            file_name=f"EA_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                        st.success("✅ PDF report generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
    
    # Main content area
    if not crossover_loaded and not impulse_loaded:
        # Welcome screen
        st.markdown("""
        ## Welcome to EA Trading Analysis Dashboard! 👋
        
        This dashboard helps you analyze your Expert Advisor (EA) trading performance with interactive charts and comprehensive reports.
        
        ### 🚀 Getting Started:
        1. **Upload your data files** using the sidebar
        2. **Explore interactive charts** and analysis
        3. **Apply filters** to focus on specific periods or conditions
        4. **Generate PDF reports** for offline analysis
        
        ### 📊 Features:
        - **Session Performance Analysis** - See how your EA performs across different trading sessions
        - **Segment Distribution** - Understand distance and impulse patterns
        - **Threshold Analysis** - Track reversal threshold success rates
        - **Direction Performance** - Compare bullish vs bearish trade performance
        - **Time-based Analysis** - Identify optimal trading hours
        - **PDF Reports** - Generate professional reports with insights and recommendations
        
        ### 📁 Supported File Types:
        - CSV files (.csv)
        - Excel files (.xlsx)
        
        **Upload your files to get started!**
        """)
        
        # Sample data info
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **Crossover Stats CSV should contain:**
            - StartTime, EndTime, Direction, StartPrice, EndPrice, MaxMinPrice, Distance, MAValue, Segment, Session_Start, Session_Peak, Session_End, Symbol, TF, MAPeriod, MAType
            
            **Impulse Reversal CSV should contain:**
            - Time, Direction, EntryPrice, Peak, TriggerPrice, Impulse, Pullback, Segment, Hit_20%, Hit_30%, Hit_40%, Hit_50%, Session_Entry, Session_Peak, Session_Trigger, Symbol, TF, MAPeriod, MAType
            """)
    
    else:
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🎯 Crossover Analysis", "⚡ Impulse Analysis", "🌍 Session Analysis", "📋 Data Tables"])
        
        with tab1:
            show_overview()
        
        with tab2:
            if crossover_loaded:
                show_crossover_analysis()
            else:
                st.info("Please upload crossover data to view this analysis.")
        
        with tab3:
            if impulse_loaded:
                show_impulse_analysis()
            else:
                st.info("Please upload impulse data to view this analysis.")
        
        with tab4:
            show_session_analysis(crossover_loaded, impulse_loaded)
        
        with tab5:
            show_data_tables()

def show_overview():
    """Show overview dashboard"""
    st.header("📊 Performance Overview")
    
    # Generate summary stats
    stats = st.session_state.report_generator.generate_summary_stats()
    
    if stats:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if 'crossover' in stats:
            cs = stats['crossover']
            with col1:
                st.metric("Total Trades", f"{cs['total_trades']:,}")
            with col2:
                st.metric("Avg Distance", f"{cs['avg_distance']:.2f}")
            with col3:
                st.metric("Bullish Trades", f"{cs['bullish_count']:,}")
            with col4:
                st.metric("Bearish Trades", f"{cs['bearish_count']:,}")
        
        if 'impulse' in stats:
            ins = stats['impulse']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Impulses", f"{ins['total_impulses']:,}")
            with col2:
                st.metric("Avg Success Rate", f"{ins['avg_success_rate']:.1f}%")
            with col3:
                st.metric("Avg Impulse", f"{ins['avg_impulse']:.2f}")
            with col4:
                st.metric("Avg Pullback %", f"{ins['avg_pullback_pct']:.1f}%")
        
        # Insights
        st.subheader("🔍 Key Insights")
        insights = st.session_state.report_generator.generate_insights(stats)
        
        for insight in insights:
            if "⚠️" in insight or "❌" in insight:
                st.markdown(f'<div class="warning-box">{insight}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        
        # Combined charts
        if st.session_state.data_processor.crossover_data is not None:
            fig = st.session_state.chart_generator.create_session_heatmap('crossover')
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.data_processor.impulse_data is not None:
            fig = st.session_state.chart_generator.create_session_heatmap('impulse')
            if fig:
                st.plotly_chart(fig, use_container_width=True)

def show_crossover_analysis():
    """Show crossover analysis"""
    st.header("🎯 Crossover Analysis")
    
    df = st.session_state.data_processor.crossover_data
    
    if df is not None and len(df) > 0:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Crossovers", len(df))
        with col2:
            st.metric("Average Distance", f"{df['Distance'].mean():.2f}")
        with col3:
            st.metric("Average Duration", f"{df['Duration_Minutes'].mean():.1f} min")
        with col4:
            bullish_pct = (df['Direction'] == 'BULLISH').mean() * 100
            st.metric("Bullish %", f"{bullish_pct:.1f}%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = st.session_state.chart_generator.create_segment_distribution()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = st.session_state.chart_generator.create_direction_performance()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Time analysis
        fig = st.session_state.chart_generator.create_time_analysis()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance by segment
        st.subheader("📈 Performance by Segment")
        segment_stats = df.groupby('Segment').agg({
            'Distance': ['count', 'mean', 'std'],
            'Duration_Minutes': 'mean'
        }).round(2)
        
        segment_stats.columns = ['Count', 'Avg_Distance', 'Std_Distance', 'Avg_Duration']
        segment_stats = segment_stats.reset_index()
        
        st.dataframe(segment_stats, use_container_width=True)

def show_impulse_analysis():
    """Show impulse analysis"""
    st.header("⚡ Impulse Analysis")
    
    df = st.session_state.data_processor.impulse_data
    
    if df is not None and len(df) > 0:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Impulses", len(df))
        with col2:
            st.metric("Average Impulse", f"{df['Impulse'].mean():.2f}")
        with col3:
            st.metric("Success Rate", f"{df['Success_Rate'].mean():.1f}%")
        with col4:
            st.metric("Average Pullback", f"{df['Pullback_Pct'].mean():.1f}%")
        
        # Charts
        # Impulse vs Pullback scatter - Full width
        fig = px.scatter(
            df, x='Impulse', y='Pullback', color='Direction',
            title='Impulse vs Pullback Analysis',
            hover_data=['Success_Rate', 'Segment']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Threshold performance table
        st.subheader("🎯 Threshold Performance")
        threshold_cols = [col for col in df.columns if col.startswith('Hit_')]
        
        if threshold_cols:
            threshold_data = []
            for col in threshold_cols:
                threshold = col.replace('Hit_', '').replace('%', '')
                hit_rate = (df[col] == True).mean() * 100
                avg_impulse_when_hit = df[df[col] == True]['Impulse'].mean()
                
                threshold_data.append({
                    'Threshold': f"{threshold}%",
                    'Hit_Rate': f"{hit_rate:.1f}%",
                    'Avg_Impulse_When_Hit': f"{avg_impulse_when_hit:.2f}" if not pd.isna(avg_impulse_when_hit) else "N/A",
                    'Total_Hits': (df[col] == True).sum()
                })
            
            threshold_df = pd.DataFrame(threshold_data)
            st.dataframe(threshold_df, use_container_width=True)

def show_session_analysis(crossover_loaded, impulse_loaded):
    """Show comprehensive session analysis"""
    st.header("🌍 Session Analysis")
    
    if not crossover_loaded and not impulse_loaded:
        st.info("Please upload data files to view session analysis.")
        return
    
    # Create sub-tabs for different session views
    session_tab1, session_tab2, session_tab3, session_tab4, session_tab5, session_tab6 = st.tabs([
        "📊 All Sessions", "🇦🇺 Sydney", "🇯🇵 Tokyo", "🇬🇧 London", "🇺🇸 New York", "🔄 Overlaps"
    ])
    
    with session_tab1:
        st.subheader("📊 All Sessions Overview")
        
        # Overall Impulse Performance Summary
        if impulse_loaded:
            stats = st.session_state.report_generator.generate_summary_stats()
            if 'impulse' in stats:
                ins = stats['impulse']
                st.subheader("⚡ Overall Impulse Performance")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Impulses", f"{ins['total_impulses']:,}")
                with col2:
                    st.metric("Avg Success Rate", f"{ins['avg_success_rate']:.1f}%")
                with col3:
                    st.metric("Avg Impulse", f"{ins['avg_impulse']:.2f}")
                with col4:
                    st.metric("Best Threshold", f"{ins['best_threshold']}%" if ins['best_threshold'] != 'N/A' else 'N/A')
                
                # Overall Insights
                if ins['total_impulses'] > 0:
                    if ins['avg_success_rate'] > 60:
                        st.markdown(f"• ✅ Excellent overall threshold performance ({ins['avg_success_rate']:.1f}%)")
                    elif ins['avg_success_rate'] < 30:
                        st.markdown(f"• ❌ Poor overall threshold performance - Consider strategy adjustment")
                    
                    if ins['best_threshold'] != 'N/A':
                        st.markdown(f"• 🎯 Overall best performing threshold: {ins['best_threshold']}%")
                st.divider()

        # Advanced session heatmaps (demo_engines style)
        if crossover_loaded:
            stats = st.session_state.report_generator.generate_summary_stats()
            if 'crossover' in stats:
                cs = stats['crossover']
                st.subheader("🎯 Overall Crossover Performance")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", f"{cs['total_trades']:,}")
                with col2:
                    st.metric("Avg Distance", f"{cs['avg_distance']:.2f}")
                with col3:
                    st.metric("Avg Duration", f"{cs['avg_duration']:.1f} min")
                with col4:
                    st.metric("Bullish %", f"{(cs['bullish_count']/cs['total_trades']*100):.1f}%" if cs['total_trades'] > 0 else "0%")
                
                # Crossover Insights
                if cs['total_trades'] > 0:
                    if cs['avg_distance'] > 10:
                        st.markdown(f"• 📈 High overall performance - Average distance: {cs['avg_distance']:.2f}")
                    st.markdown(f"• 🌍 Overall best performing session: {cs['best_session']}")
                st.divider()

            st.subheader("🎯 Advanced Crossover Session Matrix")
            fig = st.session_state.chart_generator.create_advanced_session_heatmap('crossover')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="crossover_session_matrix")
        
        # Full-width centered Impulse vs Reversal Matrix
        if impulse_loaded:
            st.subheader("🔥 Full Impulse vs Reversal Analysis")
            fig_full = st.session_state.chart_generator.create_impulse_vs_reversal_heatmap()
            if fig_full:
                st.plotly_chart(fig_full, use_container_width=True, key="impulse_reversal_full")
                
                # Add explanation
                st.info("""
                📊 **How to Read This Matrix:**
                - **Each Row:** Shows one impulse range (e.g., 10-15 points)
                - **Each Cell:** Shows count and percentage of that row's total
                - **Colors:** White (no data) → Green (low %) → Yellow → Red (high %)
                - **Row Percentages:** Each row adds up to 100% across all reversal zones
                """)
        
        # Original session distribution heatmaps
        st.subheader("📈 Basic Session Distribution")
        
        if crossover_loaded:
            st.subheader("🎯 Crossover Session Distribution")
            fig = st.session_state.chart_generator.create_session_distribution_heatmap('crossover')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="crossover_basic_distribution")
        
        if impulse_loaded:
            st.subheader("⚡ Impulse Session Distribution")
            fig = st.session_state.chart_generator.create_session_distribution_heatmap('impulse')
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="impulse_basic_distribution")
        
        # 3D Heatmap Section
        st.subheader("🌐 3D Session Performance Topography")
        if crossover_loaded:
            fig_3d = st.session_state.chart_generator.create_3d_session_heatmap('crossover')
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True, key="session_3d_topography")
                st.info("💡 **3D Heatmap Insights:** Rotate and zoom to explore session performance patterns. Higher peaks indicate higher probability zones.")
        
        # Session comparison table
        if crossover_loaded:
            st.subheader("📈 Session Performance Comparison")
            df = st.session_state.data_processor.crossover_data
            
            sessions = ['SYDNEY', 'TOKYO', 'LONDON', 'NY']
            session_stats = []
            
            for session in sessions:
                session_mask = (
                    df['Session_Start'].str.contains(session, na=False) |
                    df['Session_Peak'].str.contains(session, na=False) |
                    df['Session_End'].str.contains(session, na=False)
                )
                session_df = df[session_mask]
                
                if len(session_df) > 0:
                    stats = {
                        'Session': session,
                        'Total_Trades': len(session_df),
                        'Avg_Distance': session_df['Distance'].mean(),
                        'Avg_Duration_Min': session_df['Duration_Minutes'].mean(),
                        'Bullish_Count': (session_df['Direction'] == 'BULLISH').sum(),
                        'Bearish_Count': (session_df['Direction'] == 'BEARISH').sum(),
                        'Bullish_Pct': (session_df['Direction'] == 'BULLISH').mean() * 100
                    }
                    session_stats.append(stats)
            
            if session_stats:
                session_comparison_df = pd.DataFrame(session_stats)
                session_comparison_df = session_comparison_df.round(2)
                st.dataframe(session_comparison_df, use_container_width=True)
    
    with session_tab2:
        show_individual_session("SYDNEY", crossover_loaded, impulse_loaded)
    
    with session_tab3:
        show_individual_session("TOKYO", crossover_loaded, impulse_loaded)
    
    with session_tab4:
        show_individual_session("LONDON", crossover_loaded, impulse_loaded)
    
    with session_tab5:
        show_individual_session("NY", crossover_loaded, impulse_loaded)
    
    with session_tab6:
        show_session_overlaps(crossover_loaded, impulse_loaded)

def show_individual_session(session_name, crossover_loaded, impulse_loaded):
    """Show detailed analysis for individual session"""
    st.subheader(f"🎯 {session_name} Session Analysis")
    
    # Session time info
    session_times = {
        'SYDNEY': '2:30 - 11:30 IST',
        'TOKYO': '5:30 - 14:30 IST', 
        'LONDON': '12:30 - 21:30 IST',
        'NY': '17:30 - 2:30 IST'
    }
    
    st.info(f"📅 {session_name} Trading Hours: {session_times.get(session_name, 'N/A')}")
    
    # Individual Session Heatmaps
    st.subheader(f"🔥 {session_name} Session Heatmaps")
    
    if crossover_loaded:
        st.subheader(f"🎯 {session_name} Distance Matrix")
        fig_heatmap = st.session_state.chart_generator.create_individual_session_heatmap(session_name, 'crossover')
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True, key=f"crossover_heatmap_{session_name}")
        else:
            st.warning(f"No crossover data for {session_name} session heatmap")

    if impulse_loaded:
        st.subheader(f"⚡ {session_name} Impulse vs Reversal Matrix")
        fig_heatmap = st.session_state.chart_generator.create_impulse_vs_reversal_heatmap(session_name)
        if fig_heatmap:
            st.plotly_chart(fig_heatmap, use_container_width=True, key=f"impulse_reversal_{session_name}")
        else:
            st.warning(f"No impulse data for {session_name} session heatmap")
    
    # Crossover analysis for this session
    if crossover_loaded:
        st.subheader("🎯 Crossover Performance")
        
        fig, stats = st.session_state.chart_generator.create_individual_session_analysis(
            session_name, 'crossover'
        )
        
        if fig and stats:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", f"{stats['Total Trades']:,}")
            with col2:
                st.metric("Avg Distance", f"{stats['Average Distance']:.2f}")
            with col3:
                st.metric("Avg Duration", f"{stats['Average Duration']:.1f} min")
            with col4:
                bullish_pct = (stats['Bullish Trades'] / stats['Total Trades'] * 100) if stats['Total Trades'] > 0 else 0
                st.metric("Bullish %", f"{bullish_pct:.1f}%")
            
            # Session insights
            st.subheader("🔍 Session Insights")
            
            insights = []
            if stats['Total Trades'] > 0:
                if bullish_pct > 60:
                    insights.append(f"🟢 Strong bullish bias in {session_name} session ({bullish_pct:.1f}%)")
                elif bullish_pct < 40:
                    insights.append(f"🔴 Strong bearish bias in {session_name} session ({bullish_pct:.1f}%)")
                else:
                    insights.append(f"⚖️ Balanced direction bias in {session_name} session")
                
                if stats['Average Distance'] > 10:
                    insights.append(f"📈 High performance session - Average distance: {stats['Average Distance']:.2f}")
                elif stats['Average Distance'] < 5:
                    insights.append(f"📉 Low performance session - Consider strategy adjustment")
                
                insights.append(f"⏰ Most active hour: {stats['Most Active Hour']}:00")
                insights.append(f"🎯 Best performing hour: {stats['Best Hour']}:00")
            
            for insight in insights:
                st.markdown(f"• {insight}")
        else:
            st.warning(f"No crossover data found for {session_name} session.")
    
    # Impulse analysis for this session
    if impulse_loaded:
        st.subheader("⚡ Impulse Performance")
        
        fig, stats = st.session_state.chart_generator.create_individual_session_analysis(
            session_name, 'impulse'
        )
        
        if fig and stats:
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Impulses", f"{stats['Total Impulses']:,}")
            with col2:
                st.metric("Avg Success Rate", f"{stats['Average Success Rate']:.1f}%")
            with col3:
                st.metric("Avg Impulse", f"{stats['Average Impulse']:.2f}")
            with col4:
                st.metric("Best Threshold", stats['Best Threshold'])
            
            # Impulse insights
            impulse_insights = []
            if stats['Total Impulses'] > 0:
                if stats['Average Success Rate'] > 60:
                    impulse_insights.append(f"✅ Excellent threshold performance in {session_name} ({stats['Average Success Rate']:.1f}%)")
                elif stats['Average Success Rate'] < 30:
                    impulse_insights.append(f"❌ Poor threshold performance in {session_name} - Consider strategy adjustment")
                
                impulse_insights.append(f"🎯 Best performing threshold: {stats['Best Threshold']}")
                
                if stats['Average Pullback'] > 50:
                    impulse_insights.append(f"⚠️ High pullback risk in {session_name}: {stats['Average Pullback']:.1f}%")
            
            for insight in impulse_insights:
                st.markdown(f"• {insight}")
        else:
            st.warning(f"No impulse data found for {session_name} session.")

def show_session_overlaps(crossover_loaded, impulse_loaded):
    """Show session overlap analysis"""
    st.subheader("🔄 Session Overlap Analysis")
    
    st.info("📅 Session Overlap Periods (IST):")
    st.markdown("""
    - **Sydney-Tokyo Overlap:** 5:30 - 11:30 IST
    - **Tokyo-London Overlap:** 12:30 - 14:30 IST  
    - **London-NY Overlap:** 17:30 - 21:30 IST
    - **NY-Sydney Overlap:** 2:30 - 2:30 IST (Next Day)
    """)
    
    if crossover_loaded:
        df = st.session_state.data_processor.crossover_data
        
        # Define overlap periods
        overlaps = {
            'Sydney-Tokyo': ['SYDNEY TOKYO'],
            'Tokyo-London': ['TOKYO LONDON'],
            'London-NY': ['LONDON NY'],
            'NY-Sydney': ['NY SYDNEY']
        }
        
        overlap_stats = []
        
        for overlap_name, overlap_patterns in overlaps.items():
            overlap_trades = 0
            total_distance = 0
            
            for pattern in overlap_patterns:
                # Check all session columns for the overlap pattern
                overlap_mask = (
                    df['Session_Start'].str.contains(pattern, na=False) |
                    df['Session_Peak'].str.contains(pattern, na=False) |
                    df['Session_End'].str.contains(pattern, na=False)
                )
                overlap_df = df[overlap_mask]
                overlap_trades += len(overlap_df)
                total_distance += overlap_df['Distance'].sum()
            
            avg_distance = total_distance / overlap_trades if overlap_trades > 0 else 0
            
            overlap_stats.append({
                'Overlap_Period': overlap_name,
                'Total_Trades': overlap_trades,
                'Avg_Distance': avg_distance,
                'Percentage_of_Total': (overlap_trades / len(df) * 100) if len(df) > 0 else 0
            })
        
        overlap_df = pd.DataFrame(overlap_stats)
        overlap_df = overlap_df.round(2)
        
        # Display overlap statistics
        st.subheader("📊 Overlap Performance Statistics")
        st.dataframe(overlap_df, use_container_width=True)
        
        # Overlap performance chart
        if len(overlap_df) > 0:
            fig = px.bar(
                overlap_df,
                x='Overlap_Period',
                y='Total_Trades',
                title='Trades During Session Overlaps',
                color='Avg_Distance',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Overlap insights
        st.subheader("🔍 Overlap Insights")
        
        if len(overlap_df) > 0:
            best_overlap = overlap_df.loc[overlap_df['Avg_Distance'].idxmax()]
            most_active_overlap = overlap_df.loc[overlap_df['Total_Trades'].idxmax()]
            
            st.markdown(f"• 🏆 **Best performing overlap:** {best_overlap['Overlap_Period']} (Avg Distance: {best_overlap['Avg_Distance']:.2f})")
            st.markdown(f"• 📈 **Most active overlap:** {most_active_overlap['Overlap_Period']} ({most_active_overlap['Total_Trades']} trades)")
            
            total_overlap_trades = overlap_df['Total_Trades'].sum()
            overlap_percentage = (total_overlap_trades / len(df) * 100) if len(df) > 0 else 0
            st.markdown(f"• 📊 **Overlap activity:** {overlap_percentage:.1f}% of all trades occur during session overlaps")

def show_data_tables():
    """Show raw data tables"""
    st.header("📋 Data Tables")
    
    # Crossover data
    if st.session_state.data_processor.crossover_data is not None:
        st.subheader("🎯 Crossover Stats Data")
        
        df = st.session_state.data_processor.crossover_data
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            direction_filter = st.multiselect(
                "Filter by Direction",
                options=df['Direction'].unique(),
                default=df['Direction'].unique(),
                key="crossover_direction"
            )
        
        with col2:
            segment_filter = st.multiselect(
                "Filter by Segment",
                options=df['Segment'].unique(),
                default=df['Segment'].unique(),
                key="crossover_segment"
            )
        
        with col3:
            min_distance = st.number_input(
                "Min Distance",
                min_value=0.0,
                max_value=float(df['Distance'].max()),
                value=0.0,
                key="crossover_min_distance"
            )
        
        # Apply filters
        filtered_df = df[
            (df['Direction'].isin(direction_filter)) &
            (df['Segment'].isin(segment_filter)) &
            (df['Distance'] >= min_distance)
        ]
        
        st.write(f"Showing {len(filtered_df)} of {len(df)} records")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"crossover_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Impulse data
    if st.session_state.data_processor.impulse_data is not None:
        st.subheader("⚡ Impulse Reversal Data")
        
        df = st.session_state.data_processor.impulse_data
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            direction_filter = st.multiselect(
                "Filter by Direction",
                options=df['Direction'].unique(),
                default=df['Direction'].unique(),
                key="impulse_direction"
            )
        
        with col2:
            min_success_rate = st.slider(
                "Min Success Rate (%)",
                min_value=0,
                max_value=100,
                value=0,
                key="impulse_success_rate"
            )
        
        with col3:
            min_impulse = st.number_input(
                "Min Impulse",
                min_value=0.0,
                max_value=float(df['Impulse'].max()),
                value=0.0,
                key="impulse_min_impulse"
            )
        
        # Apply filters
        filtered_df = df[
            (df['Direction'].isin(direction_filter)) &
            (df['Success_Rate'] >= min_success_rate) &
            (df['Impulse'] >= min_impulse)
        ]
        
        st.write(f"Showing {len(filtered_df)} of {len(df)} records")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"impulse_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()