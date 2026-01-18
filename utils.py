import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64

class DataProcessor:
    def __init__(self):
        self.crossover_data = None
        self.impulse_data = None
        
    def load_crossover_data(self, file):
        """Load and process crossover stats data"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            # Convert datetime columns
            df['StartTime'] = pd.to_datetime(df['StartTime'])
            df['EndTime'] = pd.to_datetime(df['EndTime'])
            
            # Extract hour for session analysis
            df['StartHour'] = df['StartTime'].dt.hour
            df['EndHour'] = df['EndTime'].dt.hour
            
            # Calculate duration
            df['Duration_Minutes'] = (df['EndTime'] - df['StartTime']).dt.total_seconds() / 60
            
            self.crossover_data = df
            return df
        except Exception as e:
            raise Exception(f"Error loading crossover data: {str(e)}")
    
    def load_impulse_data(self, file):
        """Load and process impulse reversal data"""
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            # Convert datetime columns
            df['Time'] = pd.to_datetime(df['Time'])
            df['Hour'] = df['Time'].dt.hour
            
            # Calculate success metrics
            threshold_cols = [col for col in df.columns if col.startswith('Hit_')]
            df['Thresholds_Hit'] = df[threshold_cols].sum(axis=1)
            df['Success_Rate'] = df['Thresholds_Hit'] / len(threshold_cols) * 100
            
            # Calculate pullback percentage
            df['Pullback_Pct'] = (df['Pullback'] / df['Impulse'] * 100).round(2)
            
            self.impulse_data = df
            return df
        except Exception as e:
            raise Exception(f"Error loading impulse data: {str(e)}")

class ChartGenerator:
    def __init__(self, data_processor):
        self.dp = data_processor
        
    def create_session_heatmap(self, data_type='crossover'):
        """Create session performance heatmap"""
        if data_type == 'crossover' and self.dp.crossover_data is not None:
            df = self.dp.crossover_data
            
            # Create session matrix
            sessions = ['SYDNEY', 'TOKYO', 'LONDON', 'NY']
            directions = ['BULLISH', 'BEARISH']
            
            heatmap_data = []
            for session in sessions:
                row = []
                for direction in directions:
                    # Count occurrences where session appears in any session column
                    mask = (
                        df['Session_Start'].str.contains(session, na=False) |
                        df['Session_Peak'].str.contains(session, na=False) |
                        df['Session_End'].str.contains(session, na=False)
                    ) & (df['Direction'] == direction)
                    
                    count = mask.sum()
                    avg_distance = df[mask]['Distance'].mean() if count > 0 else 0
                    row.append(avg_distance)
                heatmap_data.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=directions,
                y=sessions,
                colorscale='Viridis',
                text=[[f'{val:.2f}' for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont=dict(size=14, family="Arial", color="white"),
                colorbar=dict(title="Avg Distance", titlefont=dict(size=14))
            ))
            
            fig.update_layout(
                title=dict(
                    text='Session Performance Heatmap (Average Distance)',
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title='Direction',
                yaxis_title='Trading Session',
                height=400,
                autosize=True,
                margin=dict(l=50, r=50, t=100, b=80),
                template="plotly_dark",
                font=dict(size=12)
            )
            
            return fig
            
        elif data_type == 'impulse' and self.dp.impulse_data is not None:
            df = self.dp.impulse_data
            
            sessions = ['SYDNEY', 'TOKYO', 'LONDON', 'NY']
            directions = ['BULLISH', 'BEARISH']
            
            heatmap_data = []
            for session in sessions:
                row = []
                for direction in directions:
                    mask = (
                        df['Session_Entry'].str.contains(session, na=False) |
                        df['Session_Peak'].str.contains(session, na=False) |
                        df['Session_Trigger'].str.contains(session, na=False)
                    ) & (df['Direction'] == direction)
                    
                    avg_success = df[mask]['Success_Rate'].mean() if mask.sum() > 0 else 0
                    row.append(avg_success)
                heatmap_data.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=directions,
                y=sessions,
                colorscale='RdYlGn',
                text=[[f'{val:.1f}%' for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont=dict(size=14, family="Arial", color="white"),
                colorbar=dict(title="Success Rate %", titlefont=dict(size=14))
            ))
            
            fig.update_layout(
                title=dict(
                    text='Session Success Rate Heatmap (Impulse Analysis)',
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title='Direction',
                yaxis_title='Trading Session',
                height=400,
                autosize=True,
                margin=dict(l=50, r=50, t=100, b=80),
                template="plotly_dark",
                font=dict(size=12)
            )
            
            return fig
    
    def create_segment_distribution(self):
        """Create segment distribution chart"""
        if self.dp.crossover_data is not None:
            df = self.dp.crossover_data
            
            segment_counts = df['Segment'].value_counts()
            
            fig = px.bar(
                x=segment_counts.index,
                y=segment_counts.values,
                title='Distance Segment Distribution',
                labels={'x': 'Segment', 'y': 'Count'},
                color=segment_counts.values,
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(height=400, showlegend=False)
            return fig
    
    def create_threshold_analysis(self):
        """Create threshold hit rate analysis"""
        if self.dp.impulse_data is not None:
            df = self.dp.impulse_data
            
            threshold_cols = [col for col in df.columns if col.startswith('Hit_')]
            threshold_rates = []
            
            for col in threshold_cols:
                rate = (df[col] == True).sum() / len(df) * 100
                threshold_rates.append({
                    'Threshold': col.replace('Hit_', '').replace('%', ''),
                    'Success_Rate': rate
                })
            
            threshold_df = pd.DataFrame(threshold_rates)
            
            fig = px.bar(
                threshold_df,
                x='Threshold',
                y='Success_Rate',
                title='Threshold Hit Rates',
                labels={'Success_Rate': 'Success Rate (%)', 'Threshold': 'Threshold (%)'},
                color='Success_Rate',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(height=400, showlegend=False)
            return fig
    
    def create_direction_performance(self):
        """Create direction performance comparison"""
        if self.dp.crossover_data is not None:
            df = self.dp.crossover_data
            
            direction_stats = df.groupby('Direction').agg({
                'Distance': ['mean', 'count'],
                'Duration_Minutes': 'mean'
            }).round(2)
            
            direction_stats.columns = ['Avg_Distance', 'Count', 'Avg_Duration']
            direction_stats = direction_stats.reset_index()
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Average Distance', 'Trade Count', 'Average Duration (min)'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Average Distance
            fig.add_trace(
                go.Bar(x=direction_stats['Direction'], y=direction_stats['Avg_Distance'], 
                       name='Avg Distance', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Count
            fig.add_trace(
                go.Bar(x=direction_stats['Direction'], y=direction_stats['Count'], 
                       name='Count', marker_color='lightgreen'),
                row=1, col=2
            )
            
            # Duration
            fig.add_trace(
                go.Bar(x=direction_stats['Direction'], y=direction_stats['Avg_Duration'], 
                       name='Avg Duration', marker_color='lightcoral'),
                row=1, col=3
            )
            
            fig.update_layout(
                title_text="Direction Performance Analysis",
                showlegend=False,
                height=400
            )
            
            return fig
    
    def create_time_analysis(self):
        """Create time-based analysis"""
        if self.dp.crossover_data is not None:
            df = self.dp.crossover_data
            
            hourly_stats = df.groupby('StartHour').agg({
                'Distance': 'mean',
                'Direction': 'count'
            }).round(2)
            
            hourly_stats.columns = ['Avg_Distance', 'Count']
            hourly_stats = hourly_stats.reset_index()
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=hourly_stats['StartHour'], y=hourly_stats['Avg_Distance'],
                          mode='lines+markers', name='Avg Distance', line=dict(color='blue')),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Bar(x=hourly_stats['StartHour'], y=hourly_stats['Count'],
                       name='Trade Count', opacity=0.6, marker_color='lightgreen'),
                secondary_y=True,
            )
            
            fig.update_xaxes(title_text="Hour of Day")
            fig.update_yaxes(title_text="Average Distance", secondary_y=False)
            fig.update_yaxes(title_text="Trade Count", secondary_y=True)
            
            fig.update_layout(
                title_text="Hourly Performance Analysis",
                height=400
            )
            
            return fig
    
    def create_session_distribution_heatmap(self, data_type='crossover'):
        """Create session distribution heatmap with counts"""
        if data_type == 'crossover' and self.dp.crossover_data is not None:
            df = self.dp.crossover_data
            
            # Define sessions and metrics
            sessions = ['SYDNEY', 'TOKYO', 'LONDON', 'NY']
            metrics = ['Total Trades', 'Bullish Trades', 'Bearish Trades', 'Avg Distance', 'Avg Duration']
            
            # Calculate session data
            heatmap_data = []
            annotations = []
            
            for i, session in enumerate(sessions):
                row = []
                # Filter data for this session
                session_mask = (
                    df['Session_Start'].str.contains(session, na=False) |
                    df['Session_Peak'].str.contains(session, na=False) |
                    df['Session_End'].str.contains(session, na=False)
                )
                session_df = df[session_mask]
                
                # Total trades
                total_trades = len(session_df)
                row.append(total_trades)
                annotations.append(dict(x=0, y=i, text=f"{total_trades}", showarrow=False, font=dict(color="white", size=12)))
                
                # Bullish trades
                bullish_count = (session_df['Direction'] == 'BULLISH').sum()
                row.append(bullish_count)
                annotations.append(dict(x=1, y=i, text=f"{bullish_count}", showarrow=False, font=dict(color="white", size=12)))
                
                # Bearish trades
                bearish_count = (session_df['Direction'] == 'BEARISH').sum()
                row.append(bearish_count)
                annotations.append(dict(x=2, y=i, text=f"{bearish_count}", showarrow=False, font=dict(color="white", size=12)))
                
                # Average distance
                avg_distance = session_df['Distance'].mean() if len(session_df) > 0 else 0
                row.append(avg_distance)
                annotations.append(dict(x=3, y=i, text=f"{avg_distance:.1f}", showarrow=False, font=dict(color="white", size=12)))
                
                # Average duration
                avg_duration = session_df['Duration_Minutes'].mean() if len(session_df) > 0 else 0
                row.append(avg_duration)
                annotations.append(dict(x=4, y=i, text=f"{avg_duration:.1f}m", showarrow=False, font=dict(color="white", size=12)))
                
                heatmap_data.append(row)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=metrics,
                y=sessions,
                colorscale='Viridis',
                showscale=True,
                textfont=dict(size=14, family="Arial", color="white"),
                colorbar=dict(title="Count/Value", titlefont=dict(size=14))
            ))
            
            # Update annotation sizes
            for ann in annotations:
                ann['font']['size'] = 14
            
            # Add annotations
            fig.update_layout(
                annotations=annotations,
                title=dict(
                    text=f'Session Distribution Analysis - Total Events: {len(df)}',
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title='Metrics',
                yaxis_title='Trading Sessions',
                height=400,
                autosize=True,
                margin=dict(l=50, r=50, t=100, b=80),
                template="plotly_dark",
                font=dict(size=12)
            )
            
            return fig
            
        elif data_type == 'impulse' and self.dp.impulse_data is not None:
            df = self.dp.impulse_data
            
            sessions = ['SYDNEY', 'TOKYO', 'LONDON', 'NY']
            metrics = ['Total Impulses', 'Avg Success Rate', 'Avg Impulse', 'Hit 20%', 'Hit 30%']
            
            heatmap_data = []
            annotations = []
            
            for i, session in enumerate(sessions):
                row = []
                # Filter data for this session
                session_mask = (
                    df['Session_Entry'].str.contains(session, na=False) |
                    df['Session_Peak'].str.contains(session, na=False) |
                    df['Session_Trigger'].str.contains(session, na=False)
                )
                session_df = df[session_mask]
                
                # Total impulses
                total_impulses = len(session_df)
                row.append(total_impulses)
                annotations.append(dict(x=0, y=i, text=f"{total_impulses}", showarrow=False, font=dict(color="white", size=12)))
                
                # Average success rate
                avg_success = session_df['Success_Rate'].mean() if len(session_df) > 0 else 0
                row.append(avg_success)
                annotations.append(dict(x=1, y=i, text=f"{avg_success:.1f}%", showarrow=False, font=dict(color="white", size=12)))
                
                # Average impulse
                avg_impulse = session_df['Impulse'].mean() if len(session_df) > 0 else 0
                row.append(avg_impulse)
                annotations.append(dict(x=2, y=i, text=f"{avg_impulse:.1f}", showarrow=False, font=dict(color="white", size=12)))
                
                # Hit 20%
                hit_20 = (session_df['Hit_20%'] == True).sum() if 'Hit_20%' in session_df.columns else 0
                row.append(hit_20)
                annotations.append(dict(x=3, y=i, text=f"{hit_20}", showarrow=False, font=dict(color="white", size=12)))
                
                # Hit 30%
                hit_30 = (session_df['Hit_30%'] == True).sum() if 'Hit_30%' in session_df.columns else 0
                row.append(hit_30)
                annotations.append(dict(x=4, y=i, text=f"{hit_30}", showarrow=False, font=dict(color="white", size=12)))
                
                heatmap_data.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=metrics,
                y=sessions,
                colorscale='RdYlGn',
                showscale=True,
                textfont=dict(size=14, family="Arial", color="white"),
                colorbar=dict(title="Count/Value", titlefont=dict(size=14))
            ))
            
            # Update annotation sizes
            for ann in annotations:
                ann['font']['size'] = 14
            
            fig.update_layout(
                annotations=annotations,
                title=dict(
                    text=f'Session Impulse Analysis - Total Events: {len(df)}',
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title='Metrics',
                yaxis_title='Trading Sessions',
                height=400,
                autosize=True,
                margin=dict(l=50, r=50, t=100, b=80),
                template="plotly_dark",
                font=dict(size=12)
            )
            
            return fig
    
    def create_advanced_session_heatmap(self, data_type='crossover'):
        """Create advanced session heatmap matrix like demo_engines style"""
        if data_type == 'crossover' and self.dp.crossover_data is not None:
            df = self.dp.crossover_data
            
            # Define sessions and distance ranges (like impulse ranges in demo)
            sessions = ['SYDNEY', 'TOKYO', 'LONDON', 'NY']
            
            # Define distance bins (0-5, 5-10, 10-15, 15-20, 20+)
            distance_bins = [0, 5, 10, 15, 20, float('inf')]
            distance_labels = ['0-5', '5-10', '10-15', '15-20', '20+']
            
            # Calculate matrix data
            matrix_counts = []
            matrix_pcts = []
            text_matrix = []
            custom_data = []
            y_labels = []
            
            for session in sessions:
                # Filter data for this session
                session_mask = (
                    df['Session_Start'].str.contains(session, na=False) |
                    df['Session_Peak'].str.contains(session, na=False) |
                    df['Session_End'].str.contains(session, na=False)
                )
                session_df = df[session_mask]
                
                # Create label with total count
                y_labels.append(f"{session} (N={len(session_df)})")
                
                if len(session_df) == 0:
                    # Empty row
                    matrix_counts.append([0] * len(distance_labels))
                    matrix_pcts.append([0] * len(distance_labels))
                    text_matrix.append([''] * len(distance_labels))
                    custom_data.append([[0, 0]] * len(distance_labels))
                else:
                    # Bin the distances
                    binned = pd.cut(session_df['Distance'], bins=distance_bins, labels=distance_labels, right=False)
                    counts_series = binned.value_counts().reindex(distance_labels).fillna(0)
                    
                    # Raw counts
                    row_counts = counts_series.tolist()
                    matrix_counts.append(row_counts)
                    
                    # Percentages
                    row_total = sum(row_counts)
                    if row_total > 0:
                        row_pcts = [(c / row_total) * 100.0 for c in row_counts]
                    else:
                        row_pcts = [0.0] * len(row_counts)
                    matrix_pcts.append(row_pcts)
                    
                    # Calculate average duration for each bin
                    row_text = []
                    row_custom = []
                    
                    for i, label in enumerate(distance_labels):
                        count = int(row_counts[i])
                        pct = row_pcts[i]
                        
                        if count > 0:
                            # Get average duration for this bin
                            if label == '20+':
                                bin_mask = session_df['Distance'] >= 20
                            else:
                                start, end = map(float, label.split('-'))
                                bin_mask = (session_df['Distance'] >= start) & (session_df['Distance'] < end)
                            
                            bin_df = session_df[bin_mask]
                            avg_duration = bin_df['Duration_Minutes'].mean() if len(bin_df) > 0 else 0
                            
                            # Simple text format - consistent sizing
                            row_text.append(f"{count}<br>({pct:.1f}%)<br>{avg_duration:.1f}m")
                            row_custom.append([count, avg_duration])
                        else:
                            row_text.append("")
                            row_custom.append([0, 0])
                    
                    text_matrix.append(row_text)
                    custom_data.append(row_custom)
            
            # Create heatmap with demo_engines style
            fig = go.Figure(data=go.Heatmap(
                z=matrix_pcts,
                x=distance_labels,
                y=y_labels,
                # Risk Matrix Style colorscale like demo_engines
                colorscale=[
                    [0.0, 'white'],       # 0% = White
                    [0.01, '#90EE90'],    # >0% = Light Green
                    [0.5, 'yellow'],      # 50% = Yellow
                    [1.0, 'red']          # 100% = Red
                ],
                reversescale=False,
                zmin=0, zmax=50,       # Cap at 50% for visibility
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=14, family="Arial", color="white"),  # Consistent text formatting
                customdata=custom_data,
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>Distance Range: %{x}<br>Probability: %{z:.1f}%<br>Count: %{customdata[0]}<br>Avg Duration: %{customdata[1]:.1f}m<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"Session vs Distance Range Matrix - Total Events: {len(df)}",
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title="Distance Range (Points)",
                yaxis_title="Trading Session",
                template="plotly_dark",
                height=len(y_labels) * 100 + 250,
                autosize=True,
                xaxis=dict(side="bottom"),
                font=dict(size=12),
                margin=dict(l=50, r=50, t=100, b=80)  # Minimal margins for full width
            )
            
            return fig
            
        elif data_type == 'impulse' and self.dp.impulse_data is not None:
            df = self.dp.impulse_data
            
            sessions = ['SYDNEY', 'TOKYO', 'LONDON', 'NY']
            
            # Define impulse bins
            impulse_bins = [0, 10, 15, 20, 25, float('inf')]
            impulse_labels = ['0-10', '10-15', '15-20', '20-25', '25+']
            
            matrix_counts = []
            matrix_pcts = []
            text_matrix = []
            custom_data = []
            y_labels = []
            
            for session in sessions:
                # Filter data for this session
                session_mask = (
                    df['Session_Entry'].str.contains(session, na=False) |
                    df['Session_Peak'].str.contains(session, na=False) |
                    df['Session_Trigger'].str.contains(session, na=False)
                )
                session_df = df[session_mask]
                
                y_labels.append(f"{session} (N={len(session_df)})")
                
                if len(session_df) == 0:
                    matrix_counts.append([0] * len(impulse_labels))
                    matrix_pcts.append([0] * len(impulse_labels))
                    text_matrix.append([''] * len(impulse_labels))
                    custom_data.append([[0, 0]] * len(impulse_labels))
                else:
                    # Bin the impulses
                    binned = pd.cut(session_df['Impulse'], bins=impulse_bins, labels=impulse_labels, right=False)
                    counts_series = binned.value_counts().reindex(impulse_labels).fillna(0)
                    
                    row_counts = counts_series.tolist()
                    matrix_counts.append(row_counts)
                    
                    row_total = sum(row_counts)
                    if row_total > 0:
                        row_pcts = [(c / row_total) * 100.0 for c in row_counts]
                    else:
                        row_pcts = [0.0] * len(row_counts)
                    matrix_pcts.append(row_pcts)
                    
                    # Calculate average success rate for each bin
                    row_text = []
                    row_custom = []
                    
                    for i, label in enumerate(impulse_labels):
                        count = int(row_counts[i])
                        pct = row_pcts[i]
                        
                        if count > 0:
                            # Get average success rate for this bin
                            if label == '25+':
                                bin_mask = session_df['Impulse'] >= 25
                            else:
                                start, end = map(float, label.split('-'))
                                bin_mask = (session_df['Impulse'] >= start) & (session_df['Impulse'] < end)
                            
                            bin_df = session_df[bin_mask]
                            avg_success = bin_df['Success_Rate'].mean() if len(bin_df) > 0 else 0
                            
                            # Simple text format - consistent sizing
                            row_text.append(f"{count}<br>({pct:.1f}%)<br>SR:{avg_success:.1f}%")
                            row_custom.append([count, avg_success])
                        else:
                            row_text.append("")
                            row_custom.append([0, 0])
                    
                    text_matrix.append(row_text)
                    custom_data.append(row_custom)
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix_pcts,
                x=impulse_labels,
                y=y_labels,
                colorscale=[
                    [0.0, 'white'],
                    [0.01, '#90EE90'],
                    [0.5, 'yellow'],
                    [1.0, 'red']
                ],
                reversescale=False,
                zmin=0, zmax=50,
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=14, family="Arial", color="white"),  # Consistent text formatting
                customdata=custom_data,
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>Impulse Range: %{x}<br>Probability: %{z:.1f}%<br>Count: %{customdata[0]}<br>Avg Success Rate: %{customdata[1]:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"Session vs Impulse Range Matrix - Total Events: {len(df)}",
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title="Impulse Range (Points)",
                yaxis_title="Trading Session",
                template="plotly_dark",
                height=len(y_labels) * 100 + 250,
                autosize=True,
                xaxis=dict(side="bottom"),
                font=dict(size=12),
                margin=dict(l=50, r=50, t=100, b=80)  # Minimal margins for full width
            )
            
            return fig
    
    def create_individual_session_analysis(self, session_name, data_type='crossover'):
        """Create detailed analysis for individual session"""
        if data_type == 'crossover' and self.dp.crossover_data is not None:
            df = self.dp.crossover_data
            
            # Filter for specific session
            session_mask = (
                df['Session_Start'].str.contains(session_name, na=False) |
                df['Session_Peak'].str.contains(session_name, na=False) |
                df['Session_End'].str.contains(session_name, na=False)
            )
            session_df = df[session_mask]
            
            if len(session_df) == 0:
                return None, None
            
            # Create hourly breakdown
            hourly_data = session_df.groupby('StartHour').agg({
                'Distance': ['count', 'mean'],
                'Direction': lambda x: (x == 'BULLISH').sum()
            }).round(2)
            
            hourly_data.columns = ['Trade_Count', 'Avg_Distance', 'Bullish_Count']
            hourly_data = hourly_data.reset_index()
            hourly_data['Bearish_Count'] = hourly_data['Trade_Count'] - hourly_data['Bullish_Count']
            
            # Hourly performance chart
            fig1 = make_subplots(
                rows=2, cols=1,
                subplot_titles=(f'{session_name} Session - Hourly Trade Distribution', 
                               f'{session_name} Session - Direction Breakdown'),
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
            )
            
            # Trade count and average distance
            fig1.add_trace(
                go.Bar(x=hourly_data['StartHour'], y=hourly_data['Trade_Count'],
                       name='Trade Count', marker_color='lightblue'),
                row=1, col=1, secondary_y=False
            )
            
            fig1.add_trace(
                go.Scatter(x=hourly_data['StartHour'], y=hourly_data['Avg_Distance'],
                          mode='lines+markers', name='Avg Distance', line=dict(color='red')),
                row=1, col=1, secondary_y=True
            )
            
            # Direction breakdown
            fig1.add_trace(
                go.Bar(x=hourly_data['StartHour'], y=hourly_data['Bullish_Count'],
                       name='Bullish', marker_color='green'),
                row=2, col=1
            )
            
            fig1.add_trace(
                go.Bar(x=hourly_data['StartHour'], y=hourly_data['Bearish_Count'],
                       name='Bearish', marker_color='red'),
                row=2, col=1
            )
            
            fig1.update_xaxes(title_text="Hour of Day", row=2, col=1)
            fig1.update_yaxes(title_text="Trade Count", row=1, col=1, secondary_y=False)
            fig1.update_yaxes(title_text="Average Distance", row=1, col=1, secondary_y=True)
            fig1.update_yaxes(title_text="Count", row=2, col=1)
            
            fig1.update_layout(
                height=600,
                title_text=f"{session_name} Session Analysis - Total Trades: {len(session_df)}"
            )
            
            # Session statistics
            stats = {
                'Total Trades': len(session_df),
                'Bullish Trades': (session_df['Direction'] == 'BULLISH').sum(),
                'Bearish Trades': (session_df['Direction'] == 'BEARISH').sum(),
                'Average Distance': session_df['Distance'].mean(),
                'Average Duration': session_df['Duration_Minutes'].mean(),
                'Best Hour': hourly_data.loc[hourly_data['Avg_Distance'].idxmax(), 'StartHour'] if len(hourly_data) > 0 else 'N/A',
                'Most Active Hour': hourly_data.loc[hourly_data['Trade_Count'].idxmax(), 'StartHour'] if len(hourly_data) > 0 else 'N/A'
            }
            
            return fig1, stats
            
        elif data_type == 'impulse' and self.dp.impulse_data is not None:
            df = self.dp.impulse_data
            
            # Filter for specific session
            session_mask = (
                df['Session_Entry'].str.contains(session_name, na=False) |
                df['Session_Peak'].str.contains(session_name, na=False) |
                df['Session_Trigger'].str.contains(session_name, na=False)
            )
            session_df = df[session_mask]
            
            if len(session_df) == 0:
                return None, None
            
            # Create threshold analysis for this session
            threshold_cols = [col for col in session_df.columns if col.startswith('Hit_')]
            threshold_data = []
            
            for col in threshold_cols:
                threshold = col.replace('Hit_', '').replace('%', '')
                hit_count = (session_df[col] == True).sum()
                hit_rate = hit_count / len(session_df) * 100
                threshold_data.append({
                    'Threshold': f"{threshold}%",
                    'Hit_Count': hit_count,
                    'Hit_Rate': hit_rate
                })
            
            threshold_df = pd.DataFrame(threshold_data)
            
            # Create threshold chart
            fig1 = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f'{session_name} - Threshold Hit Counts', 
                               f'{session_name} - Threshold Success Rates'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig1.add_trace(
                go.Bar(x=threshold_df['Threshold'], y=threshold_df['Hit_Count'],
                       name='Hit Count', marker_color='lightgreen'),
                row=1, col=1
            )
            
            fig1.add_trace(
                go.Bar(x=threshold_df['Threshold'], y=threshold_df['Hit_Rate'],
                       name='Hit Rate %', marker_color='orange'),
                row=1, col=2
            )
            
            fig1.update_layout(
                height=400,
                title_text=f"{session_name} Session Impulse Analysis - Total Impulses: {len(session_df)}"
            )
            
            # Session statistics
            stats = {
                'Total Impulses': len(session_df),
                'Average Success Rate': session_df['Success_Rate'].mean(),
                'Average Impulse': session_df['Impulse'].mean(),
                'Average Pullback': session_df['Pullback_Pct'].mean() if 'Pullback_Pct' in session_df.columns else 0,
                'Best Threshold': threshold_df.loc[threshold_df['Hit_Rate'].idxmax(), 'Threshold'] if len(threshold_df) > 0 else 'N/A'
            }
            
            return fig1, stats
        
        return None, None
    
    def create_impulse_vs_reversal_heatmap(self, session_name=None, session_filter=None):
        """Create Impulse vs Reversal heatmap exactly like demo_engines"""
        if self.dp.impulse_data is None:
            return None
            
        df = self.dp.impulse_data.copy()
        
        # Handle both parameter names for backward compatibility
        session = session_name or session_filter
        
        # Filter for specific session if provided
        if session:
            session_mask = (
                df['Session_Entry'].str.contains(session, na=False) |
                df['Session_Peak'].str.contains(session, na=False) |
                df['Session_Trigger'].str.contains(session, na=False)
            )
            df = df[session_mask]
            
        if len(df) == 0:
            return None
        
        # Calculate Reversal % from Pullback and Impulse
        df['Reversal%'] = (df['Pullback'] / df['Impulse'] * 100).fillna(0)
        
        # Define Impulse ranges (like demo_engines)
        impulse_ranges = [(0, 10), (10, 15), (15, 20), (20, 25), (25, float('inf'))]
        
        # Define Reversal bins (0-100% in 5% steps like demo_engines)
        reversal_bins = np.arange(0, 105, 5)
        reversal_labels = [f"{int(reversal_bins[i])}-{int(reversal_bins[i+1])}%" for i in range(len(reversal_bins)-1)]
        
        # Calculate matrix data
        matrix_counts = []
        matrix_pcts = []
        text_matrix = []
        custom_data = []
        y_labels = []
        
        for start, end in impulse_ranges:
            # Filter data for this impulse range
            if end == float('inf'):
                mask = df['Impulse'] >= start
                range_label = f"Impulse {start}+"
            else:
                mask = (df['Impulse'] >= start) & (df['Impulse'] < end)
                range_label = f"Impulse {start}-{end}"
                
            subset = df[mask]
            
            # Create label with total count and percentage of all data
            total_all_data = len(df)
            row_percentage_of_total = (len(subset) / total_all_data * 100) if total_all_data > 0 else 0
            y_labels.append(f"{range_label} (N={len(subset)}, {row_percentage_of_total:.1f}% of total)")
            
            if len(subset) == 0:
                # Empty row
                matrix_counts.append([0] * len(reversal_labels))
                matrix_pcts.append([0] * len(reversal_labels))
                text_matrix.append([''] * len(reversal_labels))
                custom_data.append([[0, 0]] * len(reversal_labels))
            else:
                # Bin the Reversal %
                binned = pd.cut(subset['Reversal%'], bins=reversal_bins, labels=reversal_labels, right=False, include_lowest=True)
                counts_series = binned.value_counts().reindex(reversal_labels).fillna(0)
                
                # Raw counts
                row_counts = counts_series.tolist()
                matrix_counts.append(row_counts)
                
                # Percentages (of this row's total)
                row_total = sum(row_counts)
                if row_total > 0:
                    row_pcts = [(c / row_total) * 100.0 for c in row_counts]
                else:
                    row_pcts = [0.0] * len(row_counts)
                matrix_pcts.append(row_pcts)
                
                # Calculate average impulse for each reversal bin
                row_text = []
                row_custom = []
                
                for i, label in enumerate(reversal_labels):
                    count = int(row_counts[i])
                    pct = row_pcts[i]
                    
                    if count > 0:
                        # Get average impulse for this reversal bin
                        reversal_start = reversal_bins[i]
                        reversal_end = reversal_bins[i + 1]
                        
                        bin_mask = (subset['Reversal%'] >= reversal_start) & (subset['Reversal%'] < reversal_end)
                        bin_df = subset[bin_mask]
                        avg_impulse = bin_df['Impulse'].mean() if len(bin_df) > 0 else 0
                        
                        # Simple text format that Plotly can render properly
                        row_text.append(f"{count}<br>({pct:.1f}%)<br>{avg_impulse:.1f}")
                        row_custom.append([count, avg_impulse])
                    else:
                        row_text.append("")
                        row_custom.append([0, 0])
                
                text_matrix.append(row_text)
                custom_data.append(row_custom)
        
        # Create heatmap with demo_engines style and maximum text visibility
        fig = go.Figure(data=go.Heatmap(
            z=matrix_pcts,
            x=reversal_labels,
            y=y_labels,
            # Risk Matrix Style colorscale exactly like demo_engines
            colorscale=[
                [0.0, 'white'],       # 0% = White
                [0.01, '#90EE90'],    # >0% = Light Green
                [0.5, 'yellow'],      # 50% = Yellow
                [1.0, 'red']          # 100% = Red
            ],
            reversescale=False,
            zmin=0, zmax=50,       # Cap at 50% for visibility
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=14, family="Arial", color="black"),  # Readable size with standard font
            customdata=custom_data,
            hoverongaps=False,
            showscale=True,  # Show color scale
            colorbar=dict(title="Probability %", titlefont=dict(size=14)),
            hovertemplate='<b>%{y}</b><br>Reversal: %{x}<br>Row Probability: %{z:.1f}%<br>Count: %{customdata[0]}<br>Avg Impulse: %{customdata[1]:.1f}<extra></extra>'
        ))
        
        session_suffix = f" - {session} Session" if session else ""
        total_events = len(df)
        
        fig.update_layout(
            title=dict(
                text=f"Impulse vs Reversal Matrix{session_suffix} - Total Events: {total_events}",
                x=0.5,  # Center the title
                xanchor='center'
            ),
            xaxis_title="Reversal % Zone",
            yaxis_title="Impulse Range",
            template="plotly_white",  # White background for better text visibility
            height=len(y_labels) * 120 + 300,  # Proportional cell height
            autosize=True,  # Enable auto-sizing for proper centering
            xaxis=dict(side="bottom"),
            font=dict(size=12),  # Proportional base font
            margin=dict(l=50, r=50, t=100, b=80),  # Minimal margins for full width
            plot_bgcolor='#fafafa',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_3d_session_heatmap(self, data_type='crossover'):
        """Create 3D surface plot of session performance like demo_engines"""
        if data_type == 'crossover' and self.dp.crossover_data is not None:
            df = self.dp.crossover_data
            
            sessions = ['SYDNEY', 'TOKYO', 'LONDON', 'NY']
            distance_labels = ['0-5', '5-10', '10-15', '15-20', '20+']
            distance_bins = [0, 5, 10, 15, 20, float('inf')]
            
            # Calculate matrix for 3D
            matrix_pcts = []
            
            for session in sessions:
                session_mask = (
                    df['Session_Start'].str.contains(session, na=False) |
                    df['Session_Peak'].str.contains(session, na=False) |
                    df['Session_End'].str.contains(session, na=False)
                )
                session_df = df[session_mask]
                
                if len(session_df) == 0:
                    matrix_pcts.append([0] * len(distance_labels))
                else:
                    binned = pd.cut(session_df['Distance'], bins=distance_bins, labels=distance_labels, right=False)
                    counts_series = binned.value_counts().reindex(distance_labels).fillna(0)
                    
                    row_total = sum(counts_series)
                    if row_total > 0:
                        row_pcts = [(c / row_total) * 100.0 for c in counts_series]
                    else:
                        row_pcts = [0.0] * len(distance_labels)
                    matrix_pcts.append(row_pcts)
            
            # Create 3D surface
            fig = go.Figure(data=[go.Surface(
                z=matrix_pcts,
                x=distance_labels,
                y=sessions,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Probability %'),
                contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
            )])
            
            fig.update_layout(
                title=f"3D Session Performance Topography - Total Events: {len(df)}",
                scene=dict(
                    xaxis_title="Distance Range",
                    yaxis_title="Trading Session",
                    zaxis_title="Probability %"
                ),
                template="plotly_dark",
                height=700,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            
            return fig
        
        return None
    
    def create_impulse_vs_reversal_heatmap(self, session_filter=None):
        """Create Impulse vs Reversal heatmap exactly like demo_engines"""
        if self.dp.impulse_data is None:
            return None
            
        df = self.dp.impulse_data.copy()
        
        # Calculate Reversal % from Pullback and Impulse
        df['Reversal%'] = (df['Pullback'] / df['Impulse'] * 100).fillna(0)
        
        # Filter by session if specified
        if session_filter:
            session_mask = (
                df['Session_Entry'].str.contains(session_filter, na=False) |
                df['Session_Peak'].str.contains(session_filter, na=False) |
                df['Session_Trigger'].str.contains(session_filter, na=False)
            )
            df = df[session_mask]
            
        if len(df) == 0:
            return None
        
        # Define Impulse ranges (like demo_engines)
        impulse_ranges = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 50)]
        
        # Define Reversal bins (0-100% in 5% steps like demo_engines)
        reversal_bins = np.arange(0, 105, 5)
        reversal_labels = [f"{int(reversal_bins[i])}-{int(reversal_bins[i+1])}%" for i in range(len(reversal_bins)-1)]
        
        # Calculate matrix data exactly like demo_engines
        matrix_counts = []
        matrix_pcts = []
        text_matrix = []
        custom_data = []
        y_labels = []
        
        for start, end in impulse_ranges:
            # Filter data for this impulse range
            mask = (df['Impulse'] >= start) & (df['Impulse'] < end)
            subset = df[mask]
            
            # Create label with total count (like demo_engines)
            y_labels.append(f"Impulse {start}-{end} (N={len(subset)})")
            
            if len(subset) == 0:
                # Empty row of zeros
                matrix_counts.append([0] * len(reversal_labels))
                matrix_pcts.append([0] * len(reversal_labels))
                text_matrix.append([''] * len(reversal_labels))
                custom_data.append([[0, 0]] * len(reversal_labels))
            else:
                # Bin the Reversal % (exactly like demo_engines)
                counts_series = pd.cut(subset['Reversal%'], bins=reversal_bins, include_lowest=True, right=False).value_counts().sort_index()
                
                # Reindex to ensure all bins are present
                counts_series = counts_series.reindex(pd.cut(pd.Series([0]), bins=reversal_bins, right=False).values.categories).fillna(0)
                
                # Raw counts
                row_counts = counts_series.tolist()
                matrix_counts.append(row_counts)
                
                # Probabilities (Percentage of this row's total)
                row_total = sum(row_counts)
                if row_total > 0:
                    row_pcts = [(c / row_total) * 100.0 for c in row_counts]
                else:
                    row_pcts = [0.0] * len(row_counts)
                matrix_pcts.append(row_pcts)
                
                # Calculate average impulse for each reversal bin (instead of ATR)
                subset['Bin'] = pd.cut(subset['Reversal%'], bins=reversal_bins, include_lowest=True, right=False)
                avg_impulse_series = subset.groupby('Bin')['Impulse'].mean()
                avg_impulse_series = avg_impulse_series.reindex(counts_series.index).fillna(0)
                
                # Create text matrix (exactly like demo_engines format)
                row_text = []
                row_custom = []
                
                for i in range(len(row_counts)):
                    count = int(row_counts[i])
                    pct = row_pcts[i]
                    avg_impulse = avg_impulse_series.iloc[i] if i < len(avg_impulse_series) else 0
                    
                    # Pack custom data: [count, avg_impulse]
                    row_custom.append([count, avg_impulse])
                    
                    if count > 0:
                        # Simple text format - consistent sizing
                        row_text.append(f"{count}<br>({pct:.1f}%)<br>{avg_impulse:.1f}")
                    else:
                        row_text.append("")
                
                text_matrix.append(row_text)
                custom_data.append(row_custom)
        
        # Create heatmap with exact demo_engines styling
        fig = go.Figure(data=go.Heatmap(
            z=matrix_pcts,
            x=reversal_labels,
            y=y_labels,
            # Exact demo_engines colorscale
            colorscale=[
                [0.0, 'white'],       # 0% = White
                [0.01, '#90EE90'],    # >0% = Light Green
                [0.5, 'yellow'],      # 50% = Yellow
                [1.0, 'red']          # 100% = Red
            ],
            reversescale=False,
            zmin=0, zmax=50,       # Cap at 50% for visibility (like demo_engines)
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=14, family="Arial", color="black"),  # Consistent text formatting
            customdata=custom_data,
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>Reversal: %{x}<br>Probability: %{z:.1f}%<br>Count: %{customdata[0]}<br>Avg Impulse: %{customdata[1]:.2f}<extra></extra>'
        ))
        
        # Update layout with consistent centering
        title_suffix = f" - {session_filter} Session" if session_filter else ""
        fig.update_layout(
            title=dict(
                text=f"Impulse vs. Reversal Matrix{title_suffix} - Total Events: {len(df)}",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Reversal % Zone",
            yaxis_title="Impulse Range",
            template="plotly_white",
            height=len(y_labels) * 120 + 300,
            autosize=True,
            xaxis=dict(side="bottom"),
            font=dict(size=12),
            margin=dict(l=50, r=50, t=100, b=80),  # Minimal margins for full width
            plot_bgcolor='#fafafa',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_3d_impulse_reversal_heatmap(self, session_filter=None):
        """Create 3D surface plot of Impulse vs Reversal like demo_engines"""
        if self.dp.impulse_data is None:
            return None
            
        df = self.dp.impulse_data.copy()
        
        # Calculate Reversal % from Pullback and Impulse
        df['Reversal%'] = (df['Pullback'] / df['Impulse'] * 100).fillna(0)
        
        # Filter by session if specified
        if session_filter:
            session_mask = (
                df['Session_Entry'].str.contains(session_filter, na=False) |
                df['Session_Peak'].str.contains(session_filter, na=False) |
                df['Session_Trigger'].str.contains(session_filter, na=False)
            )
            df = df[session_mask]
            
        if len(df) == 0:
            return None
        
        # Define ranges and bins
        impulse_ranges = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 50)]
        reversal_bins = np.arange(0, 105, 5)
        reversal_labels = [f"{int(reversal_bins[i])}-{int(reversal_bins[i+1])}%" for i in range(len(reversal_bins)-1)]
        
        # Calculate matrix for 3D
        matrix_pcts = []
        
        for start, end in impulse_ranges:
            mask = (df['Impulse'] >= start) & (df['Impulse'] < end)
            subset = df[mask]
            
            if len(subset) == 0:
                matrix_pcts.append([0] * len(reversal_labels))
            else:
                counts_series = pd.cut(subset['Reversal%'], bins=reversal_bins, include_lowest=True, right=False).value_counts().sort_index()
                counts_series = counts_series.reindex(pd.cut(pd.Series([0]), bins=reversal_bins, right=False).values.categories).fillna(0)
                
                row_total = sum(counts_series)
                if row_total > 0:
                    row_pcts = [(c / row_total) * 100.0 for c in counts_series]
                else:
                    row_pcts = [0.0] * len(counts_series)
                matrix_pcts.append(row_pcts)
        
        # Create 3D surface (exactly like demo_engines)
        fig = go.Figure(data=[go.Surface(
            z=matrix_pcts,
            x=reversal_labels,
            y=[f"{start}-{end}" for start, end in impulse_ranges],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Probability %'),
            contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        )])
        
        title_suffix = f" - {session_filter} Session" if session_filter else ""
        fig.update_layout(
            title=f"3D Topography: Impulse vs Reversal{title_suffix} - Total Events: {len(df)}",
            scene=dict(
                xaxis_title="Reversal %",
                yaxis_title="Impulse Range",
                zaxis_title="Probability %"
            ),
            template="plotly_dark",
            height=700,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig
    
    def create_individual_session_heatmap(self, session_name, data_type='crossover'):
        """Create heatmap for individual session showing direction vs distance/impulse ranges"""
        if data_type == 'crossover' and self.dp.crossover_data is not None:
            df = self.dp.crossover_data
            
            # Filter data for specific session
            session_mask = (
                df['Session_Start'].str.contains(session_name, na=False) |
                df['Session_Peak'].str.contains(session_name, na=False) |
                df['Session_End'].str.contains(session_name, na=False)
            )
            session_df = df[session_mask]
            
            if len(session_df) == 0:
                return None
            
            # Define distance bins and directions
            distance_bins = [0, 5, 10, 15, 20, float('inf')]
            distance_labels = ['0-5', '5-10', '10-15', '15-20', '20+']
            directions = ['BULLISH', 'BEARISH']
            
            # Calculate matrix data
            matrix_counts = []
            matrix_pcts = []
            text_matrix = []
            custom_data = []
            y_labels = []
            
            for direction in directions:
                direction_df = session_df[session_df['Direction'] == direction]
                y_labels.append(f"{direction} (N={len(direction_df)})")
                
                if len(direction_df) == 0:
                    matrix_counts.append([0] * len(distance_labels))
                    matrix_pcts.append([0] * len(distance_labels))
                    text_matrix.append([''] * len(distance_labels))
                    custom_data.append([[0, 0]] * len(distance_labels))
                else:
                    # Bin the distances
                    binned = pd.cut(direction_df['Distance'], bins=distance_bins, labels=distance_labels, right=False)
                    counts_series = binned.value_counts().reindex(distance_labels).fillna(0)
                    
                    row_counts = counts_series.tolist()
                    matrix_counts.append(row_counts)
                    
                    # Percentages
                    row_total = sum(row_counts)
                    if row_total > 0:
                        row_pcts = [(c / row_total) * 100.0 for c in row_counts]
                    else:
                        row_pcts = [0.0] * len(row_counts)
                    matrix_pcts.append(row_pcts)
                    
                    # Calculate average duration for each bin
                    row_text = []
                    row_custom = []
                    
                    for i, label in enumerate(distance_labels):
                        count = int(row_counts[i])
                        pct = row_pcts[i]
                        
                        if count > 0:
                            # Get average duration for this bin
                            if label == '20+':
                                bin_mask = direction_df['Distance'] >= 20
                            else:
                                start, end = map(float, label.split('-'))
                                bin_mask = (direction_df['Distance'] >= start) & (direction_df['Distance'] < end)
                            
                            bin_df = direction_df[bin_mask]
                            avg_duration = bin_df['Duration_Minutes'].mean() if len(bin_df) > 0 else 0
                            
                            # Simple text format - consistent sizing
                            row_text.append(f"{count}<br>({pct:.1f}%)<br>{avg_duration:.1f}m")
                            row_custom.append([count, avg_duration])
                        else:
                            row_text.append("")
                            row_custom.append([0, 0])
                    
                    text_matrix.append(row_text)
                    custom_data.append(row_custom)
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix_pcts,
                x=distance_labels,
                y=y_labels,
                colorscale=[
                    [0.0, 'white'],
                    [0.01, '#90EE90'],
                    [0.5, 'yellow'],
                    [1.0, 'red']
                ],
                reversescale=False,
                zmin=0, zmax=50,
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=14, family="Arial", color="white"),
                customdata=custom_data,
                hoverongaps=False,
                hovertemplate=f'<b>{session_name} - %{{y}}</b><br>Distance Range: %{{x}}<br>Probability: %{{z:.1f}}%<br>Count: %{{customdata[0]}}<br>Avg Duration: %{{customdata[1]:.1f}}m<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"{session_name} Session: Direction vs Distance Matrix (N={len(session_df)})",
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title="Distance Range (Points)",
                yaxis_title="Direction",
                template="plotly_dark",
                height=len(y_labels) * 100 + 200,
                autosize=True,
                xaxis=dict(side="bottom"),
                font=dict(size=12),
                margin=dict(l=50, r=50, t=100, b=80)
            )
            
            return fig
            
        elif data_type == 'impulse' and self.dp.impulse_data is not None:
            df = self.dp.impulse_data
            
            # Filter data for specific session
            session_mask = (
                df['Session_Entry'].str.contains(session_name, na=False) |
                df['Session_Peak'].str.contains(session_name, na=False) |
                df['Session_Trigger'].str.contains(session_name, na=False)
            )
            session_df = df[session_mask]
            
            if len(session_df) == 0:
                return None
            
            # Define impulse bins and directions
            impulse_bins = [0, 10, 15, 20, 25, float('inf')]
            impulse_labels = ['0-10', '10-15', '15-20', '20-25', '25+']
            directions = ['BULLISH', 'BEARISH']
            
            matrix_counts = []
            matrix_pcts = []
            text_matrix = []
            custom_data = []
            y_labels = []
            
            for direction in directions:
                direction_df = session_df[session_df['Direction'] == direction]
                y_labels.append(f"{direction} (N={len(direction_df)})")
                
                if len(direction_df) == 0:
                    matrix_counts.append([0] * len(impulse_labels))
                    matrix_pcts.append([0] * len(impulse_labels))
                    text_matrix.append([''] * len(impulse_labels))
                    custom_data.append([[0, 0]] * len(impulse_labels))
                else:
                    # Bin the impulses
                    binned = pd.cut(direction_df['Impulse'], bins=impulse_bins, labels=impulse_labels, right=False)
                    counts_series = binned.value_counts().reindex(impulse_labels).fillna(0)
                    
                    row_counts = counts_series.tolist()
                    matrix_counts.append(row_counts)
                    
                    row_total = sum(row_counts)
                    if row_total > 0:
                        row_pcts = [(c / row_total) * 100.0 for c in row_counts]
                    else:
                        row_pcts = [0.0] * len(row_counts)
                    matrix_pcts.append(row_pcts)
                    
                    # Calculate average success rate for each bin
                    row_text = []
                    row_custom = []
                    
                    for i, label in enumerate(impulse_labels):
                        count = int(row_counts[i])
                        pct = row_pcts[i]
                        
                        if count > 0:
                            # Get average success rate for this bin
                            if label == '25+':
                                bin_mask = direction_df['Impulse'] >= 25
                            else:
                                start, end = map(float, label.split('-'))
                                bin_mask = (direction_df['Impulse'] >= start) & (direction_df['Impulse'] < end)
                            
                            bin_df = direction_df[bin_mask]
                            avg_success = bin_df['Success_Rate'].mean() if len(bin_df) > 0 else 0
                            
                            # Simple text format - consistent sizing
                            row_text.append(f"{count}<br>({pct:.1f}%)<br>SR:{avg_success:.1f}%")
                            row_custom.append([count, avg_success])
                        else:
                            row_text.append("")
                            row_custom.append([0, 0])
                    
                    text_matrix.append(row_text)
                    custom_data.append(row_custom)
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix_pcts,
                x=impulse_labels,
                y=y_labels,
                colorscale=[
                    [0.0, 'white'],
                    [0.01, '#90EE90'],
                    [0.5, 'yellow'],
                    [1.0, 'red']
                ],
                reversescale=False,
                zmin=0, zmax=50,
                text=text_matrix,
                texttemplate="%{text}",
                textfont=dict(size=14, family="Arial", color="white"),
                customdata=custom_data,
                hoverongaps=False,
                hovertemplate=f'<b>{session_name} - %{{y}}</b><br>Impulse Range: %{{x}}<br>Probability: %{{z:.1f}}%<br>Count: %{{customdata[0]}}<br>Avg Success Rate: %{{customdata[1]:.1f}}%<extra></extra>'
            ))
            
            fig.update_layout(
                title=dict(
                    text=f"{session_name} Session: Direction vs Impulse Matrix (N={len(session_df)})",
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title="Impulse Range (Points)",
                yaxis_title="Direction",
                template="plotly_dark",
                height=len(y_labels) * 100 + 200,
                autosize=True,
                xaxis=dict(side="bottom"),
                font=dict(size=12),
                margin=dict(l=50, r=50, t=100, b=80)
            )
            
            return fig
        
        return None

class ReportGenerator:
    def __init__(self, data_processor, chart_generator):
        self.dp = data_processor
        self.cg = chart_generator
        
    def generate_summary_stats(self):
        """Generate summary statistics"""
        stats = {}
        
        if self.dp.crossover_data is not None:
            df = self.dp.crossover_data
            stats['crossover'] = {
                'total_trades': len(df),
                'avg_distance': df['Distance'].mean(),
                'avg_duration': df['Duration_Minutes'].mean(),
                'bullish_count': (df['Direction'] == 'BULLISH').sum(),
                'bearish_count': (df['Direction'] == 'BEARISH').sum(),
                'best_session': df.groupby('Session_Start')['Distance'].mean().idxmax() if len(df) > 0 else 'N/A',
                'date_range': f"{df['StartTime'].min().strftime('%Y-%m-%d')} to {df['StartTime'].max().strftime('%Y-%m-%d')}"
            }
        
        if self.dp.impulse_data is not None:
            df = self.dp.impulse_data
            
            # Calculate best threshold for overall impulse data
            threshold_cols = [col for col in df.columns if col.startswith('Hit_')]
            best_threshold = 'N/A'
            if threshold_cols:
                threshold_rates = []
                for col in threshold_cols:
                    rate = df[col].mean() * 100
                    threshold_rates.append({'Threshold': col.replace('Hit_', ''), 'Rate': rate})
                
                tr_df = pd.DataFrame(threshold_rates)
                if not tr_df.empty:
                    best_threshold = tr_df.loc[tr_df['Rate'].idxmax(), 'Threshold']

            stats['impulse'] = {
                'total_impulses': len(df),
                'avg_impulse': df['Impulse'].mean(),
                'avg_success_rate': df['Success_Rate'].mean(),
                'best_segment': df.groupby('Segment')['Success_Rate'].mean().idxmax() if len(df) > 0 else 'N/A',
                'avg_pullback_pct': df['Pullback_Pct'].mean(),
                'best_threshold': best_threshold
            }
        
        return stats
    
    def generate_insights(self, stats):
        """Generate AI-like insights"""
        insights = []
        
        if 'crossover' in stats:
            cs = stats['crossover']
            
            # Direction bias insight
            if cs['bullish_count'] > cs['bearish_count'] * 1.2:
                insights.append(f"📈 Strong bullish bias detected: {cs['bullish_count']} bullish vs {cs['bearish_count']} bearish trades")
            elif cs['bearish_count'] > cs['bullish_count'] * 1.2:
                insights.append(f"📉 Strong bearish bias detected: {cs['bearish_count']} bearish vs {cs['bullish_count']} bullish trades")
            else:
                insights.append(f"⚖️ Balanced market: {cs['bullish_count']} bullish vs {cs['bearish_count']} bearish trades")
            
            # Performance insight
            if cs['avg_distance'] > 10:
                insights.append(f"🎯 High performance detected: Average distance of {cs['avg_distance']:.2f} points")
            elif cs['avg_distance'] < 5:
                insights.append(f"⚠️ Low performance: Average distance of {cs['avg_distance']:.2f} points - consider strategy adjustment")
            
            # Duration insight
            if cs['avg_duration'] < 30:
                insights.append(f"⚡ Fast trades: Average duration of {cs['avg_duration']:.1f} minutes")
            elif cs['avg_duration'] > 120:
                insights.append(f"🐌 Slow trades: Average duration of {cs['avg_duration']:.1f} minutes")
        
        if 'impulse' in stats:
            ins = stats['impulse']
            
            # Success rate insight
            if ins['avg_success_rate'] > 60:
                insights.append(f"✅ Excellent threshold performance: {ins['avg_success_rate']:.1f}% average success rate")
            elif ins['avg_success_rate'] < 30:
                insights.append(f"❌ Poor threshold performance: {ins['avg_success_rate']:.1f}% average success rate")
            
            # Pullback insight
            if ins['avg_pullback_pct'] > 50:
                insights.append(f"⚠️ High pullback risk: Average {ins['avg_pullback_pct']:.1f}% pullback")
        
        return insights

def format_number(num):
    """Format numbers for display"""
    if pd.isna(num):
        return "N/A"
    if isinstance(num, (int, float)):
        return f"{num:,.2f}" if num != int(num) else f"{int(num):,}"
    return str(num)