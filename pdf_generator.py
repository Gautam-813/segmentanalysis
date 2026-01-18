from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import io
import base64
from datetime import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class PDFReportGenerator:
    def __init__(self, data_processor, chart_generator, report_generator):
        self.dp = data_processor
        self.cg = chart_generator
        self.rg = report_generator
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            alignment=TA_LEFT
        ))
        
        self.styles.add(ParagraphStyle(
            name='InsightStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leftIndent=20,
            textColor=colors.darkgreen,
            backColor=colors.lightgrey
        ))
    
    def save_plotly_as_image(self, fig, filename):
        """Save plotly figure as image"""
        try:
            # Try using kaleido for image export
            img_bytes = fig.to_image(format="png", width=800, height=400, scale=2)
            with open(filename, "wb") as f:
                f.write(img_bytes)
            return filename
        except Exception as e:
            print(f"Error saving plotly image: {e}")
            # Fallback: create a simple matplotlib placeholder
            try:
                import matplotlib.pyplot as plt
                fig_plt, ax = plt.subplots(figsize=(8, 4))
                ax.text(0.5, 0.5, 'Chart Preview\n(Install kaleido for full charts)', 
                       ha='center', va='center', fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
                return filename
            except Exception as e2:
                print(f"Error creating fallback image: {e2}")
                return None
    
    def create_summary_table(self, stats):
        """Create summary statistics table"""
        data = [['Metric', 'Value']]
        
        if 'crossover' in stats:
            cs = stats['crossover']
            data.extend([
                ['Total Crossover Trades', f"{cs['total_trades']:,}"],
                ['Average Distance', f"{cs['avg_distance']:.2f}"],
                ['Average Duration (min)', f"{cs['avg_duration']:.1f}"],
                ['Bullish Trades', f"{cs['bullish_count']:,}"],
                ['Bearish Trades', f"{cs['bearish_count']:,}"],
                ['Analysis Period', cs['date_range']]
            ])
        
        if 'impulse' in stats:
            ins = stats['impulse']
            data.extend([
                ['Total Impulse Trades', f"{ins['total_impulses']:,}"],
                ['Average Impulse', f"{ins['avg_impulse']:.2f}"],
                ['Average Success Rate', f"{ins['avg_success_rate']:.1f}%"],
                ['Average Pullback %', f"{ins['avg_pullback_pct']:.1f}%"]
            ])
        
        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        return table
    
    def create_data_table(self, df, title, max_rows=20):
        """Create data table from DataFrame"""
        if df is None or len(df) == 0:
            return None
            
        # Select key columns and limit rows
        if 'StartTime' in df.columns:  # Crossover data
            cols = ['StartTime', 'Direction', 'Distance', 'Segment', 'Session_Start']
            df_display = df[cols].head(max_rows).copy()
            df_display['StartTime'] = df_display['StartTime'].dt.strftime('%Y-%m-%d %H:%M')
        else:  # Impulse data
            cols = ['Time', 'Direction', 'Impulse', 'Success_Rate', 'Segment']
            df_display = df[cols].head(max_rows).copy()
            df_display['Time'] = df_display['Time'].dt.strftime('%Y-%m-%d %H:%M')
            df_display['Success_Rate'] = df_display['Success_Rate'].round(1)
        
        # Convert to list for table
        data = [df_display.columns.tolist()]
        for _, row in df_display.iterrows():
            data.append([str(val) for val in row.tolist()])
        
        table = Table(data, colWidths=[1.2*inch] * len(df_display.columns))
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
        ]))
        
        return table
    
    def generate_pdf_report(self, report_type='detailed'):
        """Generate PDF report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Title
        title = Paragraph("EA Trading Analysis Report", self.styles['CustomTitle'])
        elements.append(title)
        elements.append(Spacer(1, 20))
        
        # Report info
        report_info = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Report Type:</b> {report_type.title()}<br/>
        <b>Analysis Period:</b> {self.get_analysis_period()}
        """
        elements.append(Paragraph(report_info, self.styles['CustomNormal']))
        elements.append(Spacer(1, 20))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
        stats = self.rg.generate_summary_stats()
        
        if stats:
            summary_table = self.create_summary_table(stats)
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
        
        # Key Insights
        elements.append(Paragraph("Key Insights", self.styles['CustomHeading']))
        insights = self.rg.generate_insights(stats)
        
        for insight in insights:
            elements.append(Paragraph(f"• {insight}", self.styles['InsightStyle']))
        elements.append(Spacer(1, 20))
        
        if report_type == 'detailed':
            # Add charts (as images)
            elements.append(PageBreak())
            elements.append(Paragraph("Performance Analysis", self.styles['CustomHeading']))
            
            # Session Heatmap
            try:
                fig = self.cg.create_session_heatmap('crossover')
                if fig:
                    img_path = "temp_session_heatmap.png"
                    self.save_plotly_as_image(fig, img_path)
                    elements.append(Image(img_path, width=6*inch, height=3*inch))
                    elements.append(Spacer(1, 10))
            except Exception as e:
                elements.append(Paragraph(f"Chart generation error: {str(e)}", self.styles['CustomNormal']))
            
            # Segment Distribution
            try:
                fig = self.cg.create_segment_distribution()
                if fig:
                    img_path = "temp_segment_dist.png"
                    self.save_plotly_as_image(fig, img_path)
                    elements.append(Image(img_path, width=6*inch, height=3*inch))
                    elements.append(Spacer(1, 10))
            except Exception as e:
                elements.append(Paragraph(f"Chart generation error: {str(e)}", self.styles['CustomNormal']))
            
            # Data Tables
            elements.append(PageBreak())
            elements.append(Paragraph("Recent Trades Data", self.styles['CustomHeading']))
            
            if self.dp.crossover_data is not None:
                elements.append(Paragraph("Crossover Trades (Latest 20)", self.styles['CustomHeading']))
                table = self.create_data_table(self.dp.crossover_data, "Crossover Data")
                if table:
                    elements.append(table)
                elements.append(Spacer(1, 20))
            
            if self.dp.impulse_data is not None:
                elements.append(Paragraph("Impulse Trades (Latest 20)", self.styles['CustomHeading']))
                table = self.create_data_table(self.dp.impulse_data, "Impulse Data")
                if table:
                    elements.append(table)
        
        # Recommendations
        elements.append(PageBreak())
        elements.append(Paragraph("Recommendations", self.styles['CustomHeading']))
        
        recommendations = self.generate_recommendations(stats)
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", self.styles['CustomNormal']))
        
        # Footer
        elements.append(Spacer(1, 30))
        footer_text = """
        <i>This report is generated automatically based on EA trading data analysis. 
        Please review all recommendations with your trading strategy before implementation.</i>
        """
        elements.append(Paragraph(footer_text, self.styles['CustomNormal']))
        
        # Build PDF
        doc.build(elements)
        
        # Get the value of the BytesIO buffer and return it
        pdf_data = buffer.getvalue()
        buffer.close()
        
        return pdf_data
    
    def get_analysis_period(self):
        """Get analysis period string"""
        if self.dp.crossover_data is not None:
            df = self.dp.crossover_data
            start_date = df['StartTime'].min().strftime('%Y-%m-%d')
            end_date = df['StartTime'].max().strftime('%Y-%m-%d')
            return f"{start_date} to {end_date}"
        elif self.dp.impulse_data is not None:
            df = self.dp.impulse_data
            start_date = df['Time'].min().strftime('%Y-%m-%d')
            end_date = df['Time'].max().strftime('%Y-%m-%d')
            return f"{start_date} to {end_date}"
        return "N/A"
    
    def generate_recommendations(self, stats):
        """Generate trading recommendations"""
        recommendations = []
        
        if 'crossover' in stats:
            cs = stats['crossover']
            
            # Direction recommendation
            if cs['bullish_count'] > cs['bearish_count'] * 1.5:
                recommendations.append("Consider focusing on bullish setups given the strong upward bias in recent data")
            elif cs['bearish_count'] > cs['bullish_count'] * 1.5:
                recommendations.append("Consider focusing on bearish setups given the strong downward bias in recent data")
            
            # Performance recommendation
            if cs['avg_distance'] < 5:
                recommendations.append("Average distance is low - consider adjusting entry criteria or market selection")
            
            # Duration recommendation
            if cs['avg_duration'] > 180:
                recommendations.append("Trades are taking longer than average - consider tighter stop losses or profit targets")
        
        if 'impulse' in stats:
            ins = stats['impulse']
            
            if ins['avg_success_rate'] < 40:
                recommendations.append("Threshold success rate is below optimal - consider adjusting reversal thresholds")
            
            if ins['avg_pullback_pct'] > 60:
                recommendations.append("High pullback percentages detected - consider earlier exit strategies")
        
        # General recommendations
        recommendations.extend([
            "Monitor session performance and focus trading during high-performance periods",
            "Review segment analysis to identify optimal distance ranges for entry",
            "Consider risk management adjustments based on recent performance metrics"
        ])
        
        return recommendations