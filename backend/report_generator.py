from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import pandas as pd
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_pdf_report(original_df, cleaned_df, metrics, target_column, output_dir):
    """Generate comprehensive PDF report"""
    Path(output_dir).mkdir(exist_ok=True)
    report_path = Path(output_dir) / "ml_report.pdf"
    
    doc = SimpleDocTemplate(str(report_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#0ea5e9'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("AutoML Studio Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # 1. Dataset Summary
    story.append(Paragraph("1. Dataset Summary", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    if original_df is not None:
        summary_data = [
            ["Metric", "Value"],
            ["Total Rows", f"{len(original_df):,}"],
            ["Total Columns", f"{len(original_df.columns)}"],
            ["Numeric Columns", f"{len(original_df.select_dtypes(include=[np.number]).columns)}"],
            ["Categorical Columns", f"{len(original_df.select_dtypes(exclude=[np.number]).columns)}"],
            ["Missing Values", f"{original_df.isnull().sum().sum():,}"],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.2*inch))
    
    # 2. Data Cleaning Steps
    if cleaned_df is not None:
        story.append(Paragraph("2. Data Cleaning", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            f"Dataset was cleaned and preprocessed. Final dataset: {len(cleaned_df):,} rows, {len(cleaned_df.columns)} columns.",
            styles['Normal']
        ))
        story.append(Spacer(1, 0.2*inch))
    
    # 3. Model Metrics
    if metrics:
        story.append(Paragraph("3. Model Performance Metrics", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        metrics_data = [
            ["Metric", "Value"],
            ["Accuracy", f"{metrics.get('accuracy', 0):.4f}"],
            ["Precision", f"{metrics.get('precision', 0):.4f}"],
            ["Recall", f"{metrics.get('recall', 0):.4f}"],
            ["F1 Score", f"{metrics.get('f1', 0):.4f}"],
        ]
        
        if metrics.get('auc'):
            metrics_data.append(["AUC Score", f"{metrics['auc']:.4f}"])
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 3*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.2*inch))
    
    # 4. AI Insights Summary
    story.append(Paragraph("4. AI Insights & Recommendations", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    insights_text = """
    This AutoML analysis provides a comprehensive view of your dataset and model performance.
    The model has been trained and evaluated using industry-standard metrics.
    
    Key Takeaways:
    • The dataset has been automatically cleaned and preprocessed
    • Multiple visualization charts have been generated for EDA
    • Model performance metrics indicate the effectiveness of the chosen algorithm
    • Feature importance analysis helps identify the most predictive variables
    
    Next Steps:
    • Consider hyperparameter tuning for improved performance
    • Experiment with different model types
    • Collect more data if accuracy is below expectations
    • Perform feature engineering to create more informative features
    """
    
    story.append(Paragraph(insights_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Footer
    story.append(Spacer(1, 0.5*inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Generated by AutoML Studio", footer_style))
    
    doc.build(story)
    return report_path

